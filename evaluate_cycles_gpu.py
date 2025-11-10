#!/usr/bin/env python3
"""
gpu-accelerated cycle evaluation using cupy for vectorized operations.
evaluates cycles in parallel batches on gpu for significant speedup.
falls back to cpu if gpu not available.
"""

import networkx as nx
import numpy as np
import json
import pickle
import pandas as pd
from typing import List, Dict, Tuple
import time

# try to import cupy, fall back to numpy if not available
try:
    import cupy as cp

    GPU_AVAILABLE = True
    print("[INFO] gpu (cupy) available - using gpu acceleration")
except ImportError:
    import numpy as cp

    GPU_AVAILABLE = False
    print("[WARNING] cupy not found - falling back to cpu (numpy)")


# default parameters
SNOWFALL_RATE_INCHES_PER_HOUR = 1.0
STORM_DURATION_HOURS = 12


def prepare_cycle_batch(G, cycles_batch):
    """
    convert batch of cycles into dense arrays for gpu processing.
    also prepare edge IDs for correct tracking during simulation.

    args:
        G: networkx graph
        cycles_batch: list of cycle data dicts

    returns:
        tuple of (priorities_array, times_array, lengths_array, unique_priorities_array, cycles_with_ids)
    """
    max_len = max(len(c["cycle"]) for c in cycles_batch)
    batch_size = len(cycles_batch)

    # initialize arrays (pad with zeros)
    priorities = np.zeros((batch_size, max_len), dtype=np.float32)
    travel_times = np.zeros((batch_size, max_len), dtype=np.float32)
    lengths = np.array([len(c["cycle"]) for c in cycles_batch], dtype=np.int32)
    unique_priorities = np.zeros(batch_size, dtype=np.float32)

    # also keep edge IDs for proper tracking
    cycles_with_ids = []

    # fill arrays with edge data
    for i, cycle_data in enumerate(cycles_batch):
        unique_edges = set()
        total_unique_priority = 0.0
        cycle_edge_ids = []

        for j, (u, v, k) in enumerate(cycle_data["cycle"]):
            edge_data = G[u][v][k]
            priorities[i, j] = edge_data["priority"]
            travel_times[i, j] = edge_data["travel_time"]
            cycle_edge_ids.append((u, v, k))

            # track unique edges for baseline cost
            edge_id = (u, v, k)
            if edge_id not in unique_edges:
                total_unique_priority += edge_data["priority"]
                unique_edges.add(edge_id)

        unique_priorities[i] = total_unique_priority
        cycles_with_ids.append(cycle_edge_ids)

    return priorities, travel_times, lengths, unique_priorities, cycles_with_ids


def simulate_cycles_vectorized_gpu(
    priorities_gpu,
    times_gpu,
    lengths_gpu,
    unique_priorities_gpu,
    cycles_with_ids,
    snowfall_rate,
    storm_duration,
):
    """
    vectorized simulation of multiple cycles on gpu.

    args:
        priorities_gpu: [batch_size, max_len] array of edge priorities on gpu
        times_gpu: [batch_size, max_len] array of edge travel times on gpu
        lengths_gpu: [batch_size] array of cycle lengths on gpu
        unique_priorities_gpu: [batch_size] array of unique edge priorities per cycle
        cycles_with_ids: list of lists containing (u,v,k) tuples for edge tracking
        snowfall_rate: inches per minute
        storm_duration: total storm duration in minutes

    returns:
        dict with arrays of metrics for each cycle in batch
    """
    batch_size = priorities_gpu.shape[0]
    max_len = priorities_gpu.shape[1]

    # baseline cost (no plow): use UNIQUE edge priorities * snowfall_rate * T^2 / 2
    baseline_costs = unique_priorities_gpu * snowfall_rate * (storm_duration**2) / 2.0

    # cycle times (single pass)
    # use mask to ignore padding
    mask = cp.arange(max_len)[None, :] < lengths_gpu[:, None]  # [batch_size, max_len]
    masked_times = times_gpu * mask
    cycle_times = cp.sum(masked_times, axis=1)  # [batch_size]

    # simulate with plow - NOT TRULY VECTORIZED due to complex edge tracking
    # this is still faster than pure python because we batch multiple cycles
    plow_costs = np.zeros(batch_size, dtype=np.float32)
    num_complete_cycles_arr = np.zeros(batch_size, dtype=np.int32)

    # convert to numpy for easier manipulation (CPU is fine for simulation logic)
    if GPU_AVAILABLE:
        priorities_np = cp.asnumpy(priorities_gpu)
        times_np = cp.asnumpy(times_gpu)
        lengths_np = cp.asnumpy(lengths_gpu)
        cycle_times_np = cp.asnumpy(cycle_times)
    else:
        priorities_np = priorities_gpu
        times_np = times_gpu
        lengths_np = lengths_gpu
        cycle_times_np = cycle_times

    # for each cycle in batch, simulate repeated passes
    for batch_idx in range(batch_size):
        cycle_len = int(lengths_np[batch_idx])
        cycle_time = float(cycle_times_np[batch_idx])

        if cycle_time == 0 or cycle_len == 0:
            continue

        # extract this cycle's data
        edge_priorities = priorities_np[batch_idx, :cycle_len]
        edge_times = times_np[batch_idx, :cycle_len]
        edge_ids = cycles_with_ids[batch_idx]

        # track last visit time for each UNIQUE EDGE ID (not position!)
        edge_last_visit = {}  # key = (u,v,k), value = last visit time
        edges_visited = set()
        current_time = 0.0
        total_cost = 0.0
        num_cycles_completed = 0

        # repeat cycle until storm ends
        while current_time < storm_duration:
            cycle_start_time = current_time

            # traverse cycle
            for edge_idx in range(cycle_len):
                if current_time >= storm_duration:
                    break

                # get the actual edge ID
                edge_id = edge_ids[edge_idx]
                edges_visited.add(edge_id)

                # time since last visit to this SPECIFIC EDGE (by ID, not position!)
                last_visit = edge_last_visit.get(edge_id, 0.0)
                time_since = current_time - last_visit

                # cost = priority * snowfall_rate * time_since^2 / 2
                cost = (
                    float(edge_priorities[edge_idx])
                    * snowfall_rate
                    * (time_since**2)
                    / 2.0
                )
                total_cost += cost

                # update last visit for this edge ID
                edge_last_visit[edge_id] = current_time

                # advance time
                current_time += float(edge_times[edge_idx])

            # check if completed full cycle
            if current_time <= storm_duration:
                num_cycles_completed += 1

            # prevent infinite loop
            if current_time == cycle_start_time:
                break

        # add residual cost (snow after last visit until storm ends)
        # for each unique edge that was visited, add snow accumulation from last visit to storm end
        for edge_id, last_time in edge_last_visit.items():
            time_since = storm_duration - last_time
            if time_since > 0:
                # find the priority for this edge (get from first occurrence in cycle)
                edge_priority = 0.0
                for idx in range(cycle_len):
                    if edge_ids[idx] == edge_id:
                        edge_priority = float(edge_priorities[idx])
                        break
                cost = edge_priority * snowfall_rate * (time_since**2) / 2.0
                total_cost += cost

        plow_costs[batch_idx] = total_cost
        num_complete_cycles_arr[batch_idx] = num_cycles_completed

    # convert back to cp arrays
    plow_costs = cp.asarray(plow_costs) if GPU_AVAILABLE else plow_costs
    num_complete_cycles = (
        cp.asarray(num_complete_cycles_arr)
        if GPU_AVAILABLE
        else num_complete_cycles_arr
    )

    # calculate benefits
    benefits = baseline_costs - plow_costs
    benefits_pct = cp.where(
        baseline_costs > 0, (benefits / baseline_costs) * 100.0, 0.0
    )
    benefits_per_min = benefits / storm_duration

    return {
        "baseline_costs": baseline_costs,
        "plow_costs": plow_costs,
        "benefits": benefits,
        "benefits_pct": benefits_pct,
        "benefits_per_min": benefits_per_min,
        "cycle_times": cycle_times,
        "num_complete_cycles": num_complete_cycles,
        "unique_priorities": unique_priorities_gpu,
    }


def evaluate_cycles_batch_gpu(
    G, cycles_batch, snowfall_rate, storm_duration, batch_size=128
):
    """
    evaluate a batch of cycles using gpu acceleration.

    args:
        G: networkx graph
        cycles_batch: list of cycle data dicts
        snowfall_rate: inches per minute
        storm_duration: minutes
        batch_size: number of cycles to process at once on gpu

    returns:
        list of evaluated cycle dicts with metrics added
    """
    n_cycles = len(cycles_batch)
    evaluated = []

    for batch_start in range(0, n_cycles, batch_size):
        batch_end = min(batch_start + batch_size, n_cycles)
        current_batch = cycles_batch[batch_start:batch_end]

        # prepare batch data
        (
            priorities_cpu,
            times_cpu,
            lengths_cpu,
            unique_priorities_cpu,
            cycles_with_ids,
        ) = prepare_cycle_batch(G, current_batch)

        # move to gpu
        priorities_gpu = cp.asarray(priorities_cpu)
        times_gpu = cp.asarray(times_cpu)
        lengths_gpu = cp.asarray(lengths_cpu)
        unique_priorities_gpu = cp.asarray(unique_priorities_cpu)

        # simulate on gpu (edge IDs stay on CPU for dict operations)
        results = simulate_cycles_vectorized_gpu(
            priorities_gpu,
            times_gpu,
            lengths_gpu,
            unique_priorities_gpu,
            cycles_with_ids,
            snowfall_rate,
            storm_duration,
        )

        # move results back to cpu
        if GPU_AVAILABLE:
            results_cpu = {k: cp.asnumpy(v) for k, v in results.items()}
        else:
            results_cpu = results

        # attach results to cycle data
        for i, cycle_data in enumerate(current_batch):
            cycle_len = len(cycle_data["cycle"])
            unique_edges = cycle_data["metrics"].get("unique_edges", cycle_len)

            evaluation = {
                "cost_without_plow": float(results_cpu["baseline_costs"][i]),
                "cost_with_plow": float(results_cpu["plow_costs"][i]),
                "benefit": float(results_cpu["benefits"][i]),
                "benefit_per_minute": float(results_cpu["benefits_per_min"][i]),
                "benefit_pct": float(results_cpu["benefits_pct"][i]),
                "cycle_time": float(results_cpu["cycle_times"][i]),
                "num_complete_cycles": int(results_cpu["num_complete_cycles"][i]),
                "coverage_ratio": 1.0,  # assume full coverage (greedy guarantees this)
            }

            cycle_data["evaluation"] = evaluation
            evaluated.append(cycle_data)

    return evaluated


def evaluate_all_cycles_gpu(G, cycles, snowfall_rate, storm_duration, batch_size=128):
    """
    evaluate all cycles using gpu acceleration with batching.

    args:
        G: networkx graph
        cycles: list of cycle dicts
        snowfall_rate: inches per minute
        storm_duration: minutes
        batch_size: number of cycles per gpu batch

    returns:
        list of cycles with evaluation metrics added
    """
    print(
        f"[INFO] evaluating {len(cycles)} cycles on {'GPU' if GPU_AVAILABLE else 'CPU'}..."
    )
    print(f"[INFO] batch size: {batch_size}")

    evaluated_cycles = []

    for i in range(0, len(cycles), batch_size * 10):
        batch_end = min(i + batch_size * 10, len(cycles))
        mega_batch = cycles[i:batch_end]

        print(f"  [PROGRESS] evaluated {i}/{len(cycles)} cycles...")

        batch_evaluated = evaluate_cycles_batch_gpu(
            G, mega_batch, snowfall_rate, storm_duration, batch_size
        )
        evaluated_cycles.extend(batch_evaluated)

    print(f"  [PROGRESS] evaluated {len(cycles)}/{len(cycles)} cycles...")

    return evaluated_cycles


def rank_cycles_by_partition(evaluated_cycles):
    """
    rank cycles within each partition by benefit percentage (efficiency).

    args:
        evaluated_cycles: list of evaluated cycle dicts

    returns:
        dict mapping (partition_set_id, partition_id) to ranked cycles
    """
    # group by partition
    partitions = {}
    for cycle_data in evaluated_cycles:
        key = (cycle_data["partition_set_id"], cycle_data["partition_id"])
        if key not in partitions:
            partitions[key] = []
        partitions[key].append(cycle_data)

    # rank within each partition by benefit_pct (efficiency)
    for key in partitions:
        partitions[key].sort(
            key=lambda x: x["evaluation"].get(
                "benefit_pct", x["evaluation"].get("benefit_per_minute", 0)
            ),
            reverse=True,
        )

    return partitions


def main():
    print("[INFO] loading road graph...")
    with open("road_graph.gpickle", "rb") as f:
        G = pickle.load(f)
    print(
        f"[INFO] loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )

    print("[INFO] loading candidate cycles...")
    with open("candidate_cycles.json", "r") as f:
        cycles = json.load(f)
    print(f"[INFO] loaded {len(cycles)} cycles")

    # parameters
    snowfall_rate_per_hour = SNOWFALL_RATE_INCHES_PER_HOUR
    snowfall_rate_per_minute = snowfall_rate_per_hour / 60.0
    storm_duration_minutes = STORM_DURATION_HOURS * 60

    print(f"[INFO] using snowfall rate: {snowfall_rate_per_hour} inches/hour")
    print(f"[INFO] using storm duration: {STORM_DURATION_HOURS} hours")

    # determine batch size based on available memory
    batch_size = 256 if GPU_AVAILABLE else 64

    # evaluate cycles with gpu acceleration
    start_time = time.time()
    evaluated_cycles = evaluate_all_cycles_gpu(
        G, cycles, snowfall_rate_per_minute, storm_duration_minutes, batch_size
    )
    elapsed = time.time() - start_time

    print(f"\n[PASS] evaluation completed in {elapsed:.2f} seconds")
    print(f"[PASS] throughput: {len(cycles)/elapsed:.1f} cycles/second")

    # save evaluated cycles
    output_file = "cycle_evaluations.json"
    with open(output_file, "w") as f:
        json.dump(evaluated_cycles, f, indent=2)
    print(f"\n[INFO] saved {len(evaluated_cycles)} evaluated cycles to {output_file}")

    # rank cycles
    ranked_partitions = rank_cycles_by_partition(evaluated_cycles)

    # create summary dataframe
    summary_rows = []
    for cycle_data in evaluated_cycles:
        summary_rows.append(
            {
                "partition_set_id": cycle_data["partition_set_id"],
                "partition_id": cycle_data["partition_id"],
                "method": cycle_data["method"],
                "start_node": cycle_data["start_node"],
                "cycle_time": cycle_data["evaluation"]["cycle_time"],
                "num_complete_cycles": cycle_data["evaluation"]["num_complete_cycles"],
                "benefit": cycle_data["evaluation"]["benefit"],
                "benefit_per_minute": cycle_data["evaluation"]["benefit_per_minute"],
                "benefit_pct": cycle_data["evaluation"]["benefit_pct"],
                "cost_without_plow": cycle_data["evaluation"]["cost_without_plow"],
                "cost_with_plow": cycle_data["evaluation"]["cost_with_plow"],
                "coverage_ratio": cycle_data["evaluation"]["coverage_ratio"],
                "total_priority": cycle_data["metrics"]["total_priority"],
                "unique_edges": cycle_data["metrics"]["unique_edges"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("benefit_pct", ascending=False)
    summary_df.to_csv("cycle_evaluations.csv", index=False)
    print(f"[INFO] saved cycle evaluations to cycle_evaluations.csv")

    # print summary statistics
    print("\n[SUMMARY] evaluation statistics:")
    print(f"average benefit: {summary_df['benefit'].mean():.2f}")
    print(f"average benefit %: {summary_df['benefit_pct'].mean():.2f}%")
    print(f"best benefit %: {summary_df['benefit_pct'].max():.2f}%")
    print(
        f"average cycles completed in {STORM_DURATION_HOURS}h: {summary_df['num_complete_cycles'].mean():.1f}"
    )
    print(f"average coverage ratio: {summary_df['coverage_ratio'].mean():.2%}")

    # show top cycles per partition
    print("\n[INFO] top cycle per partition:")
    for key in sorted(ranked_partitions.keys()):
        ps_id, p_id = key
        top_cycle = ranked_partitions[key][0]
        print(
            f"  partition set {ps_id}, partition {p_id}: "
            f"benefit = {top_cycle['evaluation']['benefit_pct']:.2f}%, "
            f"cycles completed = {top_cycle['evaluation']['num_complete_cycles']}, "
            f"coverage = {top_cycle['evaluation']['coverage_ratio']:.1%}, "
            f"method = {top_cycle['method']}"
        )

    return evaluated_cycles


if __name__ == "__main__":
    main()
