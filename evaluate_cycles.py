#!/usr/bin/env python3
"""
evaluate cycles based on cost function: cost = integral(priority * snow_depth) dt
compare cost with plow vs without plow to measure benefit.
"""

import networkx as nx
import numpy as np
import json
import pickle
import pandas as pd


# default parameters
SNOWFALL_RATE_INCHES_PER_HOUR = 1.0
STORM_DURATION_HOURS = 12


def calculate_cost_without_plow(edge_priority, snowfall_rate, time_duration):
    """
    calculate cost accumulated if no plow visits.
    snow accumulates linearly, so integral of priority * snow over time.

    cost = priority * snowfall_rate * integral(t dt from 0 to T)
         = priority * snowfall_rate * (T^2 / 2)

    args:
        edge_priority: priority of the edge
        snowfall_rate: inches per minute
        time_duration: minutes

    returns:
        cost (priority * snow * time units)
    """
    # integral of linear accumulation: (rate * t) from 0 to T = rate * T^2 / 2
    cost = edge_priority * snowfall_rate * (time_duration**2) / 2.0
    return cost


def calculate_cost_with_plow(G, cycle, snowfall_rate, start_time=0):
    """
    calculate cost accumulated if plow follows the cycle.
    for each edge, snow accumulates from last visit (or start) until plow arrives.

    args:
        G: networkx graph
        cycle: list of (from_node, to_node, edge_key) tuples
        snowfall_rate: inches per minute
        start_time: when plow starts (minutes from storm start)

    returns:
        total cost
    """
    # track last visit time for each edge
    edge_last_visit = {}
    current_time = start_time
    total_cost = 0.0

    for from_node, to_node, edge_key in cycle:
        edge_data = G[from_node][to_node][edge_key]
        edge_id = (from_node, to_node, edge_key)

        # calculate cost since last visit (or start)
        last_visit = edge_last_visit.get(edge_id, start_time)
        time_since_visit = current_time - last_visit

        # snow accumulated = snowfall_rate * time_since_visit
        # cost = priority * integral(snowfall_rate * t dt from 0 to time_since_visit)
        #      = priority * snowfall_rate * time_since_visit^2 / 2
        cost = edge_data["priority"] * snowfall_rate * (time_since_visit**2) / 2.0
        total_cost += cost

        # update last visit time
        edge_last_visit[edge_id] = current_time

        # advance time by travel time
        current_time += edge_data["travel_time"]

    return total_cost


def calculate_cycle_benefit(G, cycle, snowfall_rate, storm_duration):
    """
    calculate benefit of using this cycle vs not plowing.

    benefit = cost_without_plow - cost_with_plow

    args:
        G: networkx graph
        cycle: list of (from_node, to_node, edge_key) tuples
        snowfall_rate: inches per minute
        storm_duration: total storm duration in minutes

    returns:
        dict with cost metrics and benefit
    """
    if not cycle:
        return {
            "cost_without_plow": 0.0,
            "cost_with_plow": 0.0,
            "benefit": 0.0,
            "benefit_per_minute": 0.0,
        }

    # calculate total priority of edges in cycle
    unique_edges = set()
    total_priority = 0.0

    for from_node, to_node, edge_key in cycle:
        edge_id = (from_node, to_node, edge_key)
        if edge_id not in unique_edges:
            edge_data = G[from_node][to_node][edge_key]
            total_priority += edge_data["priority"]
            unique_edges.add(edge_id)

    # cost without plow: each edge accumulates for full storm duration
    cost_without_plow = calculate_cost_without_plow(
        total_priority, snowfall_rate, storm_duration
    )

    # cost with plow: follow the cycle
    cost_with_plow = calculate_cost_with_plow(G, cycle, snowfall_rate, start_time=0)

    # benefit
    benefit = cost_without_plow - cost_with_plow

    # calculate cycle time
    cycle_time = sum(G[u][v][k]["travel_time"] for u, v, k in cycle)

    # benefit per minute
    benefit_per_minute = benefit / cycle_time if cycle_time > 0 else 0.0

    return {
        "cost_without_plow": float(cost_without_plow),
        "cost_with_plow": float(cost_with_plow),
        "benefit": float(benefit),
        "benefit_per_minute": float(benefit_per_minute),
        "cycle_time": float(cycle_time),
    }


def evaluate_all_cycles(G, cycles, snowfall_rate, storm_duration):
    """
    evaluate all cycles and add evaluation metrics.

    args:
        G: networkx graph
        cycles: list of cycle dicts
        snowfall_rate: inches per minute
        storm_duration: minutes

    returns:
        list of cycles with evaluation metrics added
    """
    print("[INFO] evaluating cycles...")

    evaluated_cycles = []

    for i, cycle_data in enumerate(cycles):
        if i % 100 == 0:
            print(f"  [PROGRESS] evaluated {i}/{len(cycles)} cycles...")

        cycle = cycle_data["cycle"]

        # calculate benefit
        benefit_metrics = calculate_cycle_benefit(
            G, cycle, snowfall_rate, storm_duration
        )

        # add to cycle data
        cycle_data["evaluation"] = benefit_metrics
        evaluated_cycles.append(cycle_data)

    return evaluated_cycles


def rank_cycles_by_partition(evaluated_cycles):
    """
    rank cycles within each partition by benefit per minute.

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

    # rank within each partition
    for key in partitions:
        partitions[key].sort(
            key=lambda x: x["evaluation"]["benefit_per_minute"], reverse=True
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

    # evaluate cycles
    evaluated_cycles = evaluate_all_cycles(
        G, cycles, snowfall_rate_per_minute, storm_duration_minutes
    )

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
                "benefit": cycle_data["evaluation"]["benefit"],
                "benefit_per_minute": cycle_data["evaluation"]["benefit_per_minute"],
                "cost_without_plow": cycle_data["evaluation"]["cost_without_plow"],
                "cost_with_plow": cycle_data["evaluation"]["cost_with_plow"],
                "total_priority": cycle_data["metrics"]["total_priority"],
                "unique_edges": cycle_data["metrics"]["unique_edges"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("benefit_per_minute", ascending=False)
    summary_df.to_csv("cycle_evaluations.csv", index=False)
    print(f"[INFO] saved cycle evaluations to cycle_evaluations.csv")

    # print summary statistics
    print("\n[SUMMARY] evaluation statistics:")
    print(f"average benefit: {summary_df['benefit'].mean():.2f}")
    print(f"average benefit per minute: {summary_df['benefit_per_minute'].mean():.4f}")
    print(f"best benefit per minute: {summary_df['benefit_per_minute'].max():.4f}")

    # show top cycles per partition
    print("\n[INFO] top cycle per partition:")
    for key in sorted(ranked_partitions.keys()):
        ps_id, p_id = key
        top_cycle = ranked_partitions[key][0]
        print(
            f"  partition set {ps_id}, partition {p_id}: benefit/min = {top_cycle['evaluation']['benefit_per_minute']:.4f}, method = {top_cycle['method']}"
        )

    return evaluated_cycles


if __name__ == "__main__":
    main()
