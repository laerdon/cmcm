#!/usr/bin/env python3
"""
optimize routes using local search with perturbations.
perturbation methods: edge swaps, vertex insert/remove, 2-opt style moves.
"""

import networkx as nx
import numpy as np
import json
import pickle
import random
import time
from evaluate_cycles import calculate_cycle_benefit


SNOWFALL_RATE_INCHES_PER_HOUR = 1.0
STORM_DURATION_HOURS = 12


def is_valid_cycle(G, new_cycle, original_cycle):
    """
    validate that a cycle is valid and preserves coverage.

    args:
        G: networkx graph
        new_cycle: proposed new cycle
        original_cycle: original cycle to compare against

    returns:
        bool indicating if cycle is valid
    """
    if not new_cycle or len(new_cycle) == 0:
        return False

    # check 1: cycle closes (ends at start)
    if new_cycle[0][0] != new_cycle[-1][1]:
        return False

    # check 2: cycle is connected (each edge connects to next)
    for i in range(len(new_cycle) - 1):
        if new_cycle[i][1] != new_cycle[i + 1][0]:
            return False

    # check 3: all unique edges from original still present
    original_unique = set(original_cycle)
    new_unique = set(new_cycle)
    if not original_unique.issubset(new_unique):
        return False

    return True


def reorder_by_priority_clustering(G, cycle):
    """
    reorder cycle to visit high-priority edges early.

    args:
        G: networkx graph
        cycle: list of (from_node, to_node, edge_key) tuples

    returns:
        reordered cycle or none if failed
    """
    if len(cycle) < 3:
        return None

    try:
        # extract unique edges with priorities
        unique_edges = list(set(cycle))
        edge_priorities = []
        for u, v, k in unique_edges:
            priority = G[u][v][k]["priority"]
            edge_priorities.append((u, v, k, priority))

        # sort by priority (descending)
        edge_priorities.sort(key=lambda x: x[3], reverse=True)

        # build new cycle starting from highest priority edge
        start_edge = edge_priorities[0][:3]
        visited = {start_edge}
        new_cycle = [start_edge]
        current_node = start_edge[1]

        # greedily add nearest unvisited edge
        while len(visited) < len(unique_edges):
            # find unvisited edges and their distances from current node
            candidates = []
            for u, v, k, p in edge_priorities:
                edge = (u, v, k)
                if edge not in visited:
                    try:
                        # distance from current node to edge start
                        dist = nx.shortest_path_length(
                            G, current_node, u, weight="travel_time"
                        )
                        candidates.append((edge, dist, p))
                    except nx.NetworkXNoPath:
                        pass

            if not candidates:
                break

            # choose nearest unvisited edge (with priority tie-breaker)
            candidates.sort(key=lambda x: (x[1], -x[2]))
            next_edge, dist, _ = candidates[0]

            # add connector path if needed
            if current_node != next_edge[0]:
                try:
                    path = nx.shortest_path(
                        G, current_node, next_edge[0], weight="travel_time"
                    )
                    for i in range(len(path) - 1):
                        if path[i + 1] in G.neighbors(path[i]):
                            edge_key = list(G[path[i]][path[i + 1]].keys())[0]
                            new_cycle.append((path[i], path[i + 1], edge_key))
                except nx.NetworkXNoPath:
                    return None

            # add the edge
            new_cycle.append(next_edge)
            visited.add(next_edge)
            current_node = next_edge[1]

        # close the cycle
        start_node = new_cycle[0][0]
        if current_node != start_node:
            try:
                path = nx.shortest_path(
                    G, current_node, start_node, weight="travel_time"
                )
                for i in range(len(path) - 1):
                    if path[i + 1] in G.neighbors(path[i]):
                        edge_key = list(G[path[i]][path[i + 1]].keys())[0]
                        new_cycle.append((path[i], path[i + 1], edge_key))
            except nx.NetworkXNoPath:
                return None

        return new_cycle

    except Exception as e:
        return None


def optimize_connectors(G, cycle):
    """
    optimize connector paths between required edges.

    args:
        G: networkx graph
        cycle: list of (from_node, to_node, edge_key) tuples

    returns:
        optimized cycle or none if failed
    """
    if len(cycle) < 3:
        return None

    try:
        # identify unique edges (required) vs connectors (repeated)
        edge_counts = {}
        for edge in cycle:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

        required_edges = [e for e, count in edge_counts.items() if count == 1]
        if len(required_edges) < 2:
            return None

        # rebuild cycle with optimized connectors
        new_cycle = [required_edges[0]]
        current_node = required_edges[0][1]

        for next_edge in required_edges[1:]:
            # find shortest path connector
            if current_node != next_edge[0]:
                try:
                    path = nx.shortest_path(
                        G, current_node, next_edge[0], weight="travel_time"
                    )
                    for i in range(len(path) - 1):
                        if path[i + 1] in G.neighbors(path[i]):
                            edge_key = list(G[path[i]][path[i + 1]].keys())[0]
                            new_cycle.append((path[i], path[i + 1], edge_key))
                except nx.NetworkXNoPath:
                    return None

            new_cycle.append(next_edge)
            current_node = next_edge[1]

        # close cycle
        start_node = new_cycle[0][0]
        if current_node != start_node:
            try:
                path = nx.shortest_path(
                    G, current_node, start_node, weight="travel_time"
                )
                for i in range(len(path) - 1):
                    if path[i + 1] in G.neighbors(path[i]):
                        edge_key = list(G[path[i]][path[i + 1]].keys())[0]
                        new_cycle.append((path[i], path[i + 1], edge_key))
            except nx.NetworkXNoPath:
                return None

        return new_cycle

    except Exception as e:
        return None


def swap_adjacent_sequences(G, cycle, num_attempts=10):
    """
    try swapping order of adjacent edge sequences.

    args:
        G: networkx graph
        cycle: list of (from_node, to_node, edge_key) tuples
        num_attempts: number of swap attempts

    returns:
        modified cycle or none if no improvement
    """
    if len(cycle) < 6:
        return None

    for _ in range(num_attempts):
        # choose two adjacent segments
        seq_len = random.randint(2, min(4, len(cycle) // 3))
        start1 = random.randint(0, len(cycle) - 2 * seq_len - 1)
        start2 = start1 + seq_len

        seq1 = cycle[start1 : start1 + seq_len]
        seq2 = cycle[start2 : start2 + seq_len]

        # try swapping them
        new_cycle = (
            cycle[:start1]
            + seq2
            + cycle[start1 + seq_len : start2]
            + seq1
            + cycle[start2 + seq_len :]
        )

        # check connectivity
        valid = True
        for i in range(len(new_cycle) - 1):
            if new_cycle[i][1] != new_cycle[i + 1][0]:
                valid = False
                break

        if valid and new_cycle[0][0] == new_cycle[-1][1]:
            return new_cycle

    return None


def two_opt_edge_order(G, cycle):
    """
    apply 2-opt to improve edge ordering.

    args:
        G: networkx graph
        cycle: list of (from_node, to_node, edge_key) tuples

    returns:
        improved cycle or none if no improvement
    """
    if len(cycle) < 4:
        return None

    initial_cycle = list(cycle)
    best_cycle = list(cycle)
    best_time = sum(G[u][v][k]["travel_time"] for u, v, k in best_cycle)

    max_evals = 10000            
    max_seconds = 10.0            
    eval_count = 0
    start_time = time.time()

    improved = True
    while improved:
        improved = False
        n = len(best_cycle)
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                # try reversing segment [i:j]
                # stop if we've used too much time or too many evaluations
                eval_count += 1
                if eval_count > max_evals or (time.time() - start_time) > max_seconds:
                    # return current best (if different) or None
                    return best_cycle if best_cycle != initial_cycle else None

                # try reversing segment [i:j)
                new_cycle = best_cycle[:i] + best_cycle[i:j][::-1] + best_cycle[j:]

                # check connectivity
                valid = True
                for k in range(len(new_cycle) - 1):
                    if new_cycle[k][1] != new_cycle[k + 1][0]:
                        valid = False
                        break

                if not valid:
                    continue

                # cycle must close
                if new_cycle[0][0] != new_cycle[-1][1]:
                    continue

                # compute travel time for candidate
                new_time = sum(G[u][v][k]["travel_time"] for u, v, k in new_cycle)

                if new_time < best_time - 1e-6:  # small tolerance
                    best_cycle = new_cycle
                    best_time = new_time
                    improved = True
                    break

            if improved:
                break


    return best_cycle if best_cycle != initial_cycle else None


def optimize_cycle(
    G, cycle, max_iterations=20, snowfall_rate=None, storm_duration=None
):
    """
    optimize a cycle using ordering and routing improvements.

    args:
        G: networkx graph
        cycle: initial cycle
        max_iterations: maximum iterations
        snowfall_rate: snowfall rate in inches per minute
        storm_duration: storm duration in minutes

    returns:
        optimized cycle
    """
    if snowfall_rate is None:
        snowfall_rate = SNOWFALL_RATE_INCHES_PER_HOUR / 60.0
    if storm_duration is None:
        storm_duration = STORM_DURATION_HOURS * 60

    # calculate initial benefit
    current_benefit_metrics = calculate_cycle_benefit(
        G, cycle, snowfall_rate, storm_duration
    )
    best_cycle = cycle
    best_benefit = current_benefit_metrics.get(
        "benefit_pct", current_benefit_metrics["benefit_per_minute"]
    )

    print(f"    [DEBUG] initial benefit: {best_benefit:.2f}%")

    # use 2-opt optimization
    for iteration in range(max_iterations):
        try:
            new_cycle = two_opt_edge_order(G, best_cycle)

            # validate and evaluate
            if new_cycle and is_valid_cycle(G, new_cycle, cycle):
                new_benefit_metrics = calculate_cycle_benefit(
                    G, new_cycle, snowfall_rate, storm_duration
                )
                new_benefit = new_benefit_metrics.get(
                    "benefit_pct", new_benefit_metrics["benefit_per_minute"]
                )

                # accept if improvement
                if new_benefit > best_benefit * 1.0001:  # 0.1% threshold
                    print(
                        f"    [DEBUG] iteration {iteration}: {best_benefit:.2f}% -> {new_benefit:.2f}%"
                    )
                    best_cycle = new_cycle
                    best_benefit = new_benefit
                else:
                    # no improvement, stop
                    break

            else:
                # no valid cycle found, stop
                break

        except Exception as e:
            # operator failed, stop
            break

    print(f"    [DEBUG] final benefit: {best_benefit:.2f}%")
    return best_cycle


def optimize_selected_cycles(G, selected_cycles, max_iterations=20):
    """
    optimize all selected cycles.

    args:
        G: networkx graph
        selected_cycles: dict mapping partition key to cycles
        max_iterations: max iterations per cycle

    returns:
        dict with optimized cycles
    """
    snowfall_rate = SNOWFALL_RATE_INCHES_PER_HOUR / 60.0
    storm_duration = STORM_DURATION_HOURS * 60

    optimized = {}

    for key, cycles in selected_cycles.items():
        print(f"[INFO] optimizing cycles for partition {key}...")

        optimized_cycles = []
        for i, cycle_data in enumerate(cycles):
            cycle = cycle_data["cycle"]

            # optimize
            opt_cycle = optimize_cycle(
                G,
                cycle,
                max_iterations=max_iterations,
                snowfall_rate=snowfall_rate,
                storm_duration=storm_duration,
            )

            # calculate new benefit
            benefit_metrics = calculate_cycle_benefit(
                G, opt_cycle, snowfall_rate, storm_duration
            )

            # update cycle data
            optimized_cycle_data = cycle_data.copy()
            optimized_cycle_data["cycle"] = opt_cycle
            optimized_cycle_data["evaluation"] = {
                "cost_without_plow": float(benefit_metrics["cost_without_plow"]),
                "cost_with_plow": float(benefit_metrics["cost_with_plow"]),
                "benefit": float(benefit_metrics["benefit"]),
                "benefit_per_minute": float(benefit_metrics["benefit_per_minute"]),
                "cycle_time": float(benefit_metrics["cycle_time"]),
                "num_complete_cycles": int(benefit_metrics["num_complete_cycles"]),
                "coverage_ratio": float(benefit_metrics["coverage_ratio"]),
            }

            # recalculate metrics
            unique_edges = set(
                tuple(e) if isinstance(e, list) else e for e in opt_cycle
            )
            optimized_cycle_data["metrics"] = {
                "total_time": float(benefit_metrics["cycle_time"]),
                "unique_edges": len(unique_edges),
                "num_steps": len(opt_cycle),
            }

            optimized_cycles.append(optimized_cycle_data)

            if i % 3 == 0:
                print(f"  optimized {i+1}/{len(cycles)} cycles...")

        optimized[key] = optimized_cycles

    return optimized


def main():
    print("[INFO] loading road graph...")
    with open("road_graph.gpickle", "rb") as f:
        G = pickle.load(f)
    print(
        f"[INFO] loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )

    print("[INFO] loading selected cycles...")
    with open("selected_cycles.json", "r") as f:
        selected_cycles = json.load(f)
    print(f"[INFO] loaded cycles for {len(selected_cycles)} partitions")

    # optimize cycles
    max_iterations = 20  # tunable parameter
    print(f"\n[INFO] optimizing cycles (max {max_iterations} iterations per cycle)...")

    optimized_cycles = optimize_selected_cycles(
        G, selected_cycles, max_iterations=max_iterations
    )

    # save optimized cycles
    output_file = "optimized_cycles.json"
    with open(output_file, "w") as f:
        json.dump(optimized_cycles, f, indent=2)

    print(f"\n[INFO] saved optimized cycles to {output_file}")

    # compare improvements
    print("\n[SUMMARY] optimization improvements:")
    total_original_benefit = 0
    total_optimized_benefit = 0

    for key in selected_cycles:
        original_benefit = selected_cycles[key][0]["evaluation"]["benefit_per_minute"]
        optimized_benefit = optimized_cycles[key][0]["evaluation"]["benefit_per_minute"]

        improvement = (
            ((optimized_benefit - original_benefit) / original_benefit) * 100
            if original_benefit > 0
            else 0
        )

        print(
            f"  {key}: {original_benefit:.2f} -> {optimized_benefit:.2f} ({improvement:+.2f}%)"
        )

        total_original_benefit += original_benefit
        total_optimized_benefit += optimized_benefit

    overall_improvement = (
        (total_optimized_benefit - total_original_benefit) / total_original_benefit
    ) * 100
    print(f"\noverall improvement: {overall_improvement:+.2f}%")

    return optimized_cycles


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
