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
from evaluate_cycles import calculate_cycle_benefit


SNOWFALL_RATE_INCHES_PER_HOUR = 1.0
STORM_DURATION_HOURS = 12


def edge_swap_perturbation(G, cycle, num_attempts=5):
    """
    try to swap an edge in the cycle with a nearby edge.

    args:
        G: networkx graph
        cycle: list of (from_node, to_node, edge_key) tuples
        num_attempts: number of random swap attempts

    returns:
        new cycle or none if no valid swap found
    """
    if len(cycle) < 2:
        return None

    for _ in range(num_attempts):
        # choose random position in cycle
        pos = random.randint(0, len(cycle) - 1)
        from_node, to_node, edge_key = cycle[pos]

        # find alternative edges from same start node
        available_edges = []
        for neighbor in G.neighbors(from_node):
            if neighbor != to_node:
                for alt_key in G[from_node][neighbor]:
                    available_edges.append((neighbor, alt_key))

        if not available_edges:
            continue

        # choose random alternative
        new_to_node, new_edge_key = random.choice(available_edges)

        # try to create valid cycle by connecting new edge
        # need path from new_to_node to next node in cycle
        if pos + 1 < len(cycle):
            next_node = cycle[pos + 1][0]
            try:
                # check if we can reach next node
                if nx.has_path(G, new_to_node, next_node):
                    # create new cycle with swap
                    new_cycle = (
                        cycle[:pos]
                        + [(from_node, new_to_node, new_edge_key)]
                        + cycle[pos + 1 :]
                    )
                    return new_cycle
            except:
                pass

    return None


def vertex_insert_perturbation(G, cycle, num_attempts=5):
    """
    try to insert a new vertex into the cycle.

    args:
        G: networkx graph
        cycle: list of (from_node, to_node, edge_key) tuples
        num_attempts: number of random insert attempts

    returns:
        new cycle or none if no valid insert found
    """
    if len(cycle) < 2:
        return None

    for _ in range(num_attempts):
        # choose random position in cycle
        pos = random.randint(0, len(cycle) - 1)
        from_node, to_node, edge_key = cycle[pos]

        # find neighbors of from_node that are not to_node
        intermediate_nodes = [n for n in G.neighbors(from_node) if n != to_node]

        if not intermediate_nodes:
            continue

        # choose random intermediate node
        intermediate = random.choice(intermediate_nodes)

        # check if we can go from intermediate to to_node
        if to_node in G.neighbors(intermediate):
            # create path: from_node -> intermediate -> to_node
            edge1_key = list(G[from_node][intermediate].keys())[0]
            edge2_key = list(G[intermediate][to_node].keys())[0]

            # create new cycle with inserted vertex
            new_cycle = (
                cycle[:pos]
                + [
                    (from_node, intermediate, edge1_key),
                    (intermediate, to_node, edge2_key),
                ]
                + cycle[pos + 1 :]
            )
            return new_cycle

    return None


def vertex_remove_perturbation(G, cycle, num_attempts=5):
    """
    try to remove a vertex from the cycle.

    args:
        G: networkx graph
        cycle: list of (from_node, to_node, edge_key) tuples
        num_attempts: number of random remove attempts

    returns:
        new cycle or none if no valid remove found
    """
    if len(cycle) < 3:
        return None

    for _ in range(num_attempts):
        # choose random position (not first or last)
        pos = random.randint(1, len(cycle) - 2)

        # get nodes: prev -> current -> next
        prev_node = cycle[pos - 1][0]
        current_to = cycle[pos][1]
        next_from = cycle[pos + 1][0]

        # check if current edge is skippable
        # we want to go directly from prev_node to next destination
        next_to = cycle[pos + 1][1]

        # check if we can connect prev_node directly to next_from
        if next_from in G.neighbors(prev_node):
            edge_key = list(G[prev_node][next_from].keys())[0]

            # create new cycle without the middle edge
            new_cycle = (
                cycle[:pos] + [(prev_node, next_from, edge_key)] + cycle[pos + 2 :]
            )
            return new_cycle

    return None


def two_opt_perturbation(cycle, num_attempts=5):
    """
    try 2-opt style move: reverse a subsection of the cycle.

    args:
        cycle: list of (from_node, to_node, edge_key) tuples
        num_attempts: number of random 2-opt attempts

    returns:
        new cycle or none if no valid move found
    """
    if len(cycle) < 4:
        return None

    for _ in range(num_attempts):
        # choose two random positions
        i = random.randint(0, len(cycle) - 3)
        j = random.randint(i + 2, len(cycle) - 1)

        # reverse subsection (note: this may break connectivity)
        # for now, just try it and validate later
        new_cycle = cycle[:i] + cycle[i : j + 1][::-1] + cycle[j + 1 :]

        # simple validation: check that consecutive edges connect
        valid = True
        for k in range(len(new_cycle) - 1):
            if new_cycle[k][1] != new_cycle[k + 1][0]:
                valid = False
                break

        if valid:
            return new_cycle

    return None


def optimize_cycle(
    G, cycle, max_iterations=20, snowfall_rate=None, storm_duration=None
):
    """
    optimize a cycle using local search with perturbations.

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
    current_benefit = current_benefit_metrics["benefit_per_minute"]
    best_cycle = cycle
    best_benefit = current_benefit

    no_improvement_count = 0

    for iteration in range(max_iterations):
        # try different perturbations
        perturbations = []

        # edge swap
        new_cycle = edge_swap_perturbation(G, best_cycle)
        if new_cycle:
            perturbations.append(("edge_swap", new_cycle))

        # vertex insert
        new_cycle = vertex_insert_perturbation(G, best_cycle)
        if new_cycle:
            perturbations.append(("vertex_insert", new_cycle))

        # vertex remove
        new_cycle = vertex_remove_perturbation(G, best_cycle)
        if new_cycle:
            perturbations.append(("vertex_remove", new_cycle))

        # 2-opt
        new_cycle = two_opt_perturbation(best_cycle)
        if new_cycle:
            perturbations.append(("two_opt", new_cycle))

        if not perturbations:
            break  # no valid perturbations

        # evaluate all perturbations
        improved = False
        for perturb_type, new_cycle in perturbations:
            try:
                new_benefit_metrics = calculate_cycle_benefit(
                    G, new_cycle, snowfall_rate, storm_duration
                )
                new_benefit = new_benefit_metrics["benefit_per_minute"]

                # accept if improvement
                if new_benefit > best_benefit:
                    best_cycle = new_cycle
                    best_benefit = new_benefit
                    improved = True
                    break  # accept first improvement
            except:
                pass  # invalid cycle

        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= 5:
                break  # converged
        else:
            no_improvement_count = 0

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
