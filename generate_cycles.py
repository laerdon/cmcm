#!/usr/bin/env python3
"""
generate candidate plow cycles for each partition using smart sampling.
methods: greedy construction, priority-biased random walks with varying bias parameters.
"""

import networkx as nx
import numpy as np
import json
import pickle
import random


MAX_CYCLE_TIME_MINUTES = 360  # 6 hours max per cycle


def greedy_cycle(G, start_node, max_time=MAX_CYCLE_TIME_MINUTES):
    """
    construct cycle greedily by always choosing highest priority unvisited edge.
    if no unvisited edges from current location, picks randomly to continue exploring.

    args:
        G: networkx graph
        start_node: starting node
        max_time: maximum cycle time in minutes

    returns:
        list of (node, edge_key) tuples representing the cycle
    """
    cycle = []
    current_node = start_node
    visited_edges = set()
    total_time = 0.0

    # collect all edges in the subgraph
    all_edges = set()
    for u, v, k in G.edges(keys=True):
        all_edges.add((u, v, k))

    while total_time < max_time:
        # find unvisited edges from current node
        unvisited_from_here = []
        visited_from_here = []

        for neighbor in G.neighbors(current_node):
            for edge_key in G[current_node][neighbor]:
                edge_id = (current_node, neighbor, edge_key)
                edge_data = G[current_node][neighbor][edge_key]

                if edge_id not in visited_edges:
                    unvisited_from_here.append((neighbor, edge_key, edge_data))
                else:
                    visited_from_here.append((neighbor, edge_key, edge_data))

        # check if we've visited all edges in the graph
        unvisited_global = all_edges - visited_edges

        if len(unvisited_from_here) > 0:
            # prioritize unvisited edges - pick highest priority
            unvisited_from_here.sort(key=lambda x: x[2]["priority"], reverse=True)
            next_node, edge_key, edge_data = unvisited_from_here[0]

        elif len(unvisited_global) > 0:
            # no unvisited edges here, but there are unvisited edges elsewhere
            # navigate to the nearest unvisited edge
            target_edge = max(
                unvisited_global, key=lambda e: G[e[0]][e[1]][e[2]]["priority"]
            )
            target_start = target_edge[0]

            try:
                path = nx.shortest_path(
                    G, current_node, target_start, weight="travel_time"
                )
                # traverse path to target (using already-visited edges)
                for i in range(len(path) - 1):
                    if path[i + 1] in G.neighbors(path[i]):
                        edge_key = list(G[path[i]][path[i + 1]].keys())[0]
                        edge_data = G[path[i]][path[i + 1]][edge_key]
                        cycle.append((path[i], path[i + 1], edge_key))
                        total_time += edge_data["travel_time"]
                        if total_time >= max_time * 0.9:
                            # time limit reached during navigation
                            current_node = path[i + 1]
                            break
                else:
                    # completed navigation without time limit
                    current_node = target_start
                    continue
                # if we broke due to time, exit main loop
                break
            except nx.NetworkXNoPath:
                # can't reach unvisited edges - they're in disconnected component
                break

        elif len(visited_from_here) > 0:
            # all edges visited globally, pick random edge to continue (allows cycling)
            next_node, edge_key, edge_data = random.choice(visited_from_here)

        else:
            # dead end with no outgoing edges - try to return to start
            try:
                path = nx.shortest_path(G, current_node, start_node)
                for i in range(len(path) - 1):
                    edge_key = list(G[path[i]][path[i + 1]].keys())[0]
                    edge_data = G[path[i]][path[i + 1]][edge_key]
                    cycle.append((path[i], path[i + 1], edge_key))
                    total_time += edge_data["travel_time"]
                break
            except nx.NetworkXNoPath:
                break

        # add chosen edge to cycle
        cycle.append((current_node, next_node, edge_key))
        visited_edges.add((current_node, next_node, edge_key))
        total_time += edge_data["travel_time"]
        current_node = next_node

        # check if we should stop (time limit)
        if total_time >= max_time * 0.9 and len(unvisited_global) == 0:
            # covered all edges and approaching time limit - go home
            break

    # ensure we return to start node
    if current_node != start_node:
        try:
            path = nx.shortest_path(G, current_node, start_node, weight="travel_time")
            for i in range(len(path) - 1):
                edge_key = list(G[path[i]][path[i + 1]].keys())[0]
                edge_data = G[path[i]][path[i + 1]][edge_key]
                cycle.append((path[i], path[i + 1], edge_key))
                total_time += edge_data["travel_time"]
        except nx.NetworkXNoPath:
            print(
                f"[WARNING] greedy_cycle: cannot return to start from node {current_node}"
            )

    return cycle


def biased_random_walk_cycle(
    G, start_node, max_time=MAX_CYCLE_TIME_MINUTES, priority_bias=2.0
):
    """
    construct cycle using random walk biased toward high-priority edges.

    args:
        G: networkx graph
        start_node: starting node
        max_time: maximum cycle time in minutes
        priority_bias: how strongly to bias toward high priority (higher = more bias)

    returns:
        list of (node, edge_key) tuples representing the cycle
    """
    cycle = []
    current_node = start_node
    visited_edges = set()
    total_time = 0.0

    while total_time < max_time:
        # find available edges
        available_edges = []
        for neighbor in G.neighbors(current_node):
            for edge_key in G[current_node][neighbor]:
                edge_id = (current_node, neighbor, edge_key)
                edge_data = G[current_node][neighbor][edge_key]
                available_edges.append((neighbor, edge_key, edge_data))

        if not available_edges:
            # try to return to start
            try:
                path = nx.shortest_path(G, current_node, start_node)
                for i in range(len(path) - 1):
                    edge_key = list(G[path[i]][path[i + 1]].keys())[0]
                    edge_data = G[path[i]][path[i + 1]][edge_key]
                    cycle.append((path[i], path[i + 1], edge_key))
                    total_time += edge_data["travel_time"]
                break
            except nx.NetworkXNoPath:
                break

        # calculate selection probabilities biased by priority
        priorities = np.array([e[2]["priority"] for e in available_edges])

        # bonus for unvisited edges
        edge_visited = np.array(
            [(current_node, e[0], e[1]) in visited_edges for e in available_edges]
        )
        priorities = priorities * np.where(edge_visited, 0.5, 1.0)

        # apply bias
        probabilities = priorities**priority_bias
        probabilities = probabilities / probabilities.sum()

        # choose edge
        idx = np.random.choice(len(available_edges), p=probabilities)
        next_node, edge_key, edge_data = available_edges[idx]

        # add to cycle
        cycle.append((current_node, next_node, edge_key))
        visited_edges.add((current_node, next_node, edge_key))
        total_time += edge_data["travel_time"]
        current_node = next_node

        # check if we should return
        if total_time >= max_time * 0.9:
            break

    # ensure we return to start node
    if current_node != start_node:
        try:
            path = nx.shortest_path(G, current_node, start_node, weight="travel_time")
            for i in range(len(path) - 1):
                edge_key = list(G[path[i]][path[i + 1]].keys())[0]
                edge_data = G[path[i]][path[i + 1]][edge_key]
                cycle.append((path[i], path[i + 1], edge_key))
                total_time += edge_data["travel_time"]
        except nx.NetworkXNoPath:
            print(
                f"[WARNING] biased_random_walk_cycle: cannot return to start from node {current_node}"
            )

    return cycle


def calculate_cycle_metrics(G, cycle):
    """
    calculate metrics for a cycle.

    args:
        G: networkx graph
        cycle: list of (from_node, to_node, edge_key) tuples

    returns:
        dict with metrics
    """
    if not cycle:
        return {
            "total_time": 0.0,
            "total_length": 0.0,
            "total_priority": 0.0,
            "unique_edges": 0,
            "num_steps": 0,
        }

    total_time = 0.0
    total_length = 0.0
    total_priority = 0.0
    unique_edges = set()

    for from_node, to_node, edge_key in cycle:
        edge_data = G[from_node][to_node][edge_key]
        total_time += edge_data["travel_time"]
        total_length += edge_data["length_feet"]
        total_priority += edge_data["priority"]
        unique_edges.add((from_node, to_node, edge_key))

    return {
        "total_time": total_time,
        "total_length": total_length,
        "total_priority": total_priority,
        "unique_edges": len(unique_edges),
        "num_steps": len(cycle),
    }


def generate_cycles_for_partition(G, partition_nodes, num_cycles=50):
    """
    generate candidate cycles for a partition.

    args:
        G: networkx graph
        partition_nodes: list of nodes in partition
        num_cycles: number of cycles to generate

    returns:
        list of cycles with metrics
    """
    # create subgraph
    subgraph = G.subgraph(partition_nodes).copy()

    if len(partition_nodes) == 0:
        return []

    cycles = []

    # choose random start nodes
    start_nodes = random.choices(list(partition_nodes), k=num_cycles)

    # distribute across methods and bias values
    methods_per_type = num_cycles // 4

    print(f"  [INFO] generating {methods_per_type} greedy cycles...")
    for i in range(methods_per_type):
        cycle = greedy_cycle(subgraph, start_nodes[i])
        metrics = calculate_cycle_metrics(subgraph, cycle)
        cycles.append(
            {
                "method": "greedy",
                "cycle": cycle,
                "metrics": metrics,
                "start_node": int(start_nodes[i]),
            }
        )

    # try three different priority bias values
    bias_values = [0.5, 1.0, 2.0]  # low, medium, high bias toward priority

    for bias_idx, bias in enumerate(bias_values):
        start_idx = methods_per_type + (bias_idx * methods_per_type)
        end_idx = start_idx + methods_per_type

        print(
            f"  [INFO] generating {methods_per_type} random walk cycles (bias={bias})..."
        )
        for i in range(start_idx, end_idx):
            cycle = biased_random_walk_cycle(
                subgraph, start_nodes[i], priority_bias=bias
            )
            metrics = calculate_cycle_metrics(subgraph, cycle)
            cycles.append(
                {
                    "method": f"random_walk_bias_{bias}",
                    "cycle": cycle,
                    "metrics": metrics,
                    "start_node": int(start_nodes[i]),
                }
            )

    return cycles


def main():
    print("[INFO] loading road graph...")
    with open("road_graph.gpickle", "rb") as f:
        G = pickle.load(f)
    print(
        f"[INFO] loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )

    print("[INFO] loading partitions...")
    with open("partitions.json", "r") as f:
        partition_sets = json.load(f)
    print(f"[INFO] loaded {len(partition_sets)} partition sets")

    # parameters
    num_cycles_per_partition = 50  # tunable

    # generate cycles for each partition in each partition set
    all_cycles = []

    for ps_idx, partition_set in enumerate(partition_sets):
        print(
            f"\n[INFO] processing partition set {ps_idx + 1}/{len(partition_sets)}..."
        )

        partition_set_cycles = []

        for p_idx, partition_nodes in enumerate(partition_set):
            print(
                f"[INFO] generating cycles for partition {p_idx + 1}/8 ({len(partition_nodes)} nodes)..."
            )

            cycles = generate_cycles_for_partition(
                G, partition_nodes, num_cycles_per_partition
            )

            # add metadata
            for cycle_data in cycles:
                cycle_data["partition_set_id"] = ps_idx
                cycle_data["partition_id"] = p_idx

            partition_set_cycles.extend(cycles)

            print(
                f"  [INFO] generated {len(cycles)} cycles (avg time: {np.mean([c['metrics']['total_time'] for c in cycles]):.2f} min)"
            )

        all_cycles.extend(partition_set_cycles)

    # save cycles
    output_file = "candidate_cycles.json"

    # convert cycles to serializable format
    cycles_serializable = []
    for cycle_data in all_cycles:
        cycles_serializable.append(
            {
                "partition_set_id": cycle_data["partition_set_id"],
                "partition_id": cycle_data["partition_id"],
                "method": cycle_data["method"],
                "start_node": cycle_data["start_node"],
                "cycle": [(int(u), int(v), int(k)) for u, v, k in cycle_data["cycle"]],
                "metrics": {
                    "total_time": float(cycle_data["metrics"]["total_time"]),
                    "total_length": float(cycle_data["metrics"]["total_length"]),
                    "total_priority": float(cycle_data["metrics"]["total_priority"]),
                    "unique_edges": int(cycle_data["metrics"]["unique_edges"]),
                    "num_steps": int(cycle_data["metrics"]["num_steps"]),
                },
            }
        )

    with open(output_file, "w") as f:
        json.dump(cycles_serializable, f, indent=2)

    print(f"\n[INFO] saved {len(cycles_serializable)} cycles to {output_file}")

    # summary statistics
    print("\n[SUMMARY] cycle generation statistics:")
    print(f"total cycles generated: {len(all_cycles)}")
    print(
        f"average cycle time: {np.mean([c['metrics']['total_time'] for c in all_cycles]):.2f} minutes"
    )
    print(
        f"average cycle priority: {np.mean([c['metrics']['total_priority'] for c in all_cycles]):.2f}"
    )
    print(
        f"average unique edges: {np.mean([c['metrics']['unique_edges'] for c in all_cycles]):.1f}"
    )

    return all_cycles


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
