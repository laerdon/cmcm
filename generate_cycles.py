#!/usr/bin/env python3
"""
generate candidate plow cycles for each partition using smart sampling.
methods: greedy construction, priority-biased random walks, chinese postman approximation.
"""

import networkx as nx
import numpy as np
import json
import pickle
from collections import defaultdict
import random


MAX_CYCLE_TIME_MINUTES = 120  # 2 hours max per cycle


def greedy_cycle(G, start_node, max_time=MAX_CYCLE_TIME_MINUTES):
    """
    construct cycle greedily by always choosing highest priority unvisited edge.

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

    while total_time < max_time:
        # find unvisited edges from current node
        available_edges = []
        for neighbor in G.neighbors(current_node):
            for edge_key in G[current_node][neighbor]:
                edge_id = (current_node, neighbor, edge_key)
                if edge_id not in visited_edges:
                    edge_data = G[current_node][neighbor][edge_key]
                    available_edges.append((neighbor, edge_key, edge_data))

        if not available_edges:
            # no more unvisited edges, try to return to start
            try:
                path = nx.shortest_path(G, current_node, start_node)
                for i in range(len(path) - 1):
                    # use any edge between consecutive nodes
                    edge_key = list(G[path[i]][path[i + 1]].keys())[0]
                    edge_data = G[path[i]][path[i + 1]][edge_key]
                    cycle.append((path[i], path[i + 1], edge_key))
                    total_time += edge_data["travel_time"]
                break
            except nx.NetworkXNoPath:
                break

        # choose edge with highest priority
        available_edges.sort(key=lambda x: x[2]["priority"], reverse=True)
        next_node, edge_key, edge_data = available_edges[0]

        # add to cycle
        cycle.append((current_node, next_node, edge_key))
        visited_edges.add((current_node, next_node, edge_key))
        total_time += edge_data["travel_time"]
        current_node = next_node

        # check if we should stop
        if total_time >= max_time * 0.9:  # leave buffer for return
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

    return cycle


def chinese_postman_cycle(G, start_node):
    """
    approximate chinese postman problem solution: cover all edges efficiently.

    args:
        G: networkx graph
        start_node: starting node

    returns:
        list of (node, edge_key) tuples representing the cycle
    """
    # convert to undirected for cpp
    G_undirected = G.to_undirected()

    # find nodes with odd degree
    odd_degree_nodes = [
        n for n in G_undirected.nodes() if G_undirected.degree(n) % 2 == 1
    ]

    # if graph is not eulerian, we need to add edges
    if len(odd_degree_nodes) > 0:
        # pair up odd degree nodes and add shortest paths between them
        # simple greedy pairing
        paired = set()
        edges_to_duplicate = []

        for i in range(0, len(odd_degree_nodes), 2):
            if i + 1 < len(odd_degree_nodes):
                node1 = odd_degree_nodes[i]
                node2 = odd_degree_nodes[i + 1]
                try:
                    path = nx.shortest_path(G_undirected, node1, node2)
                    for j in range(len(path) - 1):
                        edges_to_duplicate.append((path[j], path[j + 1]))
                except nx.NetworkXNoPath:
                    pass

    # create eulerian graph (simplified - just use original graph)
    # traverse using hierholzer's algorithm approximation
    cycle = []
    current_node = start_node
    edge_traversals = defaultdict(int)
    stack = [current_node]
    path = []

    while stack:
        current = stack[-1]
        found_edge = False

        for neighbor in G.neighbors(current):
            for edge_key in G[current][neighbor]:
                edge_id = (current, neighbor, edge_key)
                if edge_traversals[edge_id] == 0:
                    edge_traversals[edge_id] += 1
                    stack.append(neighbor)
                    found_edge = True
                    break
            if found_edge:
                break

        if not found_edge:
            path.append(stack.pop())

    # convert path to cycle format
    path.reverse()
    for i in range(len(path) - 1):
        # find an edge between consecutive nodes
        if path[i + 1] in G.neighbors(path[i]):
            edge_key = list(G[path[i]][path[i + 1]].keys())[0]
            cycle.append((path[i], path[i + 1], edge_key))

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

    # distribute across methods
    methods_per_type = num_cycles // 3

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

    print(f"  [INFO] generating {methods_per_type} random walk cycles...")
    for i in range(methods_per_type, 2 * methods_per_type):
        cycle = biased_random_walk_cycle(subgraph, start_nodes[i])
        metrics = calculate_cycle_metrics(subgraph, cycle)
        cycles.append(
            {
                "method": "random_walk",
                "cycle": cycle,
                "metrics": metrics,
                "start_node": int(start_nodes[i]),
            }
        )

    print(
        f"  [INFO] generating {num_cycles - 2*methods_per_type} chinese postman cycles..."
    )
    for i in range(2 * methods_per_type, num_cycles):
        cycle = chinese_postman_cycle(subgraph, start_nodes[i])
        metrics = calculate_cycle_metrics(subgraph, cycle)
        cycles.append(
            {
                "method": "chinese_postman",
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
