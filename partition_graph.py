#!/usr/bin/env python3
"""
partition road graph into 8 subgraphs using recursive balanced partitioning.
partitions are balanced by total priority and maintain connectivity.
"""

import networkx as nx
import numpy as np
import json
import pickle
from collections import defaultdict


def calculate_partition_priority(G, nodes):
    """
    calculate total priority for a set of nodes in the graph.

    args:
        G: networkx graph
        nodes: set of node ids

    returns:
        total priority
    """
    total_priority = 0.0
    for u, v, data in G.edges(data=True):
        if u in nodes and v in nodes:
            total_priority += data["priority"]
    return total_priority


def spectral_partition(G, nodes):
    """
    partition a subgraph into two parts using spectral clustering.

    args:
        G: networkx graph
        nodes: set of nodes to partition

    returns:
        tuple of (partition1, partition2) as sets of nodes
    """
    # create subgraph
    subgraph = G.subgraph(nodes).copy()

    if len(nodes) <= 2:
        # base case: split arbitrarily
        nodes_list = list(nodes)
        return {nodes_list[0]}, {nodes_list[1]} if len(nodes) > 1 else set()

    try:
        # use spectral partitioning via laplacian eigenvector (fiedler vector)
        # convert to undirected for spectral analysis
        G_undirected = subgraph.to_undirected()

        # compute laplacian matrix
        laplacian = nx.laplacian_matrix(G_undirected).todense()

        # compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        # second smallest eigenvalue's eigenvector (fiedler vector)
        fiedler_vector = eigenvectors[:, 1]

        # partition based on sign of fiedler vector
        nodes_array = np.array(list(G_undirected.nodes()))
        partition1 = set(nodes_array[fiedler_vector >= 0])
        partition2 = set(nodes_array[fiedler_vector < 0])

        # ensure both partitions are non-empty
        if len(partition1) == 0 or len(partition2) == 0:
            # fallback: split by median
            median_idx = len(nodes_array) // 2
            partition1 = set(nodes_array[:median_idx])
            partition2 = set(nodes_array[median_idx:])

        return partition1, partition2

    except Exception as e:
        print(f"[WARNING] spectral partitioning failed: {e}, using fallback")
        # fallback: simple split
        nodes_list = list(nodes)
        mid = len(nodes_list) // 2
        return set(nodes_list[:mid]), set(nodes_list[mid:])


def balanced_partition(G, nodes, target_partitions=8):
    """
    recursively partition graph into balanced subgraphs.

    args:
        G: networkx graph
        nodes: set of nodes to partition
        target_partitions: number of partitions to create (must be power of 2)

    returns:
        list of node sets, one per partition
    """
    if target_partitions == 1:
        return [nodes]

    # partition into two
    partition1, partition2 = spectral_partition(G, nodes)

    # recursively partition each half
    half_target = target_partitions // 2
    partitions1 = balanced_partition(G, partition1, half_target)
    partitions2 = balanced_partition(G, partition2, half_target)

    return partitions1 + partitions2


def refine_partition_balance(G, partitions, max_iterations=10):
    """
    refine partitions to balance priority using local moves.

    args:
        G: networkx graph
        partitions: list of node sets
        max_iterations: maximum refinement iterations

    returns:
        refined list of node sets
    """
    for iteration in range(max_iterations):
        priorities = [calculate_partition_priority(G, p) for p in partitions]

        # find most and least loaded partitions
        max_idx = np.argmax(priorities)
        min_idx = np.argmin(priorities)

        if priorities[max_idx] - priorities[min_idx] < 0.01:
            break  # balanced enough

        # try to move boundary nodes from max to min
        boundary_nodes = set()
        for node in partitions[max_idx]:
            for neighbor in G.neighbors(node):
                if neighbor in partitions[min_idx]:
                    boundary_nodes.add(node)
                    break

        if not boundary_nodes:
            break  # no boundary nodes to move

        # move one boundary node
        node_to_move = list(boundary_nodes)[0]
        partitions[max_idx].remove(node_to_move)
        partitions[min_idx].add(node_to_move)

    return partitions


def create_partition_set(G, random_seed=None):
    """
    create one set of 8 partitions.

    args:
        G: networkx graph
        random_seed: random seed for reproducibility

    returns:
        list of 8 node sets
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    all_nodes = set(G.nodes())

    # create initial partition
    partitions = balanced_partition(G, all_nodes, target_partitions=8)

    # refine for balance
    partitions = refine_partition_balance(G, partitions)

    # convert to list of sets
    partitions = [set(p) for p in partitions]

    return partitions


def analyze_partitions(G, partitions):
    """
    analyze quality of partitions.

    args:
        G: networkx graph
        partitions: list of node sets

    returns:
        dict with statistics
    """
    stats = {
        "num_partitions": len(partitions),
        "partition_sizes": [],
        "partition_priorities": [],
        "partition_edges": [],
        "partition_total_length": [],
    }

    for i, partition in enumerate(partitions):
        subgraph = G.subgraph(partition)
        priority = calculate_partition_priority(G, partition)
        total_length = sum(
            data["length_feet"] for u, v, data in subgraph.edges(data=True)
        )

        stats["partition_sizes"].append(len(partition))
        stats["partition_priorities"].append(priority)
        stats["partition_edges"].append(subgraph.number_of_edges())
        stats["partition_total_length"].append(total_length)

    return stats


def main():
    print("[INFO] loading road graph...")
    with open("road_graph.gpickle", "rb") as f:
        G = pickle.load(f)
    print(
        f"[INFO] loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )

    # create multiple partition sets
    num_partition_sets = 3  # tunable parameter
    partition_sets = []

    print(f"\n[INFO] creating {num_partition_sets} partition sets...")

    for i in range(num_partition_sets):
        print(f"\n[INFO] creating partition set {i+1}/{num_partition_sets}...")
        partitions = create_partition_set(G, random_seed=i)

        # analyze quality
        stats = analyze_partitions(G, partitions)

        print(f"partition sizes: {stats['partition_sizes']}")
        print(
            f"partition priorities: {[f'{p:.2f}' for p in stats['partition_priorities']]}"
        )
        print(
            f"priority balance (std/mean): {np.std(stats['partition_priorities'])/np.mean(stats['partition_priorities']):.3f}"
        )

        partition_sets.append(partitions)

    # save partitions
    output_file = "partitions.json"

    # convert node sets to lists for json serialization
    partition_sets_serializable = []
    for partition_set in partition_sets:
        partition_sets_serializable.append(
            [[int(node) for node in p] for p in partition_set]
        )

    with open(output_file, "w") as f:
        json.dump(partition_sets_serializable, f, indent=2)

    print(f"\n[INFO] saved {num_partition_sets} partition sets to {output_file}")

    # save detailed statistics
    all_stats = []
    for i, partitions in enumerate(partition_sets):
        stats = analyze_partitions(G, partitions)
        stats["partition_set_id"] = i
        all_stats.append(stats)

    # create summary dataframe
    import pandas as pd

    summary_rows = []
    for stats in all_stats:
        for j in range(stats["num_partitions"]):
            summary_rows.append(
                {
                    "partition_set_id": stats["partition_set_id"],
                    "partition_id": j,
                    "num_nodes": stats["partition_sizes"][j],
                    "total_priority": stats["partition_priorities"][j],
                    "num_edges": stats["partition_edges"][j],
                    "total_length_feet": stats["partition_total_length"][j],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("partition_statistics.csv", index=False)
    print(f"[INFO] saved partition statistics to partition_statistics.csv")

    return partition_sets


if __name__ == "__main__":
    main()
