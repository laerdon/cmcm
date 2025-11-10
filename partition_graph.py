#!/usr/bin/env python3
"""
partition road graph into 8 subgraphs using growing contiguous partitions.
optimizes for both travel time compactness and priority balance.
supports parallel generation of multiple partition sets using multiprocessing.
"""

import networkx as nx
import numpy as np
import json
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


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


def calculate_avg_pairwise_travel_time(G, nodes, sample_size=50):
    """
    estimate average pairwise shortest path travel time within a partition.
    uses sampling for efficiency with large partitions.

    args:
        G: networkx graph
        nodes: set of nodes in partition
        sample_size: number of pairs to sample for estimation

    returns:
        average pairwise travel time (minutes)
    """
    if len(nodes) < 2:
        return 0.0

    nodes_list = list(nodes)
    subgraph = G.subgraph(nodes_list)

    # for small partitions, compute all pairs
    if len(nodes_list) <= 10:
        total_time = 0.0
        count = 0
        for i, u in enumerate(nodes_list):
            for v in nodes_list[i + 1 :]:
                try:
                    path_length = nx.shortest_path_length(
                        subgraph, u, v, weight="travel_time"
                    )
                    total_time += path_length
                    count += 1
                except nx.NetworkXNoPath:
                    # nodes not connected within partition
                    total_time += 1000.0  # penalty for disconnection
                    count += 1
        return total_time / count if count > 0 else 0.0

    # for large partitions, sample random pairs
    total_time = 0.0
    for _ in range(sample_size):
        u, v = np.random.choice(nodes_list, 2, replace=False)
        try:
            path_length = nx.shortest_path_length(subgraph, u, v, weight="travel_time")
            total_time += path_length
        except nx.NetworkXNoPath:
            total_time += 1000.0  # penalty for disconnection

    return total_time / sample_size


def select_seed_nodes(G, num_seeds=8, method="priority_weighted", random_seed=None):
    """
    select seed nodes for growing partitions.

    args:
        G: networkx graph
        num_seeds: number of seeds to select
        method: 'random', 'geographic', or 'priority_weighted'
        random_seed: random seed for reproducibility

    returns:
        list of seed node ids
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    all_nodes = list(G.nodes())

    if method == "random":
        # pure random selection
        return list(np.random.choice(all_nodes, num_seeds, replace=False))

    elif method == "geographic":
        # spread seeds geographically using k-means-style initialization
        # get node positions
        positions = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in all_nodes])

        # simple k-means++ style initialization
        seeds = []
        first_seed = np.random.choice(all_nodes)
        seeds.append(first_seed)

        for _ in range(num_seeds - 1):
            # find node farthest from existing seeds
            max_min_dist = -1
            best_node = None
            for node_idx, node in enumerate(all_nodes):
                if node in seeds:
                    continue
                # compute min distance to existing seeds
                min_dist = float("inf")
                for seed in seeds:
                    seed_idx = all_nodes.index(seed)
                    dist = np.linalg.norm(positions[node_idx] - positions[seed_idx])
                    min_dist = min(min_dist, dist)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_node = node
            seeds.append(best_node)

        return seeds

    elif method == "priority_weighted":
        # select seeds weighted by local priority (nodes in high-priority areas)
        # calculate node importance as sum of adjacent edge priorities
        node_importance = {}
        for node in all_nodes:
            importance = 0.0
            for neighbor in G.neighbors(node):
                for edge_key in G[node][neighbor]:
                    importance += G[node][neighbor][edge_key]["priority"]
            node_importance[node] = importance

        # normalize to probabilities
        total_importance = sum(node_importance.values())
        if total_importance > 0:
            probs = np.array([node_importance[n] for n in all_nodes])
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(all_nodes)) / len(all_nodes)

        # add some randomness - mix with uniform distribution
        probs = 0.7 * probs + 0.3 * np.ones(len(all_nodes)) / len(all_nodes)
        probs = probs / probs.sum()

        return list(np.random.choice(all_nodes, num_seeds, replace=False, p=probs))


def grow_partitions(G, seed_nodes, alpha=0.5, max_iterations=10000, verbose=True):
    """
    grow partitions from seed nodes using dual objective optimization.

    args:
        G: networkx graph
        seed_nodes: list of initial seed node ids
        alpha: weight for compactness objective (0 to 1)
               alpha=1: pure compactness (min travel time)
               alpha=0: pure priority balance
        max_iterations: maximum growth iterations
        verbose: print progress

    returns:
        list of node sets, one per partition
    """
    num_partitions = len(seed_nodes)
    partitions = [set([seed]) for seed in seed_nodes]
    assigned = set(seed_nodes)
    all_nodes = set(G.nodes())
    unassigned = all_nodes - assigned

    # precompute total graph priority for normalization
    total_graph_priority = sum(data["priority"] for u, v, data in G.edges(data=True))
    target_priority = total_graph_priority / num_partitions

    iteration = 0
    while unassigned and iteration < max_iterations:
        if verbose and iteration % 100 == 0:
            print(f"[INFO] iteration {iteration}, {len(unassigned)} nodes remaining")

        # find all boundary nodes (unassigned nodes adjacent to assigned nodes)
        boundary_candidates = []
        for node in unassigned:
            adjacent_partitions = set()
            for neighbor in G.neighbors(node):
                for p_idx, partition in enumerate(partitions):
                    if neighbor in partition:
                        adjacent_partitions.add(p_idx)

            if adjacent_partitions:
                boundary_candidates.append((node, adjacent_partitions))

        if not boundary_candidates:
            # no boundary nodes found - graph might be disconnected
            # assign remaining nodes to nearest partition by distance
            if verbose:
                print(
                    f"[WARNING] no boundary nodes found, {len(unassigned)} nodes isolated"
                )
            for node in list(unassigned):
                # assign to random partition
                p_idx = np.random.randint(num_partitions)
                partitions[p_idx].add(node)
            break

        # evaluate cost for each candidate assignment
        best_cost = float("inf")
        best_assignment = None

        for node, adjacent_partitions in boundary_candidates:
            for p_idx in adjacent_partitions:
                # compute cost of adding node to partition p_idx

                # 1. compactness cost: increase in avg pairwise travel time
                # approximate by average distance from node to partition centroid
                test_partition = partitions[p_idx] | {node}

                # simplified compactness: avg travel time from new node to existing nodes
                if len(partitions[p_idx]) > 0:
                    sample_nodes = list(partitions[p_idx])
                    if len(sample_nodes) > 10:
                        sample_nodes = list(
                            np.random.choice(sample_nodes, 10, replace=False)
                        )

                    travel_times = []
                    for existing_node in sample_nodes:
                        try:
                            tt = nx.shortest_path_length(
                                G, node, existing_node, weight="travel_time"
                            )
                            travel_times.append(tt)
                        except nx.NetworkXNoPath:
                            travel_times.append(1000.0)  # large penalty

                    compactness_cost = np.mean(travel_times)
                else:
                    compactness_cost = 0.0

                # 2. priority balance cost: deviation from target priority
                current_priority = calculate_partition_priority(G, partitions[p_idx])

                # estimate priority contribution of adding this node
                priority_contrib = 0.0
                for neighbor in G.neighbors(node):
                    if neighbor in partitions[p_idx]:
                        for edge_key in G[node][neighbor]:
                            priority_contrib += G[node][neighbor][edge_key]["priority"]

                new_priority = current_priority + priority_contrib
                priority_imbalance = abs(new_priority - target_priority)

                # normalize costs to comparable scales
                # compactness in minutes, priority imbalance as % of target
                normalized_compactness = compactness_cost / 60.0  # normalize to hours
                normalized_priority = (
                    priority_imbalance / target_priority if target_priority > 0 else 0.0
                )

                # combined cost with weighting
                total_cost = (
                    alpha * normalized_compactness + (1 - alpha) * normalized_priority
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_assignment = (node, p_idx)

        # make best assignment
        if best_assignment:
            node, p_idx = best_assignment
            partitions[p_idx].add(node)
            unassigned.remove(node)
            assigned.add(node)
        else:
            # no valid assignment found
            break

        iteration += 1

    if verbose:
        print(f"[INFO] growing completed in {iteration} iterations")
        print(f"[INFO] {len(unassigned)} nodes remaining unassigned")

    return partitions


def create_partition_set(
    G, random_seed=None, alpha=0.5, seed_method="priority_weighted"
):
    """
    create one set of 8 partitions using growing algorithm.

    args:
        G: networkx graph
        random_seed: random seed for reproducibility
        alpha: weight for compactness vs priority balance (0 to 1)
        seed_method: method for selecting seed nodes

    returns:
        list of 8 node sets
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        import random

        random.seed(random_seed)

    # select seed nodes
    seed_nodes = select_seed_nodes(
        G, num_seeds=1, method=seed_method, random_seed=random_seed
    )

    print(f"[INFO] selected seed nodes: {seed_nodes}")
    print(f"[INFO] using alpha={alpha} (compactness weight)")

    # grow partitions
    partitions = grow_partitions(
        G, seed_nodes, alpha=alpha, max_iterations=10000, verbose=True
    )

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
        "partition_avg_travel_time": [],
    }

    for i, partition in enumerate(partitions):
        subgraph = G.subgraph(partition)
        priority = calculate_partition_priority(G, partition)
        total_length = sum(
            data["length_feet"] for u, v, data in subgraph.edges(data=True)
        )
        avg_travel_time = calculate_avg_pairwise_travel_time(G, partition)

        stats["partition_sizes"].append(len(partition))
        stats["partition_priorities"].append(priority)
        stats["partition_edges"].append(subgraph.number_of_edges())
        stats["partition_total_length"].append(total_length)
        stats["partition_avg_travel_time"].append(avg_travel_time)

    return stats


def create_single_partition_set(args):
    """
    wrapper function for parallel partition set creation.

    args:
        args: tuple of (G, partition_id, alpha, seed_method, random_seed)

    returns:
        tuple of (partition_id, partitions, stats)
    """
    G, partition_id, alpha, seed_method, random_seed = args

    print(
        f"[INFO] worker {partition_id}: creating partition set with alpha={alpha}, seed_method={seed_method}"
    )

    partitions = create_partition_set(
        G, random_seed=random_seed, alpha=alpha, seed_method=seed_method
    )

    stats = analyze_partitions(G, partitions)

    print(f"[INFO] worker {partition_id}: completed")
    print(
        f"  priority balance (std/mean): {np.std(stats['partition_priorities'])/np.mean(stats['partition_priorities']):.3f}"
    )
    print(
        f"  travel time balance (std/mean): {np.std(stats['partition_avg_travel_time'])/np.mean(stats['partition_avg_travel_time']):.3f}"
    )

    return partition_id, partitions, stats


def main():
    print("[INFO] loading road graph...")
    with open("road_graph.gpickle", "rb") as f:
        G = pickle.load(f)
    print(
        f"[INFO] loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )

    # hyperparameters for partitioning
    num_partition_sets = 1  # number of different partition sets to try
    alpha_values = [0.3, 0.5, 0.7]  # compactness weights to try
    seed_methods = [
        "priority_weighted",
        "geographic",
        "random",
    ]  # seed selection methods

    partition_sets = []

    print(f"\n[INFO] creating {num_partition_sets} partition sets...")
    print(f"[INFO] alpha values: {alpha_values}")
    print(f"[INFO] seed methods: {seed_methods}")

    # determine number of parallel workers (use cpu count, capped at num_partition_sets)
    num_workers = min(mp.cpu_count(), num_partition_sets, 8)
    print(f"[INFO] using {num_workers} parallel workers")

    # prepare arguments for parallel execution
    parallel_args = []
    for i in range(num_partition_sets):
        alpha = alpha_values[i % len(alpha_values)]
        seed_method = seed_methods[i % len(seed_methods)]
        parallel_args.append((G, i, alpha, seed_method, i))

    # execute partition creation in parallel
    print(f"\n[INFO] starting parallel partition generation...")
    results = []

    if num_partition_sets > 1 and num_workers > 1:
        # use multiprocessing for multiple partition sets
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(create_single_partition_set, args): args[1]
                for args in parallel_args
            }

            for future in as_completed(futures):
                partition_id, partitions, stats = future.result()
                results.append((partition_id, partitions, stats))
    else:
        # sequential execution for single partition set
        for args in parallel_args:
            partition_id, partitions, stats = create_single_partition_set(args)
            results.append((partition_id, partitions, stats))

    # sort results by partition_id to maintain order
    results.sort(key=lambda x: x[0])

    # extract partition sets and print summaries
    print(f"\n[PASS] completed all partition sets")
    for partition_id, partitions, stats in results:
        print(f"\n[INFO] partition set {partition_id}:")
        print(f"  partition sizes: {stats['partition_sizes']}")
        print(
            f"  partition priorities: {[f'{p:.2f}' for p in stats['partition_priorities']]}"
        )
        print(
            f"  partition avg travel times: {[f'{t:.2f}' for t in stats['partition_avg_travel_time']]} minutes"
        )
        print(
            f"  priority balance (std/mean): {np.std(stats['partition_priorities'])/np.mean(stats['partition_priorities']):.3f}"
        )
        print(
            f"  travel time balance (std/mean): {np.std(stats['partition_avg_travel_time'])/np.mean(stats['partition_avg_travel_time']):.3f}"
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
                    "avg_travel_time_minutes": stats["partition_avg_travel_time"][j],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("partition_statistics.csv", index=False)
    print(f"[INFO] saved partition statistics to partition_statistics.csv")

    # print summary statistics
    print(f"\n[SUMMARY] overall statistics:")
    print(
        f"  priority balance (std/mean): {summary_df.groupby('partition_set_id')['total_priority'].std().mean() / summary_df['total_priority'].mean():.3f}"
    )
    print(
        f"  avg travel time range: {summary_df['avg_travel_time_minutes'].min():.2f} - {summary_df['avg_travel_time_minutes'].max():.2f} minutes"
    )

    return partition_sets


if __name__ == "__main__":
    main()

    # Graph Visualization
    import networkx as nx
    import matplotlib.pyplot as plt
    import json
    import pickle
    import numpy as np
    from matplotlib.patches import Polygon, Patch
    from scipy.spatial import ConvexHull

    # Load the saved graph and partition data
    with open("road_graph.gpickle", "rb") as f:
        G = pickle.load(f)

    with open("partitions.json", "r") as f:
        partition_sets = json.load(f)

    # Pick the first partition set to visualize
    partitions = partition_sets[0]

    # Map node → partition index
    node_to_partition = {}
    for i, partition in enumerate(partitions):
        for node in partition:
            node_to_partition[node] = i

    # Choose a color palette (adjust as needed)
    num_partitions = len(partitions)
    colors = plt.cm.get_cmap("tab10", num_partitions)

    # Use node coordinates if available; otherwise, use a layout
    if "x" in list(G.nodes(data=True))[0][1]:
        pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
    else:
        pos = nx.spring_layout(G, seed=42)

    # Create figure and axis
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # 1️⃣ Draw faint gray edges (no arrows)
    nx.draw_networkx_edges(
        G, pos, edge_color="lightgray", alpha=0.6, width=0.8, arrows=False
    )

    # 2️⃣ Draw rigid shaded polygons for each partition
    legend_handles = []
    for i, partition in enumerate(partitions):
        color = np.array(colors(i))  # RGBA
        points = np.array([pos[n] for n in partition if n in pos])

        if len(points) >= 3:  # Need at least 3 points for a hull
            try:
                hull = ConvexHull(points)
                polygon = Polygon(
                    points[hull.vertices],
                    closed=True,
                    facecolor=color,
                    alpha=0.15,  # Shading transparency
                    edgecolor="none",
                )
                ax.add_patch(polygon)

                # Add this zone to the legend
                legend_handles.append(
                    Patch(facecolor=color, edgecolor=color, label=f"Zone {i + 1}")
                )

            except Exception as e:
                print(f"Warning: Convex hull failed for zone {i + 1}: {e}")

    # 3️⃣ Draw nodes (colored by partition)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[colors(node_to_partition[n]) for n in G.nodes()],
        node_size=15,
        alpha=0.9,
    )

    # 4️⃣ Add legend on the right side
    ax.legend(
        handles=legend_handles,
        title="Zones",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
    )

    # 5️⃣ Styling and output
    plt.title("Graph Partitioning of Ithaca into 8 Zones", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        "partition_visualization_shaded_with_legend.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
