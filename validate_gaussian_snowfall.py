#!/usr/bin/env python3
"""
validate plow routes under spatially-varying snowfall using a 2D gaussian distribution.
tests how routes perform when snow intensity varies across ithaca.
"""

import networkx as nx
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd


STORM_DURATION_HOURS = 12


def create_gaussian_snowfall(G, center=None, sigma=None, base_rate=1.0):
    """
    create spatially-varying snowfall rates using 2D gaussian.

    args:
        G: networkx graph with node positions
        center: (x, y) tuple for gaussian center (default: graph centroid)
        sigma: standard deviation in coordinate units (default: auto-scale)
        base_rate: base snowfall rate in inches/hour

    returns:
        dict mapping edge_id -> snowfall_rate (inches/hour)
    """
    # get all node positions
    node_positions = {}
    for node in G.nodes():
        node_positions[node] = (G.nodes[node]["x"], G.nodes[node]["y"])

    # calculate graph centroid if no center provided
    if center is None:
        x_coords = [pos[0] for pos in node_positions.values()]
        y_coords = [pos[1] for pos in node_positions.values()]
        center = (np.mean(x_coords), np.mean(y_coords))
        print(
            f"[INFO] using graph centroid as gaussian center: ({center[0]:.1f}, {center[1]:.1f})"
        )

    # calculate appropriate sigma if not provided (~25% of graph diameter)
    if sigma is None:
        x_coords = [pos[0] for pos in node_positions.values()]
        y_coords = [pos[1] for pos in node_positions.values()]
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        diameter = np.sqrt(x_range**2 + y_range**2)
        sigma = diameter * 0.25
        print(f"[INFO] using auto-calculated sigma: {sigma:.1f}")

    # calculate snowfall rate for each edge
    edge_snowfall_rates = {}

    for u, v, k in G.edges(keys=True):
        # get edge midpoint
        x1, y1 = node_positions[u]
        x2, y2 = node_positions[v]
        midpoint_x = (x1 + x2) / 2.0
        midpoint_y = (y1 + y2) / 2.0

        # calculate distance from center
        dx = midpoint_x - center[0]
        dy = midpoint_y - center[1]
        distance_sq = dx**2 + dy**2

        # gaussian value (peak = 1.0 at center)
        gaussian_value = np.exp(-distance_sq / (2 * sigma**2))

        # scale to range [0.1, 2.0] so peak is 2x base rate, minimum is 0.1x
        multiplier = 0.1 + 1.9 * gaussian_value

        # calculate snowfall rate for this edge
        snowfall_rate = base_rate * multiplier

        edge_snowfall_rates[(u, v, k)] = snowfall_rate

    print(
        f"[INFO] snowfall rate range: {min(edge_snowfall_rates.values()):.2f} - {max(edge_snowfall_rates.values()):.2f} inches/hour"
    )

    return edge_snowfall_rates, center, sigma


def calculate_cost_without_plow_variable(
    edge_priorities, edge_snowfall_rates, time_duration
):
    """
    calculate cost accumulated if no plow visits (with variable snowfall).

    args:
        edge_priorities: dict mapping edge_id -> priority
        edge_snowfall_rates: dict mapping edge_id -> snowfall_rate (inches/min)
        time_duration: minutes

    returns:
        total cost
    """
    total_cost = 0.0
    for edge_id, priority in edge_priorities.items():
        snowfall_rate = edge_snowfall_rates.get(edge_id, 0.0)
        # cost = priority * snowfall_rate * (T^2 / 2)
        cost = priority * snowfall_rate * (time_duration**2) / 2.0
        total_cost += cost
    return total_cost


def calculate_cost_with_variable_snow(
    G, cycle, edge_snowfall_rates, storm_duration, start_time=0
):
    """
    calculate cost with per-edge snowfall rates (variable snow).
    same logic as calculate_cost_with_plow but uses edge-specific snowfall rates.

    args:
        G: networkx graph
        cycle: list of (from_node, to_node, edge_key) tuples
        edge_snowfall_rates: dict mapping edge_id -> snowfall_rate (inches/min)
        storm_duration: total storm duration in minutes
        start_time: when plow starts (minutes from storm start)

    returns:
        tuple of (total_cost, num_complete_cycles, edges_visited_set)
    """
    if not cycle:
        return 0.0, 0, set()

    # track last visit time for each edge
    edge_last_visit = {}
    current_time = start_time
    total_cost = 0.0
    num_complete_cycles = 0
    edges_visited = set()

    # repeat cycle until storm ends
    while current_time < storm_duration:
        cycle_start_time = current_time

        for from_node, to_node, edge_key in cycle:
            edge_data = G[from_node][to_node][edge_key]
            edge_id = (from_node, to_node, edge_key)
            edges_visited.add(edge_id)

            # check if we'd exceed storm duration
            if current_time >= storm_duration:
                break

            # get snowfall rate for this specific edge
            snowfall_rate = edge_snowfall_rates.get(edge_id, 0.0)

            # calculate cost since last visit (or start)
            last_visit = edge_last_visit.get(edge_id, start_time)
            time_since_visit = current_time - last_visit

            # cost = priority * snowfall_rate * time_since_visit^2 / 2
            cost = edge_data["priority"] * snowfall_rate * (time_since_visit**2) / 2.0
            total_cost += cost

            # update last visit time
            edge_last_visit[edge_id] = current_time

            # advance time by travel time
            current_time += edge_data["travel_time"]

        # check if we completed the full cycle
        if current_time <= storm_duration:
            num_complete_cycles += 1

        # if cycle takes 0 time (shouldn't happen), break to avoid infinite loop
        if current_time == cycle_start_time:
            break

    # add residual cost for edges not visited again before storm ends
    for edge_id, last_visit in edge_last_visit.items():
        from_node, to_node, edge_key = edge_id
        edge_data = G[from_node][to_node][edge_key]
        snowfall_rate = edge_snowfall_rates.get(edge_id, 0.0)

        # time from last visit to end of storm
        time_since_visit = storm_duration - last_visit

        if time_since_visit > 0:
            # cost for snow accumulating after last visit
            cost = edge_data["priority"] * snowfall_rate * (time_since_visit**2) / 2.0
            total_cost += cost

    return total_cost, num_complete_cycles, edges_visited


def evaluate_cycles_gaussian(
    G, cycles_data, center=None, sigma=None, base_rate=1.0, storm_duration=720
):
    """
    evaluate all cycles under gaussian snowfall.

    args:
        G: networkx graph
        cycles_data: list of cycle dicts with 'cycle' field
        center: (x, y) tuple for gaussian center
        sigma: gaussian standard deviation
        base_rate: base snowfall rate in inches/hour
        storm_duration: storm duration in minutes

    returns:
        list of evaluated cycles with gaussian metrics
    """
    print("[INFO] creating gaussian snowfall model...")
    edge_snowfall_rates, actual_center, actual_sigma = create_gaussian_snowfall(
        G, center, sigma, base_rate
    )

    # convert to per-minute rates
    edge_snowfall_rates_per_min = {
        edge_id: rate / 60.0 for edge_id, rate in edge_snowfall_rates.items()
    }

    print("[INFO] evaluating cycles under gaussian snowfall...")
    evaluated_cycles = []

    for i, cycle_data in enumerate(cycles_data):
        if i % 100 == 0:
            print(f"  [PROGRESS] evaluated {i}/{len(cycles_data)} cycles...")

        cycle = cycle_data["cycle"]

        # get unique edges and their priorities
        edge_priorities = {}
        for from_node, to_node, edge_key in cycle:
            edge_id = (from_node, to_node, edge_key)
            if edge_id not in edge_priorities:
                edge_priorities[edge_id] = G[from_node][to_node][edge_key]["priority"]

        # calculate cost without plow
        cost_without_plow = calculate_cost_without_plow_variable(
            edge_priorities, edge_snowfall_rates_per_min, storm_duration
        )

        # calculate cost with plow
        cost_with_plow, num_complete_cycles, edges_visited = (
            calculate_cost_with_variable_snow(
                G, cycle, edge_snowfall_rates_per_min, storm_duration, start_time=0
            )
        )

        # calculate benefit
        benefit = cost_without_plow - cost_with_plow
        benefit_per_minute = benefit / storm_duration if storm_duration > 0 else 0.0

        # calculate cycle time
        cycle_time = sum(G[u][v][k]["travel_time"] for u, v, k in cycle)

        # add evaluation
        cycle_data_copy = cycle_data.copy()
        cycle_data_copy["gaussian_evaluation"] = {
            "cost_without_plow": float(cost_without_plow),
            "cost_with_plow": float(cost_with_plow),
            "benefit": float(benefit),
            "benefit_per_minute": float(benefit_per_minute),
            "cycle_time": float(cycle_time),
            "num_complete_cycles": int(num_complete_cycles),
            "coverage_ratio": float(
                len(edges_visited) / len(edge_priorities) if edge_priorities else 0.0
            ),
        }

        evaluated_cycles.append(cycle_data_copy)

    return evaluated_cycles, actual_center, actual_sigma, edge_snowfall_rates


def compare_uniform_vs_gaussian(uniform_results, gaussian_results):
    """
    compare cycle performance under uniform vs gaussian conditions.

    args:
        uniform_results: list of cycles with 'evaluation' field
        gaussian_results: list of cycles with 'gaussian_evaluation' field

    returns:
        comparison dataframe
    """
    comparison_rows = []

    for uniform, gaussian in zip(uniform_results, gaussian_results):
        uniform_benefit = uniform.get("evaluation", {}).get("benefit_per_minute", 0.0)
        gaussian_benefit = gaussian.get("gaussian_evaluation", {}).get(
            "benefit_per_minute", 0.0
        )

        change_pct = (
            ((gaussian_benefit - uniform_benefit) / uniform_benefit) * 100
            if uniform_benefit > 0
            else 0.0
        )

        comparison_rows.append(
            {
                "partition_set_id": uniform["partition_set_id"],
                "partition_id": uniform["partition_id"],
                "method": uniform["method"],
                "start_node": uniform["start_node"],
                "uniform_benefit": uniform_benefit,
                "gaussian_benefit": gaussian_benefit,
                "change_pct": change_pct,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    # print summary
    print("\n" + "=" * 60)
    print("GAUSSIAN SNOWFALL VALIDATION RESULTS")
    print("=" * 60)

    # overall statistics
    print(f"\naverage benefit change: {comparison_df['change_pct'].mean():.2f}%")
    print(f"std dev of change: {comparison_df['change_pct'].std():.2f}%")

    # partition-level analysis
    print("\npartition-level summary:")
    print("-" * 60)
    partition_summary = comparison_df.groupby(["partition_set_id", "partition_id"]).agg(
        {
            "uniform_benefit": "mean",
            "gaussian_benefit": "mean",
            "change_pct": "mean",
        }
    )
    partition_summary["change_pct"] = partition_summary.apply(
        lambda row: (
            (
                (row["gaussian_benefit"] - row["uniform_benefit"])
                / row["uniform_benefit"]
            )
            * 100
            if row["uniform_benefit"] > 0
            else 0.0
        ),
        axis=1,
    )

    for idx, row in partition_summary.iterrows():
        ps_id, p_id = idx
        print(
            f"partition set {ps_id}, partition {p_id}: "
            f"uniform={row['uniform_benefit']:.1f}, "
            f"gaussian={row['gaussian_benefit']:.1f} "
            f"({row['change_pct']:+.1f}%)"
        )

    # top affected cycles
    print("\ntop 10 cycles most negatively affected:")
    print("-" * 60)
    worst_cycles = comparison_df.nsmallest(10, "change_pct")
    for idx, row in worst_cycles.iterrows():
        print(
            f"ps={row['partition_set_id']}, p={row['partition_id']}, "
            f"method={row['method']}: {row['change_pct']:+.1f}%"
        )

    print("\ntop 10 cycles most positively affected:")
    print("-" * 60)
    best_cycles = comparison_df.nlargest(10, "change_pct")
    for idx, row in best_cycles.iterrows():
        print(
            f"ps={row['partition_set_id']}, p={row['partition_id']}, "
            f"method={row['method']}: {row['change_pct']:+.1f}%"
        )

    return comparison_df


def visualize_gaussian_snowfall(
    G, edge_snowfall_rates, center, sigma, output_file="gaussian_snowfall_map.png"
):
    """
    create map showing snowfall intensity across ithaca.

    args:
        G: networkx graph
        edge_snowfall_rates: dict mapping edge_id -> snowfall_rate
        center: (x, y) gaussian center
        sigma: gaussian standard deviation
        output_file: output filename
    """
    print(f"[INFO] creating visualization: {output_file}")

    # get node positions
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}

    # create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # get snowfall rate range for colormap
    rates = list(edge_snowfall_rates.values())
    min_rate = min(rates)
    max_rate = max(rates)

    norm = Normalize(vmin=min_rate, vmax=max_rate)
    cmap = plt.cm.YlOrRd

    # draw edges colored by snowfall rate
    for (u, v, k), rate in edge_snowfall_rates.items():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        color = cmap(norm(rate))
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.5, alpha=0.7)

    # draw gaussian center
    ax.scatter(
        center[0],
        center[1],
        c="blue",
        s=200,
        marker="*",
        edgecolors="black",
        linewidths=2,
        label="gaussian center",
        zorder=10,
    )

    # draw gaussian contours
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)

    # calculate gaussian values on grid
    Z = np.exp(-((X - center[0]) ** 2 + (Y - center[1]) ** 2) / (2 * sigma**2))

    # plot contours at 0.1, 0.3, 0.5, 0.7, 0.9
    contours = ax.contour(
        X,
        Y,
        Z,
        levels=[0.1, 0.3, 0.5, 0.7, 0.9],
        colors="black",
        linewidths=1,
        alpha=0.4,
    )
    ax.clabel(contours, inline=True, fontsize=8)

    # colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("snowfall rate (inches/hour)", rotation=270, labelpad=20)

    # labels
    ax.set_xlabel("x coordinate (feet)")
    ax.set_ylabel("y coordinate (feet)")
    ax.set_title(
        f"gaussian snowfall distribution\ncenter=({center[0]:.1f}, {center[1]:.1f}), sigma={sigma:.1f}"
    )
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"[INFO] saved visualization to {output_file}")
    plt.close()


def main():
    """main validation script."""
    print("[INFO] loading road graph...")
    with open("road_graph.gpickle", "rb") as f:
        G = pickle.load(f)
    print(
        f"[INFO] loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )

    print("[INFO] loading uniform evaluation results...")
    with open("cycle_evaluations.json", "r") as f:
        uniform_results = json.load(f)
    print(f"[INFO] loaded {len(uniform_results)} evaluated cycles")

    # parameters
    base_rate = 1.0  # inches/hour
    storm_duration = STORM_DURATION_HOURS * 60  # minutes

    # evaluate under gaussian snowfall
    gaussian_results, center, sigma, edge_snowfall_rates = evaluate_cycles_gaussian(
        G,
        uniform_results,
        center=None,
        sigma=None,
        base_rate=base_rate,
        storm_duration=storm_duration,
    )

    # save gaussian results
    output_file = "cycle_evaluations_gaussian.json"
    with open(output_file, "w") as f:
        json.dump(gaussian_results, f, indent=2)
    print(f"\n[INFO] saved gaussian evaluations to {output_file}")

    # compare results
    comparison_df = compare_uniform_vs_gaussian(uniform_results, gaussian_results)

    # save comparison
    comparison_df.to_csv("gaussian_comparison.csv", index=False)
    print(f"\n[INFO] saved comparison to gaussian_comparison.csv")

    # create visualization
    visualize_gaussian_snowfall(G, edge_snowfall_rates, center, sigma)

    return gaussian_results, comparison_df


if __name__ == "__main__":
    main()
