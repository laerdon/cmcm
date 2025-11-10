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
import csv


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


def calculate_cost_with_plow(G, cycle, snowfall_rate, storm_duration, start_time=0):
    """
    calculate cost accumulated if plow follows the cycle repeatedly over the storm duration.
    for each edge, snow accumulates from last visit (or start) until plow arrives.
    the cycle repeats as many times as possible within the storm duration.

    args:
        G: networkx graph
        cycle: list of (from_node, to_node, edge_key) tuples
        snowfall_rate: inches per minute
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

        # time from last visit to end of storm
        time_since_visit = storm_duration - last_visit

        if time_since_visit > 0:
            # cost for snow accumulating after last visit
            cost = edge_data["priority"] * snowfall_rate * (time_since_visit**2) / 2.0
            total_cost += cost

    return total_cost, num_complete_cycles, edges_visited


def calculate_cycle_benefit(G, cycle, snowfall_rate, storm_duration):
    """
    calculate benefit of using this cycle vs not plowing over the full storm duration.
    simulates repeated passes of the cycle throughout the storm.

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
            "cycle_time": 0.0,
            "num_complete_cycles": 0,
            "coverage_ratio": 0.0,
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

    # cost with plow: follow the cycle repeatedly over storm duration
    cost_with_plow, num_complete_cycles, edges_visited = calculate_cost_with_plow(
        G, cycle, snowfall_rate, storm_duration, start_time=0
    )

    # benefit
    benefit = cost_without_plow - cost_with_plow

    # calculate cycle time (single pass)
    cycle_time = sum(G[u][v][k]["travel_time"] for u, v, k in cycle)

    # benefit per minute of storm duration (legacy metric)
    benefit_per_minute = benefit / storm_duration if storm_duration > 0 else 0.0

    # benefit as percentage of baseline cost (efficiency metric)
    benefit_pct = (
        (benefit / cost_without_plow * 100.0) if cost_without_plow > 0 else 0.0
    )

    # coverage ratio: what fraction of partition edges are covered
    # (this will be filled in later if we have partition context)
    coverage_ratio = len(edges_visited) / len(unique_edges) if unique_edges else 0.0

    return {
        "cost_without_plow": float(cost_without_plow),
        "cost_with_plow": float(cost_with_plow),
        "benefit": float(benefit),
        "benefit_per_minute": float(benefit_per_minute),
        "benefit_pct": float(benefit_pct),
        "cycle_time": float(cycle_time),
        "num_complete_cycles": int(num_complete_cycles),
        "coverage_ratio": float(coverage_ratio),
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

def load_road_names():
    """Load road names from CSV and create lookup by edge attributes."""
    print("[INFO] loading road names from roads_with_priority.csv...")
    roads_df = pd.read_csv("roads_with_priority.csv")
    
    # Create multiple lookup strategies
    road_lookup = {}
    
    for _, row in roads_df.iterrows():
        name = row['NAME'] if pd.notna(row['NAME']) and str(row['NAME']).strip() else None
        if not name:
            continue
            
        # Create lookup key using multiple attributes for uniqueness
        priority = round(float(row['priority']), 6)
        length = round(float(row['length_feet']), 2)
        travel_time = round(float(row['travel_time_minutes']), 6)
        
        # Use combination of attributes as key
        key = (priority, length, travel_time)
        if key not in road_lookup:
            road_lookup[key] = name
    
    return road_lookup

def get_road_name(edge_data, road_lookup):
    """Get road name for an edge."""
    priority = round(float(edge_data.get('priority', 0)), 6)
    length = round(float(edge_data.get('length_feet', 0)), 2)
    travel_time = round(float(edge_data.get('travel_time', 0)), 6)
    
    key = (priority, length, travel_time)
    name = road_lookup.get(key)
    
    if name:
        return name
    else:
        return f"Road {priority:.2f}"


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

    print("\n[INFO] exporting top (optimal) cycles per partition...")

    # Load road names
    road_lookup = load_road_names()
    print(f"[INFO] loaded {len(road_lookup)} road names from CSV")

    optimal_cycles = []
    readable_path = "optimal_cycles_readable.txt"
    detailed_path = "optimal_cycles_detailed.txt"

    with open(readable_path, "w") as f_summary, open(detailed_path, "w") as f_detail:
        f_summary.write("=" * 80 + "\n")
        f_summary.write("OPTIMAL SNOW PLOW ROUTES - SUMMARY\n")
        f_summary.write("=" * 80 + "\n\n")
        
        f_detail.write("=" * 80 + "\n")
        f_detail.write("OPTIMAL SNOW PLOW ROUTES - DRIVER INSTRUCTIONS\n")
        f_detail.write("=" * 80 + "\n\n")
        
        for key, cycle_list in sorted(ranked_partitions.items()):
            ps_id, p_id = key
            top_cycle_entry = cycle_list[0]
            eval_data = top_cycle_entry["evaluation"]
            metrics = top_cycle_entry["metrics"]
            cycle = top_cycle_entry["cycle"]

            optimal_cycles.append({
                "partition_set_id": ps_id,
                "partition_id": p_id,
                "method": top_cycle_entry["method"],
                "start_node": top_cycle_entry["start_node"],
                "cycle_time": eval_data["cycle_time"],
                "num_complete_cycles": eval_data["num_complete_cycles"],
                "benefit": eval_data["benefit"],
                "benefit_per_minute": eval_data["benefit_per_minute"],
                "benefit_pct": eval_data["benefit_pct"],
                "cost_without_plow": eval_data["cost_without_plow"],
                "cost_with_plow": eval_data["cost_with_plow"],
                "coverage_ratio": eval_data["coverage_ratio"],
                "total_priority": metrics["total_priority"],
                "unique_edges": metrics["unique_edges"],
            })

            # Write summary
            f_summary.write(f"ZONE {p_id + 1} (Driver {p_id + 1})\n")
            f_summary.write(f"{'-' * 40}\n")
            f_summary.write(f"Starting Node: {top_cycle_entry['start_node']}\n")
            f_summary.write(f"Cycle Time: {eval_data['cycle_time']:.1f} min ({eval_data['cycle_time']/60:.1f} hrs)\n")
            f_summary.write(f"Cycles in 12h Storm: {eval_data['num_complete_cycles']}\n")
            f_summary.write(f"Efficiency: {eval_data['benefit_pct']:.2f}%\n")
            f_summary.write(f"Unique Roads: {metrics['unique_edges']}\n\n")

            # Write detailed instructions
            f_detail.write("=" * 80 + "\n")
            f_detail.write(f"DRIVER {p_id + 1} - ZONE {p_id + 1}\n")
            f_detail.write("=" * 80 + "\n")
            f_detail.write(f"Start at Node {top_cycle_entry['start_node']}\n")
            f_detail.write(f"Cycle Time: {eval_data['cycle_time']:.1f} min | Repeat {eval_data['num_complete_cycles']}x during storm\n")
            f_detail.write(f"Roads: {metrics['unique_edges']} | Steps: {len(cycle)}\n")
            f_detail.write("-" * 80 + "\n\n")
            
            cumulative_time = 0.0
            for step_num, (u, v, k) in enumerate(cycle, 1):
                if G.has_edge(u, v, k):
                    edge_data = G[u][v][k]
                    name = get_road_name(edge_data, road_lookup)
                    
                    travel_time = edge_data.get("travel_time", 0)
                    length_feet = edge_data.get("length_feet", 0)
                    cumulative_time += travel_time
                    
                    f_detail.write(f"{step_num}. {name} ({length_feet:.0f}ft, {travel_time:.1f}min) [{u}→{v}]\n")
                else:
                    f_detail.write(f"{step_num}. [ERROR: Missing edge {u}→{v}]\n")
            
            f_detail.write(f"\nTotal: {cumulative_time:.1f} minutes\n")
            f_detail.write("\n")

    # Save CSV
    pd.DataFrame(optimal_cycles).to_csv("optimal_cycles.csv", index=False)
    print("[INFO] saved optimal_cycles.csv")
    print(f"[INFO] saved {readable_path}")
    print(f"[INFO] saved {detailed_path}")

    return evaluated_cycles


if __name__ == "__main__":
    main()
