#!/usr/bin/env python3
"""
select top k cycles per partition based on evaluation metrics.
analyze distribution and quality of best cycles.
"""

import json
import pandas as pd
import numpy as np


def select_top_cycles(evaluated_cycles, k=10):
    """
    select top k cycles per partition based on benefit per minute.

    args:
        evaluated_cycles: list of evaluated cycle dicts
        k: number of top cycles to select per partition

    returns:
        dict mapping (partition_set_id, partition_id) to list of top cycles
    """
    # group by partition
    partitions = {}
    for cycle_data in evaluated_cycles:
        key = (cycle_data["partition_set_id"], cycle_data["partition_id"])
        if key not in partitions:
            partitions[key] = []
        partitions[key].append(cycle_data)

    # select top k from each partition
    top_cycles = {}
    for key, cycles in partitions.items():
        # sort by benefit per minute
        sorted_cycles = sorted(
            cycles, key=lambda x: x["evaluation"]["benefit_per_minute"], reverse=True
        )
        top_cycles[key] = sorted_cycles[:k]

    return top_cycles


def analyze_top_cycles(top_cycles):
    """
    analyze distribution and quality of top cycles.

    args:
        top_cycles: dict mapping partition key to list of cycles

    returns:
        dict with analysis results
    """
    analysis = {
        "num_partitions": len(top_cycles),
        "cycles_per_partition": {},
        "method_distribution": {},
        "avg_benefit_per_minute": {},
        "avg_cycle_time": {},
    }

    # analyze each partition
    for key, cycles in top_cycles.items():
        ps_id, p_id = key
        partition_name = f"ps{ps_id}_p{p_id}"

        analysis["cycles_per_partition"][partition_name] = len(cycles)

        # method distribution
        methods = [c["method"] for c in cycles]
        method_counts = pd.Series(methods).value_counts().to_dict()
        analysis["method_distribution"][partition_name] = method_counts

        # average metrics
        benefits = [c["evaluation"]["benefit_per_minute"] for c in cycles]
        times = [c["evaluation"]["cycle_time"] for c in cycles]

        analysis["avg_benefit_per_minute"][partition_name] = np.mean(benefits)
        analysis["avg_cycle_time"][partition_name] = np.mean(times)

    return analysis


def create_operational_plan(top_cycles, num_drivers=8):
    """
    create operational plan: assign best cycle to each driver (one per partition).

    args:
        top_cycles: dict mapping partition key to list of cycles
        num_drivers: number of drivers

    returns:
        list of assignments (one cycle per driver)
    """
    # for each partition set, create an assignment plan
    partition_sets = {}
    for key, cycles in top_cycles.items():
        ps_id, p_id = key
        if ps_id not in partition_sets:
            partition_sets[ps_id] = {}
        partition_sets[ps_id][p_id] = cycles[0]  # best cycle

    # create plans
    operational_plans = []
    for ps_id, assignments in partition_sets.items():
        plan = {"partition_set_id": ps_id, "assignments": []}

        for p_id in range(num_drivers):
            if p_id in assignments:
                cycle = assignments[p_id]
                plan["assignments"].append(
                    {
                        "driver_id": p_id,
                        "partition_id": p_id,
                        "cycle": cycle["cycle"],
                        "start_node": cycle["start_node"],
                        "method": cycle["method"],
                        "cycle_time": cycle["evaluation"]["cycle_time"],
                        "benefit": cycle["evaluation"]["benefit"],
                        "benefit_per_minute": cycle["evaluation"]["benefit_per_minute"],
                        "unique_edges": cycle["metrics"]["unique_edges"],
                    }
                )

        operational_plans.append(plan)

    return operational_plans


def main():
    print("[INFO] loading evaluated cycles...")
    with open("cycle_evaluations.json", "r") as f:
        evaluated_cycles = json.load(f)
    print(f"[INFO] loaded {len(evaluated_cycles)} evaluated cycles")

    # select top k cycles per partition
    k = 10  # tunable parameter
    print(f"\n[INFO] selecting top {k} cycles per partition...")
    top_cycles = select_top_cycles(evaluated_cycles, k=k)

    print(f"[INFO] selected cycles for {len(top_cycles)} partitions")

    # analyze top cycles
    analysis = analyze_top_cycles(top_cycles)

    print("\n[ANALYSIS] top cycles analysis:")
    print(f"total partitions: {analysis['num_partitions']}")

    # method distribution across all top cycles
    all_methods = []
    for partition_name, method_counts in analysis["method_distribution"].items():
        for method, count in method_counts.items():
            all_methods.extend([method] * count)

    print(f"\nmethod distribution in top {k} cycles:")
    method_series = pd.Series(all_methods)
    print(method_series.value_counts())

    # average metrics
    print(f"\naverage benefit per minute across partitions:")
    for partition_name in sorted(analysis["avg_benefit_per_minute"].keys()):
        benefit = analysis["avg_benefit_per_minute"][partition_name]
        time = analysis["avg_cycle_time"][partition_name]
        print(f"  {partition_name}: {benefit:.2f} (cycle time: {time:.2f} min)")

    # save top cycles
    output_file = "selected_cycles.json"

    # convert to serializable format
    selected_cycles_serializable = {}
    for key, cycles in top_cycles.items():
        ps_id, p_id = key
        key_str = f"ps{ps_id}_p{p_id}"
        selected_cycles_serializable[key_str] = cycles

    with open(output_file, "w") as f:
        json.dump(selected_cycles_serializable, f, indent=2)

    print(f"\n[INFO] saved selected cycles to {output_file}")

    # create operational plans
    operational_plans = create_operational_plan(top_cycles)

    # save operational plans
    plans_file = "operational_plans.json"
    with open(plans_file, "w") as f:
        json.dump(operational_plans, f, indent=2)

    print(f"[INFO] saved operational plans to {plans_file}")

    # create summary for each plan
    for plan in operational_plans:
        print(f"\n[PLAN] partition set {plan['partition_set_id']}:")
        plan_df = pd.DataFrame(plan["assignments"])
        print(f"  total drivers: {len(plan['assignments'])}")
        print(f"  average cycle time: {plan_df['cycle_time'].mean():.2f} minutes")
        print(f"  total benefit: {plan_df['benefit'].sum():.2f}")
        print(
            f"  average benefit per minute: {plan_df['benefit_per_minute'].mean():.2f}"
        )

        # show each driver assignment
        for assignment in plan["assignments"]:
            print(
                f"    driver {assignment['driver_id']}: "
                f"partition {assignment['partition_id']}, "
                f"time={assignment['cycle_time']:.1f}min, "
                f"benefit/min={assignment['benefit_per_minute']:.2f}, "
                f"method={assignment['method']}"
            )

    return top_cycles, operational_plans


if __name__ == "__main__":
    main()
