#!/usr/bin/env python3
"""
main pipeline to orchestrate the entire snow plow optimization system.
run all steps from data classification to final optimization.
"""

import argparse
import subprocess
import json
import pandas as pd
import time
from pathlib import Path


def run_step(script_name, description):
    """
    run a pipeline step and report results.

    args:
        script_name: name of python script to run
        description: description of the step

    returns:
        elapsed time in seconds
    """
    print(f"\n{'='*80}")
    print(f"step: {description}")
    print(f"running: {script_name}")
    print(f"{'='*80}\n")

    start_time = time.time()

    result = subprocess.run(["python3", script_name], capture_output=True, text=True)

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"[ERROR] step failed with return code {result.returncode}")
        print(f"stderr: {result.stderr}")
        raise RuntimeError(f"step {script_name} failed")

    print(result.stdout)
    print(f"\n[PASS] step completed in {elapsed:.2f} seconds")

    return elapsed


def generate_summary_report(args):
    """
    generate comprehensive summary report.

    args:
        args: command line arguments
    """
    print(f"\n{'='*80}")
    print("generating summary report")
    print(f"{'='*80}\n")

    report = []
    report.append("# snow plow optimization system - summary report\n")
    report.append(f"generated: {pd.Timestamp.now()}\n\n")

    # parameters
    report.append("## parameters\n")
    report.append(f"- partition sets: {args.num_partitions}\n")
    report.append(f"- cycles per subgraph: {args.cycles_per_subgraph}\n")
    report.append(f"- snowfall rate: {args.snowfall_rate} inches/hour\n")
    report.append(f"- storm duration: {args.storm_duration} hours\n\n")

    # graph statistics
    if Path("graph_statistics.csv").exists():
        report.append("## graph statistics\n")
        graph_stats = pd.read_csv("graph_statistics.csv")
        for col in graph_stats.columns:
            value = graph_stats[col].values[0]
            if "time" in col:
                report.append(f"- {col}: {value:.2f} minutes ({value/60:.2f} hours)\n")
            elif "length" in col:
                report.append(f"- {col}: {value:.2f} feet ({value/5280:.2f} miles)\n")
            else:
                report.append(f"- {col}: {value}\n")
        report.append("\n")

    # partition statistics
    if Path("partition_statistics.csv").exists():
        report.append("## partition statistics\n")
        partition_stats = pd.read_csv("partition_statistics.csv")
        report.append(f"- total partitions: {len(partition_stats)}\n")
        report.append(
            f"- partition sets: {partition_stats['partition_set_id'].nunique()}\n"
        )
        report.append(
            f"- partitions per set: {partition_stats.groupby('partition_set_id')['partition_id'].nunique().values[0]}\n"
        )
        report.append(
            f"- avg nodes per partition: {partition_stats['num_nodes'].mean():.1f}\n"
        )
        report.append(
            f"- avg priority per partition: {partition_stats['total_priority'].mean():.2f}\n\n"
        )

    # cycle generation statistics
    if Path("cycle_evaluations.csv").exists():
        report.append("## cycle generation and evaluation\n")
        cycle_evals = pd.read_csv("cycle_evaluations.csv")
        report.append(f"- total cycles generated: {len(cycle_evals)}\n")
        report.append(
            f"- avg cycle time: {cycle_evals['cycle_time'].mean():.2f} minutes\n"
        )
        report.append(f"- avg benefit: {cycle_evals['benefit'].mean():.2f}\n")
        report.append(
            f"- avg benefit per minute: {cycle_evals['benefit_per_minute'].mean():.2f}\n"
        )
        report.append(
            f"- best benefit per minute: {cycle_evals['benefit_per_minute'].max():.2f}\n\n"
        )

        # method distribution
        report.append("### method distribution\n")
        method_counts = cycle_evals["method"].value_counts()
        for method, count in method_counts.items():
            pct = (count / len(cycle_evals)) * 100
            report.append(f"- {method}: {count} ({pct:.1f}%)\n")
        report.append("\n")

    # operational plans
    if Path("operational_plans.json").exists():
        report.append("## operational plans\n")
        with open("operational_plans.json", "r") as f:
            plans = json.load(f)

        for plan in plans:
            ps_id = plan["partition_set_id"]
            assignments = plan["assignments"]

            report.append(f"\n### partition set {ps_id}\n")
            report.append(f"- drivers: {len(assignments)}\n")

            total_benefit = sum(a["benefit"] for a in assignments)
            avg_time = sum(a["cycle_time"] for a in assignments) / len(assignments)
            avg_benefit_per_min = sum(
                a["benefit_per_minute"] for a in assignments
            ) / len(assignments)

            report.append(f"- total benefit: {total_benefit:.2f}\n")
            report.append(f"- avg cycle time: {avg_time:.2f} minutes\n")
            report.append(f"- avg benefit per minute: {avg_benefit_per_min:.2f}\n\n")

            report.append("#### driver assignments\n")
            for assignment in assignments:
                report.append(
                    f"- driver {assignment['driver_id']}: "
                    f"partition {assignment['partition_id']}, "
                    f"time={assignment['cycle_time']:.1f}min, "
                    f"benefit/min={assignment['benefit_per_minute']:.2f}, "
                    f"method={assignment['method']}\n"
                )
            report.append("\n")

    # optimization results
    if Path("optimized_cycles.json").exists():
        report.append("## optimization results\n")
        report.append(
            "local optimization was applied using perturbation-based search.\n"
        )
        report.append(
            "perturbation methods: edge swap, vertex insert/remove, 2-opt style moves.\n\n"
        )

    # recommendations
    report.append("## recommendations\n")
    report.append(
        "1. use the operational plan for the partition set with highest total benefit\n"
    )
    report.append(
        "2. drivers should start at designated start nodes in their assigned partitions\n"
    )
    report.append(
        "3. follow the optimized cycle routes to maximize benefit (priority * snow cleared)\n"
    )
    report.append("4. plan for multiple passes during the 12-hour storm\n")
    report.append("5. monitor weather conditions and adjust if snowfall rate changes\n")
    report.append("6. coordinate salt usage based on temperature and priority routes\n")
    report.append("7. ensure driver breaks comply with federal regulations\n\n")

    # write report
    report_file = "optimization_report.md"
    with open(report_file, "w") as f:
        f.writelines(report)

    print(f"[INFO] summary report saved to {report_file}")

    # also print to console
    print("\n" + "".join(report))


def main():
    parser = argparse.ArgumentParser(description="snow plow optimization pipeline")

    parser.add_argument(
        "--num-partitions",
        type=int,
        default=3,
        help="number of partition sets to generate (default: 3)",
    )

    parser.add_argument(
        "--cycles-per-subgraph",
        type=int,
        default=50,
        help="number of candidate cycles per subgraph (default: 50)",
    )

    parser.add_argument(
        "--snowfall-rate",
        type=float,
        default=1.0,
        help="snowfall rate in inches per hour (default: 1.0)",
    )

    parser.add_argument(
        "--storm-duration",
        type=int,
        default=12,
        help="storm duration in hours (default: 12)",
    )

    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="skip road classification step (use existing classified_roads.csv)",
    )

    parser.add_argument(
        "--skip-to",
        type=str,
        choices=[
            "priority",
            "graph",
            "partition",
            "cycles",
            "evaluate",
            "sample",
            "optimize",
        ],
        help="skip to a specific step in the pipeline",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("snow plow optimization pipeline")
    print("=" * 80)
    print(f"\nparameters:")
    print(f"  num partition sets: {args.num_partitions}")
    print(f"  cycles per subgraph: {args.cycles_per_subgraph}")
    print(f"  snowfall rate: {args.snowfall_rate} inches/hour")
    print(f"  storm duration: {args.storm_duration} hours")

    start_time = time.time()
    step_times = {}

    # pipeline steps
    steps = [
        ("classify_roads.py", "classify roads and calculate scores", "classification"),
        ("calculate_priority.py", "calculate priority scores", "priority"),
        ("build_graph.py", "build road network graph", "graph"),
        ("partition_graph.py", "partition graph into subgraphs", "partition"),
        ("generate_cycles.py", "generate candidate cycles", "cycles"),
        ("evaluate_cycles.py", "evaluate cycles with cost function", "evaluate"),
        ("sample_cycles.py", "select top cycles per partition", "sample"),
        ("optimize_routes.py", "optimize routes with local search", "optimize"),
    ]

    # determine which steps to run
    skip_until = None
    if args.skip_to:
        skip_until = args.skip_to
    elif args.skip_classification:
        skip_until = "priority"

    for script, description, step_name in steps:
        if skip_until and step_name != skip_until:
            continue
        skip_until = None  # start running after we reach the skip-to step

        try:
            elapsed = run_step(script, description)
            step_times[step_name] = elapsed
        except RuntimeError as e:
            print(f"\n[ERROR] pipeline failed at step: {script}")
            print(f"error: {e}")
            return 1

    # generate summary report
    generate_summary_report(args)

    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print("pipeline completed successfully")
    print(f"{'='*80}\n")
    print(
        f"total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    print("\nstep timing breakdown:")
    for step_name, elapsed in step_times.items():
        pct = (elapsed / total_time) * 100
        print(f"  {step_name}: {elapsed:.2f}s ({pct:.1f}%)")

    print("\n[INFO] all output files generated:")
    output_files = [
        "classified_roads.csv",
        "classified_roads.geojson",
        "roads_with_priority.csv",
        "road_graph.gpickle",
        "graph_statistics.csv",
        "partitions.json",
        "partition_statistics.csv",
        "candidate_cycles.json",
        "cycle_evaluations.json",
        "cycle_evaluations.csv",
        "selected_cycles.json",
        "operational_plans.json",
        "optimized_cycles.json",
        "optimization_report.md",
    ]

    for file in output_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"  - {file} ({size/1024:.1f} KB)")

    return 0


if __name__ == "__main__":
    exit(main())
