#!/usr/bin/env python3
"""
calculate priority scores for each road segment using weighted equation.
priority = 0.31*sEM + 0.25*sGR + 0.23*sGP + 0.13*sRC + 0.08*sST
also calculate travel time based on 25 mph plow speed.
"""

import pandas as pd
import numpy as np


# weights for priority calculation
WEIGHT_EMERGENCY = 0.31
WEIGHT_GRADUATION_ROUTE = 0.25
WEIGHT_GRADUATION_PARKING = 0.23
WEIGHT_ROAD_CLASS = 0.13
WEIGHT_STEEPNESS = 0.08

# plow speed in mph
PLOW_SPEED_MPH = 25


def calculate_priority(row):
    """
    calculate priority score for a road segment.

    args:
        row: pandas series with score columns

    returns:
        priority score (0-1 scale)
    """
    priority = (
        WEIGHT_EMERGENCY * row["score_emergency"]
        + WEIGHT_GRADUATION_ROUTE * row["score_graduation_route"]
        + WEIGHT_GRADUATION_PARKING * row["score_graduation_parking"]
        + WEIGHT_ROAD_CLASS * row["score_road_class"]
        + WEIGHT_STEEPNESS * row["score_steepness"]
    )
    return priority


def calculate_travel_time(length_feet):
    """
    calculate travel time in minutes for a road segment.

    args:
        length_feet: length of road in feet

    returns:
        travel time in minutes
    """
    if pd.isna(length_feet) or length_feet <= 0:
        return 0.0

    # convert feet to miles
    length_miles = length_feet / 5280.0

    # time = distance / speed (in hours), convert to minutes
    time_minutes = (length_miles / PLOW_SPEED_MPH) * 60.0

    return time_minutes


def main():
    print("[INFO] loading classified roads...")
    roads = pd.read_csv("classified_roads.csv")
    print(f"[INFO] loaded {len(roads)} road segments")

    # calculate priority for each road
    print("[INFO] calculating priority scores...")
    roads["priority"] = roads.apply(calculate_priority, axis=1)

    # calculate travel time
    print("[INFO] calculating travel times...")
    roads["travel_time_minutes"] = roads["length_feet"].apply(calculate_travel_time)

    # save results
    output_file = "roads_with_priority.csv"
    roads.to_csv(output_file, index=False)
    print(f"[INFO] saved roads with priority to {output_file}")

    # print summary statistics
    print("\n[SUMMARY] priority statistics:")
    print(f"mean priority: {roads['priority'].mean():.4f}")
    print(f"median priority: {roads['priority'].median():.4f}")
    print(f"min priority: {roads['priority'].min():.4f}")
    print(f"max priority: {roads['priority'].max():.4f}")
    print(f"std dev: {roads['priority'].std():.4f}")

    print("\n[SUMMARY] travel time statistics:")
    print(f"mean travel time: {roads['travel_time_minutes'].mean():.2f} minutes")
    print(f"median travel time: {roads['travel_time_minutes'].median():.2f} minutes")
    print(
        f"total travel time (all roads): {roads['travel_time_minutes'].sum():.2f} minutes ({roads['travel_time_minutes'].sum()/60:.2f} hours)"
    )

    # show top priority roads
    print("\n[INFO] top 10 priority roads:")
    top_roads = roads.nlargest(10, "priority")[
        [
            "NAME",
            "priority",
            "travel_time_minutes",
            "FCLASS2014",
            "score_emergency",
            "score_graduation_route",
            "score_graduation_parking",
        ]
    ]
    print(top_roads.to_string(index=False))

    # priority distribution by road class
    print("\n[INFO] priority by road class:")
    priority_by_class = roads.groupby("FCLASS2014")["priority"].agg(
        ["mean", "median", "count"]
    )
    print(priority_by_class)

    return roads


if __name__ == "__main__":
    main()
