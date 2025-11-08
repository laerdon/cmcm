#!/usr/bin/env python3
"""
classify roads based on emergency routes, campus proximity, graduation routes,
parking areas, road class, and steepness for snow plow prioritization.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point


def classify_emergency_routes(roads_gdf):
    """
    classify emergency routes using heuristics.
    returns series with 1.0 for emergency routes, 0 otherwise.
    """
    emergency_score = pd.Series(0.0, index=roads_gdf.index)

    # principal arterials are emergency routes
    emergency_score[roads_gdf["FCLASS2014"] == "Principal Arterial"] = 1.0

    # roads with "HOSPITAL" in name
    if "NAME" in roads_gdf.columns:
        hospital_mask = roads_gdf["NAME"].str.contains("HOSPITAL", case=False, na=False)
        emergency_score[hospital_mask] = 1.0

    return emergency_score


def classify_campus_routes(roads_gdf):
    """
    classify campus routes based on proximity to cornell campus.
    campus center approximate: 42.45°N, -76.48°W
    returns series with scores: 1.0 for on-campus, 0.6 for nearby, 0 otherwise.
    """
    # cornell campus center (approximate barton hall location)
    campus_center = Point(-76.48, 42.45)

    # reproject to geographic coordinates for distance calculation
    roads_geo = roads_gdf.to_crs("EPSG:4326")

    # calculate distance from each road to campus center (in degrees, approximate)
    # 1 degree latitude ≈ 111 km, so 1.5 km ≈ 0.0135 degrees
    campus_radius_on = 0.0135  # ~1.5 km
    campus_radius_nearby = 0.045  # ~5 km (for routes to/from campus)

    campus_score = pd.Series(0.0, index=roads_gdf.index)

    for idx, road in roads_geo.iterrows():
        distance = road.geometry.distance(campus_center)

        if distance < campus_radius_on:
            campus_score[idx] = 1.0  # on campus
        elif distance < campus_radius_nearby:
            campus_score[idx] = 0.6  # nearby/access routes

    # also check for key road names that connect to campus
    campus_road_names = [
        "STEWART",
        "SENECA",
        "STATE",
        "TRIPHAMMER",
        "UNIVERSITY",
        "CAMPUS",
        "THURSTON",
        "CORNELL",
        "COLLEGE",
        "TOWER",
    ]

    if "NAME" in roads_gdf.columns:
        for road_name in campus_road_names:
            name_mask = roads_gdf["NAME"].str.contains(road_name, case=False, na=False)
            # upgrade score if not already high
            campus_score[name_mask] = campus_score[name_mask].apply(
                lambda x: max(x, 0.6)
            )

    return campus_score


def classify_graduation_parking(roads_gdf):
    """
    classify parking areas near graduation ceremony location.
    barton hall approximate: 42.45°N, -76.48°W
    returns series with 1.0 for parking within 2km, 0 otherwise.
    """
    # graduation location (barton hall)
    graduation_location = Point(-76.48, 42.45)

    # reproject to geographic coordinates
    roads_geo = roads_gdf.to_crs("EPSG:4326")

    # 2 km radius ≈ 0.018 degrees
    parking_radius = 0.018

    parking_score = pd.Series(0.0, index=roads_gdf.index)

    for idx, road in roads_geo.iterrows():
        distance = road.geometry.distance(graduation_location)

        if distance < parking_radius:
            parking_score[idx] = 1.0

    return parking_score


def classify_road_class(roads_gdf):
    """
    map road class to scores.
    returns series with road class scores.
    """
    road_class_mapping = {
        "Local Road": 0.25,
        "Collector": 0.5,
        "Minor Arterial": 0.7,
        "Principal Arterial": 0.85,
        "NA": 0.25,  # treat NA as local
    }

    road_class_score = roads_gdf["FCLASS2014"].map(road_class_mapping).fillna(0.25)

    return road_class_score


def classify_steepness(steepness_pct):
    """
    map steepness percentage to scores.
    returns series with steepness scores.
    """

    def get_steepness_score(pct):
        if pd.isna(pct):
            return 0.2
        elif pct < 5:
            return 0.2
        elif pct < 10:
            return 0.4
        elif pct < 15:
            return 0.6
        elif pct < 20:
            return 0.8
        else:
            return 1.0

    return steepness_pct.apply(get_steepness_score)


def main():
    print("[INFO] loading road network...")
    roads = gpd.read_file("Ithaca NY Roads/Roads.shp")
    print(f"[INFO] loaded {len(roads)} road segments")

    print("[INFO] loading steepness data...")
    steepness = pd.read_csv("road_steepness_results.csv")
    print(f"[INFO] loaded {len(steepness)} steepness records")

    # merge on objectid
    print("[INFO] merging datasets...")
    roads_merged = roads.merge(
        steepness[
            ["objectid", "max_grade_pct", "avg_grade_pct", "elevation_change_ft"]
        ],
        left_on="OBJECTID",
        right_on="objectid",
        how="left",
    )

    print(f"[INFO] merged dataset has {len(roads_merged)} rows")

    # classify roads
    print("[INFO] classifying emergency routes...")
    roads_merged["score_emergency"] = classify_emergency_routes(roads_merged)

    print("[INFO] classifying campus routes...")
    roads_merged["score_graduation_route"] = classify_campus_routes(roads_merged)

    print("[INFO] classifying parking areas...")
    roads_merged["score_graduation_parking"] = classify_graduation_parking(roads_merged)

    print("[INFO] classifying road classes...")
    roads_merged["score_road_class"] = classify_road_class(roads_merged)

    print("[INFO] classifying steepness...")
    roads_merged["score_steepness"] = classify_steepness(roads_merged["max_grade_pct"])

    # calculate road length in feet
    # reproject to feet-based crs for accurate length
    roads_projected = roads_merged.to_crs("EPSG:2261")  # ny state plane, feet
    roads_merged["length_feet"] = roads_projected.geometry.length

    # save results
    output_file = "classified_roads.csv"

    # prepare output columns
    output_cols = [
        "OBJECTID",
        "NAME",
        "FCLASS2014",
        "TOTALANES",
        "length_feet",
        "max_grade_pct",
        "avg_grade_pct",
        "elevation_change_ft",
        "score_emergency",
        "score_graduation_route",
        "score_graduation_parking",
        "score_road_class",
        "score_steepness",
    ]

    # save without geometry for csv
    roads_output = roads_merged[output_cols].copy()
    roads_output.to_csv(output_file, index=False)
    print(f"[INFO] saved classified roads to {output_file}")

    # also save full geodataframe with geometry
    roads_merged.to_file("classified_roads.geojson", driver="GeoJSON")
    print(f"[INFO] saved classified roads with geometry to classified_roads.geojson")

    # print summary statistics
    print("\n[SUMMARY] classification statistics:")
    print(
        f"emergency routes (score=1.0): {(roads_merged['score_emergency'] == 1.0).sum()}"
    )
    print(
        f"campus roads (score=1.0): {(roads_merged['score_graduation_route'] == 1.0).sum()}"
    )
    print(
        f"nearby campus roads (score=0.6): {(roads_merged['score_graduation_route'] == 0.6).sum()}"
    )
    print(
        f"parking areas (score=1.0): {(roads_merged['score_graduation_parking'] == 1.0).sum()}"
    )
    print(f"\nroad class distribution:")
    print(roads_merged.groupby("FCLASS2014")["score_road_class"].first())
    print(f"\nsteepness score distribution:")
    print(roads_merged["score_steepness"].value_counts().sort_index())

    return roads_merged


if __name__ == "__main__":
    main()
