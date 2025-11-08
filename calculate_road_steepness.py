#!/usr/bin/env python3
"""
calculate road steepness from contour lines and road network data.
estimates grade (steepness) as percentage for each road segment.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import pandas as pd
from scipy.interpolate import LinearNDInterpolator


def get_elevation_at_point(point, contours, interpolator=None):
    """
    get elevation at a point using nearest contour or interpolation.

    args:
        point: shapely Point object
        contours: geodataframe of contour lines
        interpolator: optional scipy interpolator for elevation

    returns:
        elevation in feet
    """
    if interpolator is not None:
        try:
            elev = float(interpolator(point.x, point.y))
            if not np.isnan(elev):
                return elev
        except:
            pass

    # fallback: find nearest contour
    distances = contours.geometry.distance(point)
    nearest_idx = distances.idxmin()
    return contours.loc[nearest_idx, "Contour"]


def sample_elevation_along_road(road_geom, contours, num_samples=10, interpolator=None):
    """
    sample elevation at multiple points along a road segment.

    args:
        road_geom: linestring geometry of road
        contours: geodataframe of contour lines
        num_samples: number of points to sample along road
        interpolator: optional scipy interpolator

    returns:
        list of (distance, elevation) tuples
    """
    elevations = []
    total_length = road_geom.length

    for i in range(num_samples):
        # sample at evenly spaced distances
        distance = (i / (num_samples - 1)) * total_length if num_samples > 1 else 0
        point = road_geom.interpolate(distance)
        elevation = get_elevation_at_point(point, contours, interpolator)
        elevations.append((distance, elevation))

    return elevations


def calculate_steepness(elevation_samples):
    """
    calculate maximum and average steepness from elevation samples.

    args:
        elevation_samples: list of (distance, elevation) tuples

    returns:
        dict with max_grade, avg_grade, elevation_change
    """
    if len(elevation_samples) < 2:
        return {"max_grade": 0, "avg_grade": 0, "elevation_change": 0}

    grades = []
    for i in range(1, len(elevation_samples)):
        dist_diff = elevation_samples[i][0] - elevation_samples[i - 1][0]
        elev_diff = elevation_samples[i][1] - elevation_samples[i - 1][1]

        if dist_diff > 0:
            # grade as percentage: (rise/run) * 100
            grade = abs(elev_diff / dist_diff) * 100
            grades.append(grade)

    total_elev_change = abs(elevation_samples[-1][1] - elevation_samples[0][1])
    total_dist = elevation_samples[-1][0] - elevation_samples[0][0]

    avg_grade = (total_elev_change / total_dist * 100) if total_dist > 0 else 0
    max_grade = max(grades) if grades else 0

    return {
        "max_grade": max_grade,
        "avg_grade": avg_grade,
        "elevation_change": total_elev_change,
        "min_elevation": min(e[1] for e in elevation_samples),
        "max_elevation": max(e[1] for e in elevation_samples),
    }


def create_elevation_interpolator(contours):
    """
    create a 2d interpolator for elevation data from contour lines.
    samples points from contour lines to build interpolation grid.

    args:
        contours: geodataframe of contour lines

    returns:
        scipy LinearNDInterpolator
    """
    print("[INFO] building elevation interpolator from contours...")

    points = []
    elevations = []

    # sample points from contours
    for idx, row in contours.iterrows():
        geom = row.geometry
        elevation = row["Contour"]

        # sample multiple points along each contour
        if hasattr(geom, "geoms"):  # multilinestring
            for line in geom.geoms:
                num_pts = max(5, int(line.length / 100))  # sample every ~100 units
                for i in range(num_pts):
                    pt = line.interpolate(i / num_pts, normalized=True)
                    points.append([pt.x, pt.y])
                    elevations.append(elevation)
        else:  # linestring
            num_pts = max(5, int(geom.length / 100))
            for i in range(num_pts):
                pt = geom.interpolate(i / num_pts, normalized=True)
                points.append([pt.x, pt.y])
                elevations.append(elevation)

    points = np.array(points)
    elevations = np.array(elevations)

    print(f"[INFO] created interpolator with {len(points)} sample points")

    return LinearNDInterpolator(points, elevations)


def main():
    print("[INFO] loading shapefiles...")

    # load contours
    contours = gpd.read_file("cugir-008148/Ithaca2ftContours.shp")
    print(f"[INFO] loaded {len(contours)} contour lines")
    print(
        f'[INFO] elevation range: {contours["Contour"].min():.1f} - {contours["Contour"].max():.1f} feet'
    )

    # load roads
    roads = gpd.read_file("Ithaca NY Roads/Roads.shp")
    print(f"[INFO] loaded {len(roads)} road segments")

    # reproject roads to match contours crs
    print(f"[INFO] reprojecting roads from {roads.crs} to {contours.crs}...")
    roads = roads.to_crs(contours.crs)

    # create elevation interpolator for better accuracy
    interpolator = create_elevation_interpolator(contours)

    # calculate steepness for each road
    print("[INFO] calculating steepness for each road segment...")
    results = []

    for idx, road in roads.iterrows():
        if idx % 100 == 0:
            print(f"[PROGRESS] processing road {idx+1}/{len(roads)}...")

        road_name = road["NAME"] if pd.notna(road["NAME"]) else f"unnamed_{idx}"

        # sample elevation along road
        elevation_samples = sample_elevation_along_road(
            road.geometry,
            contours,
            num_samples=20,  # sample 20 points along each road
            interpolator=interpolator,
        )

        # calculate steepness metrics
        steepness = calculate_steepness(elevation_samples)

        results.append(
            {
                "road_name": road_name,
                "objectid": road["OBJECTID"],
                "length_feet": road.geometry.length,
                "max_grade_pct": round(steepness["max_grade"], 2),
                "avg_grade_pct": round(steepness["avg_grade"], 2),
                "elevation_change_ft": round(steepness["elevation_change"], 1),
                "min_elevation_ft": round(steepness["min_elevation"], 1),
                "max_elevation_ft": round(steepness["max_elevation"], 1),
            }
        )

    # create results dataframe
    results_df = pd.DataFrame(results)

    # sort by steepest roads
    results_df = results_df.sort_values("max_grade_pct", ascending=False)

    # save results
    output_file = "road_steepness_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n[INFO] results saved to {output_file}")

    # print summary statistics
    print("\n[SUMMARY] steepness statistics:")
    print(
        f'steepest road (max grade): {results_df.iloc[0]["road_name"]} - {results_df.iloc[0]["max_grade_pct"]:.2f}%'
    )
    print(
        f'average max grade across all roads: {results_df["max_grade_pct"].mean():.2f}%'
    )
    print(f'median max grade: {results_df["max_grade_pct"].median():.2f}%')

    print(f"\n[INFO] top 10 steepest roads:")
    print(
        results_df[
            ["road_name", "max_grade_pct", "avg_grade_pct", "elevation_change_ft"]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(f"\n[INFO] roads with grade > 10%:")
    steep_roads = results_df[results_df["max_grade_pct"] > 10]
    print(f"found {len(steep_roads)} roads with max grade > 10%")

    return results_df


if __name__ == "__main__":
    main()
