#!/usr/bin/env python3
"""
construct directed graph from road network.
nodes are intersections, edges are road segments.
"""

import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
from shapely.geometry import Point
import pickle


def extract_endpoints(geometry):
    """
    extract start and end points from a linestring or multilinestring.

    args:
        geometry: shapely linestring or multilinestring

    returns:
        tuple of (start_point, end_point) as (x, y) tuples
    """
    # handle multilinestring by using first and last geometries
    if hasattr(geometry, "geoms"):
        # multilinestring
        first_line = geometry.geoms[0]
        last_line = geometry.geoms[-1]
        start = list(first_line.coords)[0]
        end = list(last_line.coords)[-1]
    else:
        # simple linestring
        coords = list(geometry.coords)
        start = coords[0]
        end = coords[-1]
    return start, end


def build_road_graph(roads_gdf, priority_df):
    """
    build directed graph from road network.

    args:
        roads_gdf: geodataframe with road geometries
        priority_df: dataframe with priority scores and attributes

    returns:
        networkx directed multigraph
    """
    # create directed multigraph (allows multiple edges between same nodes)
    G = nx.MultiDiGraph()

    # merge priority data with geometries
    roads_merged = roads_gdf.merge(
        priority_df,
        left_on="OBJECTID",
        right_on="OBJECTID",
        how="inner",
        suffixes=("", "_dup"),
    )

    # drop duplicate columns
    roads_merged = roads_merged.loc[:, ~roads_merged.columns.str.endswith("_dup")]

    print(f"[INFO] processing {len(roads_merged)} road segments...")

    # tolerance for snapping nearby endpoints (in coordinate units)
    # since we're in state plane feet, use 10 feet
    snap_tolerance = 10.0

    # collect all endpoints first to create consistent node ids
    endpoint_to_node = {}
    node_counter = 0

    for idx, road in roads_merged.iterrows():
        start, end = extract_endpoints(road.geometry)

        # snap to existing nearby nodes or create new ones
        for point in [start, end]:
            found_existing = False
            for existing_point, node_id in endpoint_to_node.items():
                # check if within snap tolerance
                dist = np.sqrt(
                    (point[0] - existing_point[0]) ** 2
                    + (point[1] - existing_point[1]) ** 2
                )
                if dist < snap_tolerance:
                    found_existing = True
                    break

            if not found_existing:
                endpoint_to_node[point] = node_counter
                G.add_node(node_counter, pos=point, x=point[0], y=point[1])
                node_counter += 1

    print(f"[INFO] created {G.number_of_nodes()} nodes (intersections)")

    # add edges
    edge_counter = 0
    for idx, road in roads_merged.iterrows():
        start, end = extract_endpoints(road.geometry)

        # find node ids for start and end
        start_node = None
        end_node = None

        for point, node_id in endpoint_to_node.items():
            dist_start = np.sqrt(
                (start[0] - point[0]) ** 2 + (start[1] - point[1]) ** 2
            )
            dist_end = np.sqrt((end[0] - point[0]) ** 2 + (end[1] - point[1]) ** 2)

            if dist_start < snap_tolerance:
                start_node = node_id
            if dist_end < snap_tolerance:
                end_node = node_id

        if start_node is None or end_node is None:
            print(f"[WARNING] could not find nodes for road {idx}")
            continue

        # add directed edge (both directions for undirected roads)
        edge_attrs = {
            "objectid": int(road["OBJECTID"]),
            "name": str(road["NAME"]) if pd.notna(road["NAME"]) else f"unnamed_{idx}",
            "road_class": str(road["FCLASS2014"]),
            "priority": float(road["priority"]),
            "travel_time": float(road["travel_time_minutes"]),
            "length_feet": float(road["length_feet"]),
            "max_grade_pct": (
                float(road["max_grade_pct"]) if pd.notna(road["max_grade_pct"]) else 0.0
            ),
            "score_emergency": float(road["score_emergency"]),
            "score_graduation_route": float(road["score_graduation_route"]),
            "score_graduation_parking": float(road["score_graduation_parking"]),
            "geometry": road.geometry,
        }

        # add edge in forward direction
        G.add_edge(start_node, end_node, key=edge_counter, **edge_attrs)
        edge_counter += 1

        # add edge in reverse direction (roads are bidirectional)
        G.add_edge(end_node, start_node, key=edge_counter, **edge_attrs)
        edge_counter += 1

    print(f"[INFO] created {G.number_of_edges()} directed edges")

    return G


def analyze_graph_connectivity(G):
    """
    analyze connectivity of the graph.

    args:
        G: networkx graph
    """
    print("\n[ANALYSIS] graph connectivity:")

    # convert to undirected for connectivity analysis
    G_undirected = G.to_undirected()

    # find connected components
    components = list(nx.connected_components(G_undirected))
    print(f"number of connected components: {len(components)}")

    # analyze each component
    for i, component in enumerate(components):
        print(f"component {i+1}: {len(component)} nodes")

    # find largest component
    largest_component = max(components, key=len)
    print(
        f"\nlargest component has {len(largest_component)} nodes ({len(largest_component)/G.number_of_nodes()*100:.1f}% of total)"
    )

    # degree distribution
    degrees = [G.degree(n) for n in G.nodes()]
    print(f"\ndegree statistics:")
    print(f"  mean degree: {np.mean(degrees):.2f}")
    print(f"  median degree: {np.median(degrees):.2f}")
    print(f"  max degree: {np.max(degrees)}")

    return largest_component


def main():
    print("[INFO] loading road geometries...")
    roads_gdf = gpd.read_file("classified_roads.geojson")
    print(f"[INFO] loaded {len(roads_gdf)} road segments with geometries")

    print("[INFO] loading priority data...")
    priority_df = pd.read_csv("roads_with_priority.csv")
    print(f"[INFO] loaded {len(priority_df)} priority records")

    # build graph
    print("\n[INFO] building road graph...")
    G = build_road_graph(roads_gdf, priority_df)

    # analyze connectivity
    largest_component = analyze_graph_connectivity(G)

    # extract largest connected component as main graph
    G_main = G.subgraph(largest_component).copy()
    print(
        f"\n[INFO] extracted main graph with {G_main.number_of_nodes()} nodes and {G_main.number_of_edges()} edges"
    )

    # save graph
    output_file = "road_graph.gpickle"
    with open(output_file, "wb") as f:
        pickle.dump(G_main, f)
    print(f"[INFO] saved graph to {output_file}")

    # save graph statistics
    stats = {
        "num_nodes": G_main.number_of_nodes(),
        "num_edges": G_main.number_of_edges(),
        "num_components": 1,
        "total_priority": sum(
            data["priority"] for u, v, data in G_main.edges(data=True)
        ),
        "total_length_feet": sum(
            data["length_feet"] for u, v, data in G_main.edges(data=True)
        ),
        "total_travel_time_minutes": sum(
            data["travel_time"] for u, v, data in G_main.edges(data=True)
        ),
    }

    stats_df = pd.DataFrame([stats])
    stats_df.to_csv("graph_statistics.csv", index=False)
    print(f"[INFO] saved statistics to graph_statistics.csv")

    print("\n[SUMMARY] graph statistics:")
    for key, value in stats.items():
        if "time" in key:
            print(f"{key}: {value:.2f} minutes ({value/60:.2f} hours)")
        elif "length" in key:
            print(f"{key}: {value:.2f} feet ({value/5280:.2f} miles)")
        else:
            print(f"{key}: {value}")

    return G_main


if __name__ == "__main__":
    main()
