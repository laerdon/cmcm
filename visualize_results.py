#!/usr/bin/env python3
"""
visualize optimization results: partitions, cycles, priority heatmaps.
creates interactive html maps using folium (if available) or static plots.
"""

import geopandas as gpd
import pandas as pd
import networkx as nx
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

# try to import folium, but don't fail if not available
try:
    import folium
    from folium import plugins

    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("[WARNING] folium not available, will create static plots only")

try:
    import pyproj

    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    print("[WARNING] pyproj not available, coordinate transformations may be limited")


def create_priority_heatmap(roads_gdf, output_file="priority_heatmap.html"):
    """
    create heatmap showing road priorities.

    args:
        roads_gdf: geodataframe with road geometries and priorities
        output_file: output html file
    """
    if not FOLIUM_AVAILABLE:
        print(f"[INFO] creating static priority heatmap instead...")
        create_static_priority_map(roads_gdf)
        return None

    print(f"[INFO] creating priority heatmap...")

    # convert to geographic coordinates for folium
    roads_geo = roads_gdf.to_crs("EPSG:4326")

    # calculate center
    bounds = roads_geo.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    # create map
    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap"
    )

    # color map for priority
    colormap = cm.get_cmap("YlOrRd")

    # normalize priorities
    priorities = roads_geo["priority"].values
    priority_min = priorities.min()
    priority_max = priorities.max()

    # add roads colored by priority
    for idx, road in roads_geo.iterrows():
        # normalize priority to 0-1
        normalized_priority = (road["priority"] - priority_min) / (
            priority_max - priority_min
        )
        color = colormap(normalized_priority)
        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
        )

        # create tooltip
        tooltip = (
            f"road: {road['NAME']}<br>"
            f"priority: {road['priority']:.3f}<br>"
            f"class: {road['FCLASS2014']}<br>"
            f"steepness: {road.get('max_grade_pct', 0):.1f}%"
        )

        # add line
        coords = [(pt[1], pt[0]) for pt in road.geometry.coords]
        folium.PolyLine(
            coords, color=color_hex, weight=3, opacity=0.7, tooltip=tooltip
        ).add_to(m)

    # add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p style="margin: 0"><b>Priority Legend</b></p>
    <p style="margin: 5px 0"><span style="background-color: #ffffcc; padding: 2px 10px">&#9632;</span> Low</p>
    <p style="margin: 5px 0"><span style="background-color: #feb24c; padding: 2px 10px">&#9632;</span> Medium</p>
    <p style="margin: 5px 0"><span style="background-color: #e31a1c; padding: 2px 10px">&#9632;</span> High</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # save map
    m.save(output_file)
    print(f"[INFO] saved priority heatmap to {output_file}")

    return m


def create_static_priority_map(roads_gdf, output_file="plots/priority_map.png"):
    """
    create static plot of road priorities.

    args:
        roads_gdf: geodataframe with roads
        output_file: output file
    """
    Path(output_file).parent.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    roads_gdf.plot(column="priority", cmap="YlOrRd", linewidth=1.5, ax=ax, legend=True)
    ax.set_title("road priority heatmap", fontsize=16)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] saved static priority map to {output_file}")


def create_partition_map(
    G, partitions, partition_set_id=0, output_file="partition_map.html"
):
    """
    create map showing graph partitions with different colors.

    args:
        G: networkx graph
        partitions: list of partition sets
        partition_set_id: which partition set to visualize
        output_file: output html file
    """
    if not FOLIUM_AVAILABLE:
        print(f"[INFO] creating static partition map instead...")
        create_static_partition_map(G, partitions, partition_set_id)
        return None

    print(f"[INFO] creating partition map for set {partition_set_id}...")

    # get partition
    partition_set = partitions[partition_set_id]

    # create node to partition mapping
    node_to_partition = {}
    for p_id, partition_nodes in enumerate(partition_set):
        for node in partition_nodes:
            node_to_partition[node] = p_id

    # get node positions (convert to lat/lon)
    node_positions = nx.get_node_attributes(G, "pos")

    # calculate center
    all_x = [pos[0] for pos in node_positions.values()]
    all_y = [pos[1] for pos in node_positions.values()]
    center_x = np.mean(all_x)
    center_y = np.mean(all_y)

    # approximate conversion (ny state plane to lat/lon)
    # this is rough - ideally use proper projection
    center_lon = -76.5
    center_lat = 42.45

    # create map
    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap"
    )

    # colors for partitions
    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "lightblue",
        "darkgreen",
    ]

    # add edges colored by partition
    for u, v, data in G.edges(data=True):
        if u in node_to_partition and v in node_to_partition:
            partition_id = node_to_partition[u]
            color = colors[partition_id % len(colors)]

            # get geometry if available
            if "geometry" in data:
                geom = data["geometry"]
                # convert to geographic
                from shapely.ops import transform
                import pyproj

                # ny state plane to wgs84
                project = pyproj.Transformer.from_crs(
                    "EPSG:2261", "EPSG:4326", always_xy=True
                ).transform
                geom_geo = transform(project, geom)

                coords = [(pt[1], pt[0]) for pt in geom_geo.coords]

                tooltip = (
                    f"partition: {partition_id}<br>"
                    f"priority: {data['priority']:.3f}<br>"
                    f"road: {data['name']}"
                )

                folium.PolyLine(
                    coords, color=color, weight=2, opacity=0.6, tooltip=tooltip
                ).add_to(m)

    # add legend
    legend_html = '<div style="position: fixed; bottom: 50px; right: 50px; width: 180px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px">'
    legend_html += '<p style="margin: 0"><b>Partition Legend</b></p>'
    for i in range(8):
        color = colors[i % len(colors)]
        legend_html += f'<p style="margin: 5px 0"><span style="color: {color}; padding: 2px 10px">&#9632;</span> Partition {i}</p>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    # save map
    m.save(output_file)
    print(f"[INFO] saved partition map to {output_file}")

    return m


def create_static_partition_map(
    G, partitions, partition_set_id=0, output_file="plots/partition_map.png"
):
    """
    create static plot of partitions.

    args:
        G: networkx graph
        partitions: list of partition sets
        partition_set_id: which partition set to visualize
        output_file: output file
    """
    Path(output_file).parent.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    # get partition assignment
    partition_set = partitions[partition_set_id]
    node_to_partition = {}
    for p_id, partition_nodes in enumerate(partition_set):
        for node in partition_nodes:
            node_to_partition[node] = p_id

    # colors
    colors_list = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "cyan",
        "darkgreen",
    ]

    # draw edges by partition
    for u, v, data in G.edges(data=True):
        if u in node_to_partition and "geometry" in data:
            p_id = node_to_partition[u]
            color = colors_list[p_id % len(colors_list)]

            geom = data["geometry"]

            # handle both linestring and multilinestring
            if hasattr(geom, "geoms"):
                # multilinestring
                for line in geom.geoms:
                    x, y = line.xy
                    ax.plot(x, y, color=color, linewidth=1, alpha=0.6)
            else:
                # simple linestring
                x, y = geom.xy
                ax.plot(x, y, color=color, linewidth=1, alpha=0.6)

    ax.set_title(f"graph partitions (set {partition_set_id})", fontsize=16)
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")

    # legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors_list[i % len(colors_list)], label=f"partition {i}")
        for i in range(8)
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] saved static partition map to {output_file}")


def create_cycle_visualization(G, operational_plans, output_file="cycle_routes.html"):
    """
    create map showing optimal cycle routes for each driver.

    args:
        G: networkx graph
        operational_plans: operational plan data
        output_file: output html file
    """
    if not FOLIUM_AVAILABLE:
        print(f"[INFO] skipping cycle visualization (requires folium)")
        return None

    print(f"[INFO] creating cycle route visualization...")

    # use first partition set
    plan = operational_plans[0]
    assignments = plan["assignments"]

    # create map center
    center_lat = 42.45
    center_lon = -76.5

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap"
    )

    # colors for drivers
    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "lightblue",
        "darkgreen",
    ]

    # add each driver's cycle
    for assignment in assignments:
        driver_id = assignment["driver_id"]
        cycle = assignment["cycle"]
        color = colors[driver_id % len(colors)]

        # draw cycle
        for u, v, k in cycle:
            if G.has_edge(u, v, key=k):
                edge_data = G[u][v][k]

                if "geometry" in edge_data:
                    geom = edge_data["geometry"]

                    # convert to geographic
                    from shapely.ops import transform
                    import pyproj

                    project = pyproj.Transformer.from_crs(
                        "EPSG:2261", "EPSG:4326", always_xy=True
                    ).transform
                    geom_geo = transform(project, geom)

                    coords = [(pt[1], pt[0]) for pt in geom_geo.coords]

                    tooltip = (
                        f"driver: {driver_id}<br>"
                        f"road: {edge_data['name']}<br>"
                        f"priority: {edge_data['priority']:.3f}"
                    )

                    folium.PolyLine(
                        coords, color=color, weight=3, opacity=0.7, tooltip=tooltip
                    ).add_to(m)

    # add legend
    legend_html = '<div style="position: fixed; bottom: 50px; right: 50px; width: 180px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px">'
    legend_html += '<p style="margin: 0"><b>Driver Routes</b></p>'
    for assignment in assignments:
        driver_id = assignment["driver_id"]
        color = colors[driver_id % len(colors)]
        time = assignment["cycle_time"]
        legend_html += f'<p style="margin: 5px 0"><span style="color: {color}; padding: 2px 10px">&#9632;</span> Driver {driver_id} ({time:.1f}min)</p>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    # save map
    m.save(output_file)
    print(f"[INFO] saved cycle routes to {output_file}")

    return m


def create_summary_plots(output_dir="plots"):
    """
    create summary plots and statistics.

    args:
        output_dir: directory for output plots
    """
    print(f"[INFO] creating summary plots...")

    Path(output_dir).mkdir(exist_ok=True)

    # load data
    cycle_evals = pd.read_csv("cycle_evaluations.csv")
    partition_stats = pd.read_csv("partition_statistics.csv")

    # plot 1: benefit distribution by method
    plt.figure(figsize=(10, 6))
    cycle_evals.boxplot(column="benefit_per_minute", by="method")
    plt.title("benefit per minute by cycle generation method")
    plt.xlabel("method")
    plt.ylabel("benefit per minute")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benefit_by_method.png", dpi=150)
    plt.close()

    # plot 2: partition balance
    plt.figure(figsize=(12, 6))
    for ps_id in partition_stats["partition_set_id"].unique():
        ps_data = partition_stats[partition_stats["partition_set_id"] == ps_id]
        plt.plot(
            ps_data["partition_id"],
            ps_data["total_priority"],
            marker="o",
            label=f"set {ps_id}",
        )
    plt.xlabel("partition id")
    plt.ylabel("total priority")
    plt.title("priority balance across partitions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/partition_balance.png", dpi=150)
    plt.close()

    # plot 3: cycle time distribution
    plt.figure(figsize=(10, 6))
    plt.hist(cycle_evals["cycle_time"], bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("cycle time (minutes)")
    plt.ylabel("frequency")
    plt.title("distribution of cycle times")
    plt.axvline(
        cycle_evals["cycle_time"].mean(),
        color="red",
        linestyle="--",
        label=f'mean: {cycle_evals["cycle_time"].mean():.1f} min',
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cycle_time_distribution.png", dpi=150)
    plt.close()

    print(f"[INFO] saved plots to {output_dir}/")


def main():
    print("[INFO] loading data for visualization...")

    # load graph
    with open("road_graph.gpickle", "rb") as f:
        G = pickle.load(f)
    print(f"[INFO] loaded graph with {G.number_of_nodes()} nodes")

    # load roads with priorities
    roads_gdf = gpd.read_file("classified_roads.geojson")
    priority_df = pd.read_csv("roads_with_priority.csv")
    roads_gdf = roads_gdf.merge(
        priority_df[["OBJECTID", "priority"]], on="OBJECTID", how="left"
    )
    print(f"[INFO] loaded {len(roads_gdf)} roads")

    # load partitions
    with open("partitions.json", "r") as f:
        partitions = json.load(f)
    print(f"[INFO] loaded {len(partitions)} partition sets")

    # load operational plans
    with open("operational_plans.json", "r") as f:
        operational_plans = json.load(f)
    print(f"[INFO] loaded {len(operational_plans)} operational plans")

    # create visualizations
    print("\n[INFO] generating visualizations...")

    # create static visualizations (folium not available)
    create_static_priority_map(roads_gdf)
    create_static_partition_map(G, partitions, partition_set_id=0)
    create_summary_plots("plots")

    print("\n[INFO] all visualizations created:")
    print("  - plots/priority_map.png: static map of road priorities")
    print("  - plots/partition_map.png: graph partitions visualization")
    print("  - plots/: summary plots and charts")

    if not FOLIUM_AVAILABLE:
        print("\n[NOTE] install folium and pyproj for interactive html maps:")
        print("  pip install folium pyproj")

    return 0


if __name__ == "__main__":
    main()
