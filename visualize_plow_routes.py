#!/usr/bin/env python3
"""
visualize optimized snow plow routes for each driver.
creates individual maps per driver and combined overview map.
"""

import json
import pickle
import folium
from folium import plugins
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import pyproj
from shapely.ops import transform


def load_data():
    """
    load necessary data files.
    
    returns:
        tuple of (graph, optimized_cycles, operational_plans)
    """
    print("[INFO] loading data files...")
    
    # load graph
    with open("road_graph.gpickle", "rb") as f:
        G = pickle.load(f)
    
    # load optimized cycles
    with open("optimized_cycles.json", "r") as f:
        optimized_cycles = json.load(f)
    
    # load operational plans
    with open("operational_plans.json", "r") as f:
        operational_plans = json.load(f)
    
    print(f"[INFO] loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G, optimized_cycles, operational_plans


def get_partition_set_cycles(optimized_cycles, partition_set_id=2):
    """
    extract cycles for specific partition set.
    
    args:
        optimized_cycles: dict of all optimized cycles
        partition_set_id: which partition set to use
    
    returns:
        dict mapping driver_id to cycle data
    """
    driver_cycles = {}
    
    for key, cycles in optimized_cycles.items():
        # parse key like "ps0_p0" 
        parts = key.split("_")
        ps_id = int(parts[0][2:])  # extract number from "ps2"
        p_id = int(parts[1][1:])   # extract number from "p0"
        
        if ps_id == partition_set_id:
            # get best cycle (first one after optimization)
            driver_cycles[p_id] = cycles[0]
    
    return driver_cycles


def convert_geometry_to_latlon(geometry):
    """
    convert geometry from web mercator to wgs84.
    
    args:
        geometry: shapely geometry in epsg:3857
    
    returns:
        geometry in epsg:4326
    """
    project = pyproj.Transformer.from_crs(
        "EPSG:3857", "EPSG:4326", always_xy=True
    ).transform
    return transform(project, geometry)


def create_driver_map(G, driver_id, cycle_data, output_file):
    """
    create individual map for one driver's route.
    
    args:
        G: networkx graph
        driver_id: driver number
        cycle_data: cycle information
        output_file: where to save html
    """
    print(f"[INFO] creating map for driver {driver_id}...")
    
    cycle = cycle_data["cycle"]
    
    if not cycle:
        print(f"[WARNING] driver {driver_id} has empty cycle")
        return
    
    # collect all coordinates for bounds
    all_coords = []
    
    # extract route segments
    route_segments = []
    cumulative_time = 0.0
    
    for u, v, k in cycle:
        if G.has_edge(u, v, key=k):
            edge_data = G[u][v][k]
            
            if "geometry" in edge_data:
                geom = edge_data["geometry"]
                geom_latlon = convert_geometry_to_latlon(geom)
                
                # handle multilinestring
                if hasattr(geom_latlon, "geoms"):
                    for line in geom_latlon.geoms:
                        coords = [(pt[1], pt[0]) for pt in line.coords]
                        all_coords.extend(coords)
                else:
                    coords = [(pt[1], pt[0]) for pt in geom_latlon.coords]
                    all_coords.extend(coords)
                
                route_segments.append({
                    "coords": coords,
                    "name": edge_data["name"],
                    "priority": edge_data["priority"],
                    "travel_time": edge_data["travel_time"],
                    "cumulative_time": cumulative_time,
                })
                
                cumulative_time += edge_data["travel_time"]
    
    if not all_coords:
        print(f"[WARNING] no valid coordinates for driver {driver_id}")
        return
    
    # calculate map center
    center_lat = np.mean([c[0] for c in all_coords])
    center_lon = np.mean([c[1] for c in all_coords])
    
    # create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap"
    )
    
    # add route segments
    driver_color = ["red", "blue", "green", "purple", "orange", "darkred", "lightblue", "darkgreen"][driver_id % 8]
    
    for i, segment in enumerate(route_segments):
        tooltip_text = (
            f"<b>Road:</b> {segment['name']}<br>"
            f"<b>Priority:</b> {segment['priority']:.3f}<br>"
            f"<b>Segment time:</b> {segment['travel_time']:.2f} min<br>"
            f"<b>Cumulative time:</b> {segment['cumulative_time']:.2f} min<br>"
            f"<b>Segment:</b> {i+1}/{len(route_segments)}"
        )
        
        folium.PolyLine(
            segment["coords"],
            color=driver_color,
            weight=5,
            opacity=0.8,
            tooltip=tooltip_text
        ).add_to(m)
    
    # add start marker
    if all_coords:
        folium.Marker(
            all_coords[0],
            popup=f"<b>Start</b><br>Driver {driver_id}",
            icon=folium.Icon(color="green", icon="play", prefix="fa")
        ).add_to(m)
        
        # add end marker
        folium.Marker(
            all_coords[-1],
            popup=f"<b>End</b><br>Driver {driver_id}",
            icon=folium.Icon(color="red", icon="stop", prefix="fa")
        ).add_to(m)
    
    # add info box
    info_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 250px; 
                background-color: white; border:2px solid grey; 
                z-index:9999; font-size:14px; padding: 10px;
                border-radius: 5px;">
    <h4 style="margin: 0 0 10px 0;">Driver {driver_id} Route</h4>
    <p style="margin: 5px 0;"><b>Partition:</b> {cycle_data.get('partition_id', driver_id)}</p>
    <p style="margin: 5px 0;"><b>Method:</b> {cycle_data.get('method', 'N/A')}</p>
    <p style="margin: 5px 0;"><b>Total time:</b> {cycle_data['evaluation']['cycle_time']:.2f} min</p>
    <p style="margin: 5px 0;"><b>Benefit/min:</b> {cycle_data['evaluation']['benefit_per_minute']:.2f}</p>
    <p style="margin: 5px 0;"><b>Segments:</b> {len(cycle)}</p>
    <p style="margin: 5px 0;"><b>Unique edges:</b> {cycle_data['metrics']['unique_edges']}</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))
    
    # save map
    m.save(output_file)
    print(f"[INFO] saved map to {output_file}")


def create_combined_map(G, driver_cycles, output_file):
    """
    create combined map showing all driver routes.
    
    args:
        G: networkx graph
        driver_cycles: dict mapping driver_id to cycle data
        output_file: where to save html
    """
    print("[INFO] creating combined overview map...")
    
    # create map centered on ithaca
    m = folium.Map(
        location=[42.45, -76.5],
        zoom_start=13,
        tiles="OpenStreetMap"
    )
    
    colors = ["red", "blue", "green", "purple", "orange", "darkred", "lightblue", "darkgreen"]
    
    # create feature groups for each driver
    driver_groups = {}
    for driver_id in sorted(driver_cycles.keys()):
        driver_groups[driver_id] = folium.FeatureGroup(name=f"Driver {driver_id}")
    
    # add routes for each driver
    for driver_id in sorted(driver_cycles.keys()):
        cycle_data = driver_cycles[driver_id]
        cycle = cycle_data["cycle"]
        color = colors[driver_id % len(colors)]
        
        cumulative_time = 0.0
        
        for i, (u, v, k) in enumerate(cycle):
            if G.has_edge(u, v, key=k):
                edge_data = G[u][v][k]
                
                if "geometry" in edge_data:
                    geom = edge_data["geometry"]
                    geom_latlon = convert_geometry_to_latlon(geom)
                    
                    # handle multilinestring
                    if hasattr(geom_latlon, "geoms"):
                        for line in geom_latlon.geoms:
                            coords = [(pt[1], pt[0]) for pt in line.coords]
                            tooltip = f"Driver {driver_id} - {edge_data['name']}"
                            folium.PolyLine(
                                coords,
                                color=color,
                                weight=3,
                                opacity=0.7,
                                tooltip=tooltip
                            ).add_to(driver_groups[driver_id])
                    else:
                        coords = [(pt[1], pt[0]) for pt in geom_latlon.coords]
                        tooltip = f"Driver {driver_id} - {edge_data['name']}"
                        folium.PolyLine(
                            coords,
                            color=color,
                            weight=3,
                            opacity=0.7,
                            tooltip=tooltip
                        ).add_to(driver_groups[driver_id])
                    
                    cumulative_time += edge_data["travel_time"]
    
    # add all feature groups to map
    for driver_id, group in driver_groups.items():
        group.add_to(m)
    
    # add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; 
                background-color: white; border:2px solid grey; 
                z-index:9999; font-size:14px; padding: 10px;
                border-radius: 5px;">
    <h4 style="margin: 0 0 10px 0;">Driver Routes</h4>
    """
    
    for driver_id in sorted(driver_cycles.keys()):
        color = colors[driver_id % len(colors)]
        cycle_data = driver_cycles[driver_id]
        time = cycle_data["evaluation"]["cycle_time"]
        benefit = cycle_data["evaluation"]["benefit_per_minute"]
        legend_html += f'<p style="margin: 5px 0;"><span style="color: {color}; font-size: 20px;">&#9632;</span> Driver {driver_id}: {time:.1f}min, {benefit:.0f} b/m</p>'
    
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # save map
    m.save(output_file)
    print(f"[INFO] saved combined map to {output_file}")


def create_static_summary(G, driver_cycles, output_file):
    """
    create static matplotlib summary plot.
    
    args:
        G: networkx graph
        driver_cycles: dict mapping driver_id to cycle data
        output_file: where to save png
    """
    print("[INFO] creating static summary plot...")
    
    Path(output_file).parent.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    colors = ["red", "blue", "green", "purple", "orange", "darkred", "cyan", "darkgreen"]
    
    for driver_id in sorted(driver_cycles.keys()):
        cycle_data = driver_cycles[driver_id]
        cycle = cycle_data["cycle"]
        color = colors[driver_id % len(colors)]
        
        for u, v, k in cycle:
            if G.has_edge(u, v, key=k):
                edge_data = G[u][v][k]
                
                if "geometry" in edge_data:
                    geom = edge_data["geometry"]
                    
                    # handle multilinestring
                    if hasattr(geom, "geoms"):
                        for line in geom.geoms:
                            x, y = line.xy
                            ax.plot(x, y, color=color, linewidth=2, alpha=0.7, label=f"Driver {driver_id}" if u == cycle[0][0] else "")
                    else:
                        x, y = geom.xy
                        ax.plot(x, y, color=color, linewidth=2, alpha=0.7, label=f"Driver {driver_id}" if u == cycle[0][0] else "")
    
    ax.set_title("all driver routes - partition set 2", fontsize=16, fontweight="bold")
    ax.set_xlabel("x coordinate (ny state plane feet)")
    ax.set_ylabel("y coordinate (ny state plane feet)")
    
    # create legend with unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[INFO] saved static summary to {output_file}")


def create_statistics_table(G, driver_cycles, output_file):
    """
    create csv with detailed statistics per driver.
    
    args:
        G: networkx graph
        driver_cycles: dict mapping driver_id to cycle data
        output_file: where to save csv
    """
    print("[INFO] creating statistics table...")
    
    stats_rows = []
    
    for driver_id in sorted(driver_cycles.keys()):
        cycle_data = driver_cycles[driver_id]
        cycle = cycle_data["cycle"]
        
        # calculate statistics
        total_length = 0.0
        total_priority = 0.0
        
        for u, v, k in cycle:
            if G.has_edge(u, v, key=k):
                edge_data = G[u][v][k]
                total_length += edge_data["length_feet"]
                total_priority += edge_data["priority"]
        
        stats_rows.append({
            "driver_id": driver_id,
            "partition_id": cycle_data.get("partition_id", driver_id),
            "method": cycle_data.get("method", "N/A"),
            "total_length_feet": total_length,
            "total_length_miles": total_length / 5280,
            "total_time_minutes": cycle_data["evaluation"]["cycle_time"],
            "num_edges": len(cycle),
            "unique_edges": cycle_data["metrics"]["unique_edges"],
            "total_priority": total_priority,
            "avg_priority_per_edge": total_priority / len(cycle) if cycle else 0,
            "benefit": cycle_data["evaluation"]["benefit"],
            "benefit_per_minute": cycle_data["evaluation"]["benefit_per_minute"],
            "start_node": cycle[0][0] if cycle else "N/A",
            "end_node": cycle[-1][1] if cycle else "N/A",
        })
    
    df = pd.DataFrame(stats_rows)
    df.to_csv(output_file, index=False)
    print(f"[INFO] saved statistics to {output_file}")
    
    # print summary
    print("\n[SUMMARY] driver statistics:")
    print(df.to_string(index=False))


def main():
    print("="*80)
    print("snow plow route visualization")
    print("="*80)
    print()
    
    # load data
    G, optimized_cycles, operational_plans = load_data()
    
    # get partition set 2 cycles (best performing)
    partition_set_id = 2
    print(f"\n[INFO] extracting routes for partition set {partition_set_id}...")
    driver_cycles = get_partition_set_cycles(optimized_cycles, partition_set_id)
    print(f"[INFO] found {len(driver_cycles)} driver routes")
    
    # create html output directory
    Path("html").mkdir(exist_ok=True)
    
    # create individual driver maps
    print("\n[INFO] creating individual driver maps...")
    for driver_id in sorted(driver_cycles.keys()):
        output_file = f"html/driver_{driver_id}_route.html"
        create_driver_map(G, driver_id, driver_cycles[driver_id], output_file)
    
    # create combined map
    print("\n[INFO] creating combined overview map...")
    create_combined_map(G, driver_cycles, "html/all_drivers_routes.html")
    
    # create static summary
    print("\n[INFO] creating static summary plot...")
    create_static_summary(G, driver_cycles, "plots/all_drivers_summary.png")
    
    # create statistics table
    print("\n[INFO] creating statistics table...")
    create_statistics_table(G, driver_cycles, "route_statistics.csv")
    
    print("\n" + "="*80)
    print("[PASS] visualization complete!")
    print("="*80)
    print("\n[INFO] output files created:")
    print("  individual maps:")
    for driver_id in sorted(driver_cycles.keys()):
        print(f"    - html/driver_{driver_id}_route.html")
    print("  combined map:")
    print("    - html/all_drivers_routes.html")
    print("  static summary:")
    print("    - plots/all_drivers_summary.png")
    print("  statistics:")
    print("    - route_statistics.csv")
    
    print("\n[INFO] open the html files in your browser to view interactive maps!")


if __name__ == "__main__":
    main()

