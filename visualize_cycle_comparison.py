#!/usr/bin/env python3
"""
visualize and compare pre-optimization cycles.
creates interactive maps with layer toggles to explore different cycle options.
"""

import json
import pickle
import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from pathlib import Path
import pyproj
from shapely.ops import transform


def load_data():
    """
    load necessary data files.
    
    returns:
        tuple of (graph, selected_cycles, optimized_cycles)
    """
    print("[INFO] loading data files...")
    
    # load graph
    with open("road_graph.gpickle", "rb") as f:
        G = pickle.load(f)
    
    # load selected cycles (top 10 per partition)
    with open("selected_cycles.json", "r") as f:
        selected_cycles = json.load(f)
    
    # load optimized cycles
    with open("optimized_cycles.json", "r") as f:
        optimized_cycles = json.load(f)
    
    print(f"[INFO] loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G, selected_cycles, optimized_cycles


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


def get_rank_color(rank, total):
    """
    get color for cycle based on rank (best = green, worst = red).
    
    args:
        rank: 0-indexed rank (0 = best)
        total: total number of items
    
    returns:
        hex color string
    """
    # create color gradient from green to orange to red
    if total == 1:
        return "#00ff00"  # green
    
    ratio = rank / (total - 1)
    
    if ratio < 0.5:
        # green to orange
        r = int(255 * (ratio * 2))
        g = 255
        b = 0
    else:
        # orange to red
        r = 255
        g = int(255 * (1 - (ratio - 0.5) * 2))
        b = 0
    
    return f"#{r:02x}{g:02x}{b:02x}"


def create_top10_cycles_map(G, partition_id, cycles, optimized_cycle, output_file):
    """
    create map showing top 10 cycles for a partition with layer controls.
    
    args:
        G: networkx graph
        partition_id: partition number
        cycles: list of top 10 cycle data dicts
        optimized_cycle: optimized cycle data dict
        output_file: where to save html
    """
    print(f"[INFO] creating top 10 cycles map for partition {partition_id}...")
    
    # create map centered on ithaca
    m = folium.Map(
        location=[42.45, -76.5],
        zoom_start=13,
        tiles="OpenStreetMap"
    )
    
    # create feature group for each cycle
    cycle_groups = []
    
    # add top 10 generated cycles
    for rank, cycle_data in enumerate(cycles[:10]):
        cycle = cycle_data["cycle"]
        method = cycle_data.get("method", "unknown")
        benefit_per_min = cycle_data["evaluation"]["benefit_per_minute"]
        cycle_time = cycle_data["evaluation"]["cycle_time"]
        
        color = get_rank_color(rank, 10)
        
        group_name = f"Rank {rank+1}: {method} (b/m={benefit_per_min:.0f}, t={cycle_time:.1f}min)"
        feature_group = folium.FeatureGroup(name=group_name, show=(rank == 0))
        
        # add route segments
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
                            tooltip = f"Rank {rank+1} - {edge_data['name']} (priority: {edge_data['priority']:.3f})"
                            folium.PolyLine(
                                coords,
                                color=color,
                                weight=4,
                                opacity=0.7,
                                tooltip=tooltip
                            ).add_to(feature_group)
                    else:
                        coords = [(pt[1], pt[0]) for pt in geom_latlon.coords]
                        tooltip = f"Rank {rank+1} - {edge_data['name']} (priority: {edge_data['priority']:.3f})"
                        folium.PolyLine(
                            coords,
                            color=color,
                            weight=4,
                            opacity=0.7,
                            tooltip=tooltip
                        ).add_to(feature_group)
        
        feature_group.add_to(m)
        cycle_groups.append((rank+1, group_name, color, benefit_per_min))
    
    # add optimized cycle as special layer
    if optimized_cycle:
        opt_cycle = optimized_cycle["cycle"]
        opt_benefit = optimized_cycle["evaluation"]["benefit_per_minute"]
        opt_time = optimized_cycle["evaluation"]["cycle_time"]
        
        group_name = f"★ OPTIMIZED (b/m={opt_benefit:.0f}, t={opt_time:.1f}min)"
        feature_group = folium.FeatureGroup(name=group_name, show=False)
        
        for u, v, k in opt_cycle:
            if G.has_edge(u, v, key=k):
                edge_data = G[u][v][k]
                
                if "geometry" in edge_data:
                    geom = edge_data["geometry"]
                    geom_latlon = convert_geometry_to_latlon(geom)
                    
                    if hasattr(geom_latlon, "geoms"):
                        for line in geom_latlon.geoms:
                            coords = [(pt[1], pt[0]) for pt in line.coords]
                            tooltip = f"OPTIMIZED - {edge_data['name']} (priority: {edge_data['priority']:.3f})"
                            folium.PolyLine(
                                coords,
                                color="#0000ff",  # blue for optimized
                                weight=5,
                                opacity=0.9,
                                tooltip=tooltip
                            ).add_to(feature_group)
                    else:
                        coords = [(pt[1], pt[0]) for pt in geom_latlon.coords]
                        tooltip = f"OPTIMIZED - {edge_data['name']} (priority: {edge_data['priority']:.3f})"
                        folium.PolyLine(
                            coords,
                            color="#0000ff",
                            weight=5,
                            opacity=0.9,
                            tooltip=tooltip
                        ).add_to(feature_group)
        
        feature_group.add_to(m)
    
    # add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # add info box with statistics
    stats_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 350px; 
                background-color: white; border:2px solid grey; 
                z-index:9999; font-size:12px; padding: 10px;
                border-radius: 5px; max-height: 90vh; overflow-y: auto;">
    <h4 style="margin: 0 0 10px 0;">Partition {partition_id} - Top 10 Cycles</h4>
    <p style="margin: 5px 0; font-size: 11px;"><b>Use layer control</b> (top right) to toggle cycles on/off</p>
    <hr style="margin: 10px 0;">
    <table style="width: 100%; font-size: 11px; border-collapse: collapse;">
    <tr style="background-color: #f0f0f0;">
        <th style="padding: 3px; text-align: left;">Rank</th>
        <th style="padding: 3px; text-align: right;">Benefit/min</th>
        <th style="padding: 3px; text-align: center;">Color</th>
    </tr>
    """
    
    for rank, name, color, benefit in cycle_groups:
        stats_html += f"""
        <tr>
            <td style="padding: 3px;">{rank}</td>
            <td style="padding: 3px; text-align: right;">{benefit:.0f}</td>
            <td style="padding: 3px; text-align: center;"><span style="color: {color}; font-size: 16px;">●</span></td>
        </tr>
        """
    
    if optimized_cycle:
        stats_html += f"""
        <tr style="background-color: #e6f2ff;">
            <td style="padding: 3px;"><b>★ OPT</b></td>
            <td style="padding: 3px; text-align: right;"><b>{opt_benefit:.0f}</b></td>
            <td style="padding: 3px; text-align: center;"><span style="color: #0000ff; font-size: 16px;">●</span></td>
        </tr>
        """
    
    stats_html += """
    </table>
    <hr style="margin: 10px 0;">
    <p style="margin: 5px 0; font-size: 10px;"><b>Legend:</b><br>
    🟢 Green = Best<br>
    🟠 Orange = Middle<br>
    🔴 Red = Worst<br>
    🔵 Blue = Optimized</p>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(stats_html))
    
    # save map
    m.save(output_file)
    print(f"[INFO] saved map to {output_file}")


def create_comparison_map(G, partition_id, best_cycle, worst_cycle, optimized_cycle, output_file):
    """
    create comparison map showing best, worst, and optimized cycles.
    
    args:
        G: networkx graph
        partition_id: partition number
        best_cycle: best generated cycle
        worst_cycle: worst generated cycle
        optimized_cycle: optimized cycle
        output_file: where to save html
    """
    print(f"[INFO] creating comparison map for partition {partition_id}...")
    
    # create map
    m = folium.Map(
        location=[42.45, -76.5],
        zoom_start=13,
        tiles="OpenStreetMap"
    )
    
    cycles_to_show = [
        ("Best Generated", best_cycle, "#00ff00", 4, True),
        ("Worst Generated", worst_cycle, "#ff0000", 3, False),
        ("Optimized", optimized_cycle, "#0000ff", 5, False),
    ]
    
    for label, cycle_data, color, weight, show in cycles_to_show:
        cycle = cycle_data["cycle"]
        benefit = cycle_data["evaluation"]["benefit_per_minute"]
        time = cycle_data["evaluation"]["cycle_time"]
        method = cycle_data.get("method", "N/A")
        
        group_name = f"{label} ({method}, b/m={benefit:.0f}, t={time:.1f}min)"
        feature_group = folium.FeatureGroup(name=group_name, show=show)
        
        for u, v, k in cycle:
            if G.has_edge(u, v, key=k):
                edge_data = G[u][v][k]
                
                if "geometry" in edge_data:
                    geom = edge_data["geometry"]
                    geom_latlon = convert_geometry_to_latlon(geom)
                    
                    if hasattr(geom_latlon, "geoms"):
                        for line in geom_latlon.geoms:
                            coords = [(pt[1], pt[0]) for pt in line.coords]
                            tooltip = f"{label} - {edge_data['name']}"
                            folium.PolyLine(
                                coords,
                                color=color,
                                weight=weight,
                                opacity=0.8,
                                tooltip=tooltip
                            ).add_to(feature_group)
                    else:
                        coords = [(pt[1], pt[0]) for pt in geom_latlon.coords]
                        tooltip = f"{label} - {edge_data['name']}"
                        folium.PolyLine(
                            coords,
                            color=color,
                            weight=weight,
                            opacity=0.8,
                            tooltip=tooltip
                        ).add_to(feature_group)
        
        feature_group.add_to(m)
    
    # add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # add comparison info
    improvement = ((optimized_cycle["evaluation"]["benefit_per_minute"] - 
                   best_cycle["evaluation"]["benefit_per_minute"]) / 
                   best_cycle["evaluation"]["benefit_per_minute"] * 100)
    
    info_html = f"""
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 300px; 
                background-color: white; border:2px solid grey; 
                z-index:9999; font-size:13px; padding: 10px;
                border-radius: 5px;">
    <h4 style="margin: 0 0 10px 0;">Partition {partition_id} - Comparison</h4>
    <table style="width: 100%; font-size: 12px;">
    <tr><td><b>Best Generated:</b></td><td style="text-align: right;">{best_cycle["evaluation"]["benefit_per_minute"]:.0f} b/m</td></tr>
    <tr><td><b>Worst Generated:</b></td><td style="text-align: right;">{worst_cycle["evaluation"]["benefit_per_minute"]:.0f} b/m</td></tr>
    <tr style="background-color: #e6f2ff;"><td><b>Optimized:</b></td><td style="text-align: right;">{optimized_cycle["evaluation"]["benefit_per_minute"]:.0f} b/m</td></tr>
    <tr><td colspan="2"><hr style="margin: 5px 0;"></td></tr>
    <tr><td><b>Improvement:</b></td><td style="text-align: right; color: {"green" if improvement > 0 else "red"};">{improvement:+.1f}%</td></tr>
    </table>
    <p style="margin: 10px 0 0 0; font-size: 11px;">
    Toggle layers to compare routes!<br>
    🟢 Green = Best initial<br>
    🔴 Red = Worst initial<br>
    🔵 Blue = After optimization
    </p>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(info_html))
    
    # save map
    m.save(output_file)
    print(f"[INFO] saved comparison map to {output_file}")


def main():
    print("="*80)
    print("cycle comparison visualization")
    print("="*80)
    print()
    
    # load data
    G, selected_cycles, optimized_cycles = load_data()
    
    # create html directory
    Path("html").mkdir(exist_ok=True)
    
    # process partition set 2 (best performing)
    partition_set_id = 2
    
    print(f"\n[INFO] creating visualizations for partition set {partition_set_id}...")
    
    # get partitions for set 2
    partitions = {}
    for key, cycles in selected_cycles.items():
        if key.startswith(f"ps{partition_set_id}_"):
            partition_id = int(key.split("_")[1][1:])
            partitions[partition_id] = cycles
    
    # create visualizations for each partition
    for partition_id in sorted(partitions.keys()):
        cycles = partitions[partition_id]
        
        # get optimized cycle
        opt_key = f"ps{partition_set_id}_p{partition_id}"
        optimized_cycle = optimized_cycles[opt_key][0] if opt_key in optimized_cycles else None
        
        # create top 10 cycles map
        output_file = f"html/partition_{partition_id}_top10_cycles.html"
        create_top10_cycles_map(G, partition_id, cycles, optimized_cycle, output_file)
        
        # create comparison map (best vs worst vs optimized)
        if len(cycles) >= 2 and optimized_cycle:
            best_cycle = cycles[0]  # already sorted by benefit
            worst_cycle = cycles[-1]
            
            output_file = f"html/partition_{partition_id}_comparison.html"
            create_comparison_map(G, partition_id, best_cycle, worst_cycle, optimized_cycle, output_file)
    
    print("\n" + "="*80)
    print("[PASS] visualization complete!")
    print("="*80)
    print("\n[INFO] output files created:")
    print("  top 10 cycles (toggle layers):")
    for partition_id in sorted(partitions.keys()):
        print(f"    - html/partition_{partition_id}_top10_cycles.html")
    print("\n  comparison views (best vs worst vs optimized):")
    for partition_id in sorted(partitions.keys()):
        print(f"    - html/partition_{partition_id}_comparison.html")
    
    print("\n[INFO] open the html files in your browser to explore cycles!")
    print("  - use layer control to toggle cycles on/off")
    print("  - hover over routes to see road details")


if __name__ == "__main__":
    main()

