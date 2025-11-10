#!/usr/bin/env python3
"""
Visualize optimal snow plow routes with arrows showing the path for each driver.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch, Polygon, Patch, ArrowStyle
from matplotlib.collections import LineCollection
import json
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def load_optimal_cycles():
    """Load optimal cycles from CSV."""
    df = pd.read_csv("optimal_cycles.csv")
    
    # Load the detailed cycle paths from cycle_evaluations.json
    print("[INFO] Loading cycle evaluations...")
    with open("cycle_evaluations.json", "r") as f:
        all_cycles = json.load(f)
    
    print(f"[INFO] Found {len(all_cycles)} total cycles in evaluations")
    
    # Create a lookup for cycles
    cycle_lookup = {}
    for cycle_data in all_cycles:
        key = (
            cycle_data["partition_set_id"],
            cycle_data["partition_id"],
            cycle_data["method"],
            cycle_data["start_node"]
        )
        cycle_lookup[key] = cycle_data["cycle"]
    
    print(f"[INFO] Created lookup with {len(cycle_lookup)} cycles")
    
    # Match optimal cycles with their paths
    optimal_routes = {}
    for _, row in df.iterrows():
        partition_id = int(row["partition_id"])
        key = (
            int(row["partition_set_id"]),
            partition_id,
            row["method"],
            int(row["start_node"])
        )
        
        if key in cycle_lookup:
            cycle = cycle_lookup[key]
            print(f"[INFO] Matched Driver {partition_id + 1}: {len(cycle)} edges")
            optimal_routes[partition_id] = {
                "cycle": cycle,
                "start_node": int(row["start_node"]),
                "method": row["method"]
            }
        else:
            print(f"[WARNING] Could not find cycle for partition {partition_id}, key: {key}")
    
    return optimal_routes


def darken_color(color, factor=0.6):
    """Make a color darker by multiplying RGB values."""
    rgb = np.array(color[:3])  # Get RGB, ignore alpha
    darkened = rgb * factor
    return tuple(darkened) + (1.0,)  # Return with full alpha


def draw_route_with_arrows(ax, G, pos, route, color, alpha=0.9):
    """
    Draw all edges in route with arrows - ENSURES EVERY EDGE IS DRAWN.
    
    Args:
        ax: matplotlib axis
        G: networkx graph
        pos: node positions dict
        route: list of (from_node, to_node, edge_key) tuples
        color: color for the arrows
        alpha: transparency
    """
    print(f"    Drawing {len(route)} edges with arrows...")
    edges_drawn = 0
    edges_missing = 0
    edges_skipped = 0
    
    # Track unique edges to ensure we draw each one
    unique_edges = {}
    for u, v, k in route:
        edge_id = (u, v, k)
        if edge_id not in unique_edges:
            unique_edges[edge_id] = 0
        unique_edges[edge_id] += 1
    
    print(f"    Found {len(unique_edges)} unique edges (some traversed multiple times)")
    
    for (u, v, k), count in unique_edges.items():
        # Check if both nodes exist in position dict
        if u not in pos:
            print(f"    WARNING: Node {u} not in position dict")
            edges_missing += 1
            continue
        if v not in pos:
            print(f"    WARNING: Node {v} not in position dict")
            edges_missing += 1
            continue
            
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        # Calculate arrow position
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 1e-6:  # Skip zero-length edges
            print(f"    WARNING: Zero-length edge {u}→{v}")
            edges_skipped += 1
            continue
        
        # Create smaller, cleaner arrow with very short arrowhead
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->,head_width=0.15,head_length=0.2',
            color=color,
            alpha=alpha,
            linewidth=1.5,
            zorder=4,
            mutation_scale=8
        )
        ax.add_patch(arrow)
        edges_drawn += 1
    
    print(f"    Drew {edges_drawn} unique edges, {edges_missing} missing positions, {edges_skipped} skipped")
    
    if edges_missing > 0 or edges_skipped > 0:
        print(f"    ⚠️  WARNING: Not all edges were drawn for this zone!")
    
    return edges_drawn


def main():
    print("[INFO] Loading road graph...")
    with open("road_graph.gpickle", "rb") as f:
        G = pickle.load(f)
    
    print("[INFO] Loading partitions...")
    with open("partitions.json", "r") as f:
        partition_sets = json.load(f)
    
    # Use the first (optimal) partition set
    # Automatically detect which partition_set_id was used in the optimal_cycles.csv
    partitions = partition_sets[0]
    df_optimal = pd.read_csv("optimal_cycles.csv")
    partition_set_id_used = int(df_optimal["partition_set_id"].iloc[0])

    print(f"[INFO] Detected partition_set_id = {partition_set_id_used}")

    partitions = partition_sets[partition_set_id_used]
    
    print("[INFO] Loading optimal routes...")
    optimal_routes = load_optimal_cycles()
    print(f"[INFO] Loaded {len(optimal_routes)} optimal routes")
    
    # Map node to partition
    node_to_partition = {}
    for i, partition in enumerate(partitions):
        for node in partition:
            node_to_partition[node] = i
    
    # Get node positions
    if "x" in list(G.nodes(data=True))[0][1]:
        pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
    else:
        print("[WARNING] No coordinates found, using spring layout")
        pos = nx.spring_layout(G, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 18))
    
    # Color palette for zones
    num_partitions = len(partitions)
    zone_colors = plt.colormaps.get_cmap("tab10").resampled(num_partitions)
    
    # 1. Draw background edges (very faint)
    nx.draw_networkx_edges(
        G, pos, 
        edge_color="lightgray", 
        alpha=0.2, 
        width=0.4, 
        arrows=False,
        ax=ax
    )
    
    # 2. Draw partition zones (shaded polygons)
    print("[INFO] Drawing partition zones...")
    legend_zones = []
    zone_color_map = {}
    
    for i, partition in enumerate(partitions):
        base_color = zone_colors(i)
        zone_color_map[i] = base_color
        points = np.array([pos[n] for n in partition if n in pos])
        
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                polygon = Polygon(
                    points[hull.vertices],
                    closed=True,
                    facecolor=base_color,
                    alpha=0.15,
                    edgecolor='none',
                    zorder=1
                )
                ax.add_patch(polygon)
                legend_zones.append(Patch(facecolor=base_color, edgecolor=base_color, label=f"Zone {i + 1}"))
            except Exception as e:
                print(f"[WARNING] Convex hull failed for zone {i + 1}: {e}")
    
    # 3. Draw route arrows for each partition (darker version of zone color)
    print("[INFO] Drawing optimal routes with arrows...")
    print("[INFO] " + "="*60)
    legend_routes = []
    
    total_edges_expected = sum(len(route_data["cycle"]) for route_data in optimal_routes.values())
    total_unique_expected = 0
    
    # First pass: count unique edges per partition
    for partition_id, route_data in sorted(optimal_routes.items()):
        route = route_data["cycle"]
        unique = set((u, v, k) for u, v, k in route)
        total_unique_expected += len(unique)
    
    print(f"[INFO] Total edges to draw: {total_edges_expected} (including repeats)")
    print(f"[INFO] Total unique edges: {total_unique_expected}")
    print("[INFO] " + "="*60)
    
    total_drawn = 0
    all_edges_covered = True
    
    for partition_id, route_data in sorted(optimal_routes.items()):
        print(f"\n  ▶ Processing Driver {partition_id + 1} (Zone {partition_id + 1})...")
        route = route_data["cycle"]
        
        # Count expected unique edges for this partition
        expected_unique = len(set((u, v, k) for u, v, k in route))
        
        # Get darker version of zone color
        base_color = zone_color_map[partition_id]
        dark_color = darken_color(base_color, factor=0.5)
        
        # Draw all edges with arrows
        num_drawn = draw_route_with_arrows(ax, G, pos, route, dark_color, alpha=0.85)
        total_drawn += num_drawn
        
        # Verify coverage
        if num_drawn < expected_unique:
            print(f"    ❌ INCOMPLETE: Drew {num_drawn}/{expected_unique} edges")
            all_edges_covered = False
        else:
            print(f"    ✓ COMPLETE: Drew all {num_drawn} edges")
        
        # Add to legend
        legend_routes.append(
            mlines.Line2D(
                [], [],
                color=dark_color,
                marker='>',
                markersize=10,
                label=f"Driver {partition_id + 1} Route",
                linewidth=3
            )
        )
        
        # Mark start node with a large star
        start_node = route_data["start_node"]
        if start_node in pos:
            ax.plot(
                pos[start_node][0], 
                pos[start_node][1],
                marker='*',
                markersize=25,
                color='gold',
                markeredgecolor='black',
                markeredgewidth=2,
                zorder=10
            )
    
    # Print final summary
    print("\n" + "="*80)
    if all_edges_covered:
        print(f"SUCCESS: ALL {total_drawn}/{total_unique_expected} EDGES COVERED")
    else:
        print(f"WARNING: Only {total_drawn}/{total_unique_expected} edges covered")
    print("="*80 + "\n")
    
    # 4. Draw nodes (small dots colored by partition)
    node_colors = [zone_color_map.get(node_to_partition.get(n, 0), (0.5, 0.5, 0.5, 1.0)) for n in G.nodes()]
    nodes = ax.scatter(
        [pos[n][0] for n in G.nodes()],
        [pos[n][1] for n in G.nodes()],
        c=node_colors,
        s=10,
        alpha=0.7,
        zorder=2
    )
    
    # 5. Create combined legend
    print("[INFO] Creating legend...")
    
    # Add star marker to legend
    star_marker = mlines.Line2D(
        [], [],
        color='gold',
        marker='*',
        linestyle='None',
        markersize=18,
        label='Start Points',
        markeredgecolor='black',
        markeredgewidth=2
    )
    
    # Combine legends
    legend1 = ax.legend(
        handles=legend_zones,
        title="Zones",
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=11,
        title_fontsize=12
    )
    
    ax.add_artist(legend1)  # Add first legend
    
    legend2 = ax.legend(
        handles=legend_routes + [star_marker],
        title="Optimal Routes (Directed)",
        loc='upper left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=11,
        title_fontsize=12
    )
    
    # 6. Title and styling
    ax.set_title(
        "Optimal Snow Plow Routes - Ithaca Road Network\n" +
        "Dark arrows show complete driver paths | Gold stars mark start points",
        fontsize=17,
        fontweight='bold',
        pad=20
    )
    ax.axis('off')
    
    plt.tight_layout()
    plt.title(f"Optimal Snow Plow Routes", fontsize=18)
    
    # Save figure
    output_file = "optimal_routes_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[INFO] Saved visualization to {output_file}")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("VISUALIZATION SUMMARY")
    print("="*80)
    for partition_id, route_data in sorted(optimal_routes.items()):
        route = route_data["cycle"]
        print(f"Driver {partition_id + 1} (Zone {partition_id + 1}):")
        print(f"  Start Node: {route_data['start_node']}")
        print(f"  Method: {route_data['method']}")
        print(f"  Route Steps: {len(route)} edges")
    print("="*80)


if __name__ == "__main__":
    main()