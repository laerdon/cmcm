#!/usr/bin/env python3
"""
visualize and compare pre-optimization cycles with simple javascript controls.
creates interactive maps with dropdown selectors to flip through cycles.
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import pyproj
from shapely.ops import transform


def load_data():
    """load necessary data files."""
    print("[INFO] loading data files...")
    
    with open("road_graph.gpickle", "rb") as f:
        G = pickle.load(f)
    
    with open("selected_cycles.json", "r") as f:
        selected_cycles = json.load(f)
    
    with open("optimized_cycles.json", "r") as f:
        optimized_cycles = json.load(f)
    
    print(f"[INFO] loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G, selected_cycles, optimized_cycles


def convert_geometry_to_latlon(geometry):
    """convert geometry from web mercator to wgs84."""
    project = pyproj.Transformer.from_crs(
        "EPSG:3857", "EPSG:4326", always_xy=True
    ).transform
    return transform(project, geometry)


def get_rank_color(rank, total):
    """get color for cycle based on rank (best = green, worst = red)."""
    if total == 1:
        return "#00ff00"
    
    ratio = rank / (total - 1)
    
    if ratio < 0.5:
        r = int(255 * (ratio * 2))
        g = 255
        b = 0
    else:
        r = 255
        g = int(255 * (1 - (ratio - 0.5) * 2))
        b = 0
    
    return f"#{r:02x}{g:02x}{b:02x}"


def get_cycle_paths(G, cycle):
    """
    extract lat/lon paths from a cycle.
    
    returns:
        list of coordinate lists for each edge
    """
    paths = []
    
    for u, v, k in cycle:
        if G.has_edge(u, v, key=k):
            edge_data = G[u][v][k]
            
            if "geometry" in edge_data:
                geom = edge_data["geometry"]
                geom_latlon = convert_geometry_to_latlon(geom)
                
                if hasattr(geom_latlon, "geoms"):
                    for line in geom_latlon.geoms:
                        coords = [[pt[1], pt[0]] for pt in line.coords]
                        paths.append({
                            "coords": coords,
                            "name": edge_data.get("name", "Unnamed"),
                            "priority": edge_data.get("priority", 0.0)
                        })
                else:
                    coords = [[pt[1], pt[0]] for pt in geom_latlon.coords]
                    paths.append({
                        "coords": coords,
                        "name": edge_data.get("name", "Unnamed"),
                        "priority": edge_data.get("priority", 0.0)
                    })
    
    return paths


def create_interactive_cycle_viewer(G, partition_id, cycles, optimized_cycle, output_file):
    """
    create html with javascript controls to flip through cycles.
    """
    print(f"[INFO] creating interactive viewer for partition {partition_id}...")
    
    # prepare cycle data
    cycle_data = []
    
    for rank, cycle_info in enumerate(cycles[:10]):
        cycle = cycle_info["cycle"]
        method = cycle_info.get("method", "unknown")
        benefit = cycle_info["evaluation"]["benefit_per_minute"]
        time = cycle_info["evaluation"]["cycle_time"]
        color = get_rank_color(rank, 10)
        
        paths = get_cycle_paths(G, cycle)
        
        cycle_data.append({
            "id": f"cycle_{rank}",
            "rank": rank + 1,
            "method": method,
            "benefit": benefit,
            "time": time,
            "color": color,
            "paths": paths
        })
    
    # add optimized cycle
    if optimized_cycle:
        opt_cycle = optimized_cycle["cycle"]
        opt_benefit = optimized_cycle["evaluation"]["benefit_per_minute"]
        opt_time = optimized_cycle["evaluation"]["cycle_time"]
        
        paths = get_cycle_paths(G, opt_cycle)
        
        cycle_data.append({
            "id": "cycle_optimized",
            "rank": "OPT",
            "method": "optimized",
            "benefit": opt_benefit,
            "time": opt_time,
            "color": "#0000ff",
            "paths": paths
        })
    
    # convert to json
    cycle_json = json.dumps(cycle_data)
    
    # create html
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Partition {partition_id} - Cycle Viewer</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        #map {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 100%;
        }}
        .control-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            z-index: 1000;
            width: 320px;
        }}
        .control-panel h3 {{
            margin: 0 0 15px 0;
            font-size: 16px;
        }}
        .cycle-selector {{
            width: 100%;
            padding: 8px;
            font-size: 14px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 3px;
        }}
        .nav-buttons {{
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
        }}
        .nav-button {{
            flex: 1;
            padding: 8px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 14px;
        }}
        .nav-button:hover {{
            background: #0056b3;
        }}
        .nav-button:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}
        .stats-box {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 3px;
            font-size: 13px;
        }}
        .stats-box table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .stats-box td {{
            padding: 4px 0;
        }}
        .stats-box .label {{
            font-weight: bold;
        }}
        .stats-box .value {{
            text-align: right;
        }}
        .color-indicator {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 3px;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="control-panel">
        <h3>Partition {partition_id} - Cycle Viewer</h3>
        
        <select id="cycleSelector" class="cycle-selector">
            <!-- Options filled by JS -->
        </select>
        
        <div class="nav-buttons">
            <button id="prevBtn" class="nav-button">← Previous</button>
            <button id="nextBtn" class="nav-button">Next →</button>
        </div>
        
        <div id="statsBox" class="stats-box">
            <!-- Stats filled by JS -->
        </div>
    </div>

    <script>
        // Cycle data
        const cycles = {cycle_json};
        
        // Initialize map
        const map = L.map('map').setView([42.45, -76.5], 13);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);
        
        // Current cycle layer
        let currentLayer = null;
        let currentIndex = 0;
        
        // Populate selector
        const selector = document.getElementById('cycleSelector');
        cycles.forEach((cycle, idx) => {{
            const option = document.createElement('option');
            option.value = idx;
            option.textContent = `Rank ${{cycle.rank}} - ${{cycle.method}} (b/m=${{Math.round(cycle.benefit)}})`;
            selector.appendChild(option);
        }});
        
        // Display cycle
        function displayCycle(index) {{
            currentIndex = index;
            const cycle = cycles[index];
            
            // Remove old layer
            if (currentLayer) {{
                map.removeLayer(currentLayer);
            }}
            
            // Create new layer group
            currentLayer = L.layerGroup();
            
            // Add paths
            cycle.paths.forEach(path => {{
                const polyline = L.polyline(path.coords, {{
                    color: cycle.color,
                    weight: 5,
                    opacity: 0.8
                }});
                
                polyline.bindPopup(`
                    <b>${{path.name}}</b><br>
                    Priority: ${{path.priority.toFixed(3)}}
                `);
                
                currentLayer.addLayer(polyline);
            }});
            
            currentLayer.addTo(map);
            
            // Fit bounds to cycle
            if (cycle.paths.length > 0) {{
                const bounds = L.latLngBounds(
                    cycle.paths.flatMap(p => p.coords)
                );
                map.fitBounds(bounds, {{ padding: [50, 50] }});
            }}
            
            // Update stats
            updateStats(cycle);
            
            // Update selector
            selector.value = index;
            
            // Update buttons
            document.getElementById('prevBtn').disabled = (index === 0);
            document.getElementById('nextBtn').disabled = (index === cycles.length - 1);
        }}
        
        // Update stats box
        function updateStats(cycle) {{
            const statsBox = document.getElementById('statsBox');
            statsBox.innerHTML = `
                <table>
                    <tr>
                        <td class="label">Rank:</td>
                        <td class="value">${{cycle.rank}}</td>
                    </tr>
                    <tr>
                        <td class="label">Method:</td>
                        <td class="value">${{cycle.method}}</td>
                    </tr>
                    <tr>
                        <td class="label">Benefit/min:</td>
                        <td class="value">${{Math.round(cycle.benefit)}}</td>
                    </tr>
                    <tr>
                        <td class="label">Time:</td>
                        <td class="value">${{cycle.time.toFixed(1)}} min</td>
                    </tr>
                    <tr>
                        <td class="label">Edges:</td>
                        <td class="value">${{cycle.paths.length}}</td>
                    </tr>
                    <tr>
                        <td class="label">Color:</td>
                        <td class="value"><span class="color-indicator" style="background: ${{cycle.color}}"></span></td>
                    </tr>
                </table>
            `;
        }}
        
        // Event listeners
        selector.addEventListener('change', (e) => {{
            displayCycle(parseInt(e.target.value));
        }});
        
        document.getElementById('prevBtn').addEventListener('click', () => {{
            if (currentIndex > 0) {{
                displayCycle(currentIndex - 1);
            }}
        }});
        
        document.getElementById('nextBtn').addEventListener('click', () => {{
            if (currentIndex < cycles.length - 1) {{
                displayCycle(currentIndex + 1);
            }}
        }});
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft' && currentIndex > 0) {{
                displayCycle(currentIndex - 1);
            }} else if (e.key === 'ArrowRight' && currentIndex < cycles.length - 1) {{
                displayCycle(currentIndex + 1);
            }}
        }});
        
        // Display first cycle
        displayCycle(0);
    </script>
</body>
</html>
"""
    
    # write file
    Path(output_file).parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"[INFO] saved viewer to {output_file}")


def main():
    print("="*80)
    print("cycle comparison visualization (javascript version)")
    print("="*80)
    print()
    
    # load data
    G, selected_cycles, optimized_cycles = load_data()
    
    # create html directory
    Path("html").mkdir(exist_ok=True)
    
    # process partition set 2
    partition_set_id = 2
    
    print(f"\n[INFO] creating viewers for partition set {partition_set_id}...")
    
    # get partitions for set 2
    partitions = {}
    for key, cycles in selected_cycles.items():
        if key.startswith(f"ps{partition_set_id}_"):
            partition_id = int(key.split("_")[1][1:])
            partitions[partition_id] = cycles
    
    # create viewers for each partition
    for partition_id in sorted(partitions.keys()):
        cycles = partitions[partition_id]
        
        # get optimized cycle
        opt_key = f"ps{partition_set_id}_p{partition_id}"
        optimized_cycle = optimized_cycles[opt_key][0] if opt_key in optimized_cycles else None
        
        # create viewer
        output_file = f"html/partition_{partition_id}_cycles.html"
        create_interactive_cycle_viewer(G, partition_id, cycles, optimized_cycle, output_file)
    
    print("\n" + "="*80)
    print("[PASS] visualization complete!")
    print("="*80)
    print("\n[INFO] output files created:")
    for partition_id in sorted(partitions.keys()):
        print(f"  - html/partition_{partition_id}_cycles.html")
    
    print("\n[INFO] features:")
    print("  ✓ dropdown menu to select cycles")
    print("  ✓ previous/next buttons")
    print("  ✓ keyboard navigation (arrow keys)")
    print("  ✓ stats panel for each cycle")
    print("  ✓ color-coded by rank (green=best, red=worst, blue=optimized)")


if __name__ == "__main__":
    main()

