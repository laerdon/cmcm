#!/usr/bin/env python3
"""
Partition road graph into 8 spatially-clustered subgraphs with balanced priority.
Combines spectral partitioning for clustering with priority-aware refinement.
"""


import networkx as nx
import numpy as np
import json
import pickle
import pandas as pd
from collections import defaultdict




def calculate_partition_priority(G, nodes):
   """Calculate total edge priority within a partition."""
   total_priority = 0.0
   nodes_set = set(nodes)
   for u, v, data in G.edges(data=True):
       if u in nodes_set and v in nodes_set:
           total_priority += data["priority"]
   return total_priority




def spectral_partition_balanced(G, nodes, node_priorities, rng):
   """
   Partition nodes using spectral clustering with priority-aware split point.
  
   Args:
       G: networkx graph
       nodes: set of nodes to partition
       node_priorities: dict mapping node to its total incident edge priority
       rng: numpy random generator for reproducible randomness
  
   Returns:
       tuple of (partition1, partition2) as sets of nodes
   """
   if len(nodes) <= 2:
       nodes_list = list(nodes)
       rng.shuffle(nodes_list)
       return {nodes_list[0]}, set(nodes_list[1:]) if len(nodes) > 1 else set()


   try:
       # Create a proper copy of the subgraph (not a view)
       subgraph = G.subgraph(nodes).copy()
      
       # Convert to undirected - this creates a new graph, not a view
       G_undirected = nx.Graph()
       G_undirected.add_nodes_from(subgraph.nodes())
      
       # Add edges with random perturbation for diversity
       for u, v in subgraph.edges():
           weight = 1.0 + rng.uniform(-0.1, 0.1)
           # Add edge in both directions (undirected)
           if G_undirected.has_edge(u, v):
               G_undirected[u][v]['weight'] = (G_undirected[u][v]['weight'] + weight) / 2
           else:
               G_undirected.add_edge(u, v, weight=weight)
      
       # Ensure graph is connected, otherwise spectral won't work well
       if not nx.is_connected(G_undirected):
           # Find largest connected component
           largest_cc = max(nx.connected_components(G_undirected), key=len)
           remaining = set(nodes) - largest_cc
          
           if len(remaining) > 0 and len(largest_cc) > 0:
               # Put smaller components in one partition
               return largest_cc, remaining
      
       # Compute Laplacian and Fiedler vector
       laplacian = nx.laplacian_matrix(G_undirected, weight='weight').todense()
       eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
      
       # Second smallest eigenvalue's eigenvector (Fiedler vector)
       fiedler_vector = eigenvectors[:, 1]
      
       # Sort nodes by Fiedler vector value
       nodes_array = np.array(list(G_undirected.nodes()))
       sorted_indices = np.argsort(fiedler_vector)
       sorted_nodes = nodes_array[sorted_indices]
      
       # Find split point that balances priority (not just node count)
       total_priority = sum(node_priorities.get(n, 0) for n in sorted_nodes)
       target_priority = total_priority / 2
      
       cumulative_priority = 0
       best_split = len(sorted_nodes) // 2
       best_diff = float('inf')
      
       for i in range(1, len(sorted_nodes)):
           cumulative_priority += node_priorities.get(sorted_nodes[i-1], 0)
           diff = abs(cumulative_priority - target_priority)
           if diff < best_diff:
               best_diff = diff
               best_split = i
      
       partition1 = set(sorted_nodes[:best_split])
       partition2 = set(sorted_nodes[best_split:])
      
       # Ensure both partitions are non-empty
       if len(partition1) == 0 or len(partition2) == 0:
           mid = len(sorted_nodes) // 2
           partition1 = set(sorted_nodes[:mid])
           partition2 = set(sorted_nodes[mid:])
      
       return partition1, partition2
      
   except Exception as e:
       print(f"[WARNING] Spectral partitioning failed: {e}, using fallback")
       nodes_list = list(nodes)
       rng.shuffle(nodes_list)
       mid = len(nodes_list) // 2
       return set(nodes_list[:mid]), set(nodes_list[mid:])




def compute_node_priorities(G):
   """Compute priority for each node as sum of incident edge priorities."""
   node_priorities = defaultdict(float)
   for u, v, data in G.edges(data=True):
       priority = data.get("priority", 0)
       node_priorities[u] += priority
       node_priorities[v] += priority
   return dict(node_priorities)




def recursive_balanced_partition(G, nodes, node_priorities, target_partitions, rng):
   """
   Recursively partition graph using spectral clustering with priority balance.
  
   Args:
       G: networkx graph
       nodes: set of nodes to partition
       node_priorities: dict of node priorities
       target_partitions: number of partitions (must be power of 2)
       rng: numpy random generator
  
   Returns:
       list of node sets, one per partition
   """
   if target_partitions == 1:
       return [nodes]
  
   if len(nodes) < target_partitions:
       # Edge case: fewer nodes than target partitions
       return [set([n]) for n in nodes] + [set() for _ in range(target_partitions - len(nodes))]
  
   # Partition into two using priority-aware spectral split
   partition1, partition2 = spectral_partition_balanced(G, nodes, node_priorities, rng)
  
   # Calculate how many partitions each half should get based on priority
   total_priority = sum(node_priorities.get(n, 0) for n in nodes)
   priority1 = sum(node_priorities.get(n, 0) for n in partition1)
   priority2 = sum(node_priorities.get(n, 0) for n in partition2)
  
   # Allocate partitions proportionally to priority (but at least 1 each)
   if priority1 + priority2 > 0:
       ratio1 = priority1 / (priority1 + priority2)
       partitions_for_1 = max(1, min(target_partitions - 1, round(target_partitions * ratio1)))
       partitions_for_2 = target_partitions - partitions_for_1
   else:
       # Fallback: split evenly
       partitions_for_1 = target_partitions // 2
       partitions_for_2 = target_partitions - partitions_for_1
  
   # Recursively partition each half
   partitions1 = recursive_balanced_partition(G, partition1, node_priorities, partitions_for_1, rng)
   partitions2 = recursive_balanced_partition(G, partition2, node_priorities, partitions_for_2, rng)
  
   return partitions1 + partitions2




def compute_spatial_compactness(G, partition):
   """
   Compute how spatially compact a partition is.
   Lower is better (nodes are closer together).
   """
   if len(partition) <= 1:
       return 0.0
  
   partition_list = list(partition)
   total_distance = 0
   count = 0
  
   # Sample pairs to avoid O(n^2) for large partitions
   sample_size = min(100, len(partition_list))
   rng_local = np.random.default_rng(42)
   sampled = rng_local.choice(partition_list, size=sample_size, replace=False)
  
   for i, u in enumerate(sampled):
       for v in sampled[i+1:]:
           try:
               # Use shortest path length as distance metric
               dist = nx.shortest_path_length(G, u, v)
               total_distance += dist
               count += 1
           except nx.NetworkXNoPath:
               # If no path, assign large penalty
               total_distance += 1000
               count += 1
  
   return total_distance / count if count > 0 else 0.0




def remove_outliers_from_partitions(G, partitions, node_priorities, max_iterations=20):
   """
   Aggressively remove outlier nodes from partitions and reassign them to nearest partition.
   Prioritizes spatial clustering over individual node count balance.
  
   Args:
       G: networkx graph
       partitions: list of node sets
       node_priorities: dict of node priorities
       max_iterations: maximum refinement iterations
  
   Returns:
       refined list of node sets
   """
   partitions = [set(p) for p in partitions]
  
   for iteration in range(max_iterations):
       moved_any = False
      
       for i, partition in enumerate(partitions):
           if len(partition) <= 2:
               continue  # Don't break up tiny partitions
          
           outliers_to_move = []
          
           for node in list(partition):
               # Count how many neighbors are in the same partition
               neighbors = list(G.neighbors(node))
               same_partition_neighbors = sum(1 for n in neighbors if n in partition)
              
               # Check neighbors in other partitions
               other_partition_neighbors = defaultdict(int)
               for neighbor in neighbors:
                   for j, other_partition in enumerate(partitions):
                       if j != i and neighbor in other_partition:
                           other_partition_neighbors[j] += 1
              
               # If node has more connections to another partition, it's an outlier
               if other_partition_neighbors:
                   max_other_connections = max(other_partition_neighbors.values())
                   if max_other_connections > same_partition_neighbors:
                       # Find which partition it's most connected to
                       best_partition = max(other_partition_neighbors.items(), key=lambda x: x[1])[0]
                       outliers_to_move.append((node, best_partition))
          
           # Move outliers
           for node, target_partition_idx in outliers_to_move:
               partitions[i].remove(node)
               partitions[target_partition_idx].add(node)
               moved_any = True
      
       if not moved_any:
           break
  
   return partitions




def refine_partition_balance(G, partitions, node_priorities, max_iterations=200):
   """
   Refine partitions by moving boundary nodes to balance priority.
   STRONGLY prioritizes maintaining spatial clustering.
  
   Args:
       G: networkx graph
       partitions: list of node sets
       node_priorities: dict of node priorities
       max_iterations: maximum refinement iterations
  
   Returns:
       refined list of node sets
   """
   partitions = [set(p) for p in partitions]
  
   # Remove any empty partitions
   partitions = [p for p in partitions if len(p) > 0]
  
   for iteration in range(max_iterations):
       priorities = [calculate_partition_priority(G, p) for p in partitions]
       mean_priority = np.mean(priorities)
      
       # More lenient threshold - 15% variation is acceptable
       if np.std(priorities) / mean_priority < 0.15:
           break
      
       # Find most overloaded and most underloaded partitions
       max_idx = np.argmax(priorities)
       min_idx = np.argmin(priorities)
      
       if priorities[max_idx] - priorities[min_idx] < mean_priority * 0.05:
           break
      
       # Find boundary nodes: nodes with neighbors in other partitions
       # BUT only consider nodes that have MINORITY connections in current partition
       candidate_moves = []
      
       for i, partition in enumerate(partitions):
           if len(partition) == 0:
               continue
              
           # Consider moving from overloaded partitions OR partitions significantly above mean
           if priorities[i] < mean_priority * 0.9:
               continue
          
           for node in list(partition):
               # Count connections
               neighbors = list(G.neighbors(node))
               if len(neighbors) == 0:
                   continue
                  
               same_partition_neighbors = sum(1 for n in neighbors if n in partition)
              
               # Only move if node has weak connection to current partition
               if same_partition_neighbors <= len(neighbors) / 2:
                   node_priority = node_priorities.get(node, 0)
                  
                   # Find best target partition (must have actual connection)
                   for j, other_partition in enumerate(partitions):
                       if i == j or len(other_partition) == 0:
                           continue
                          
                       # Prioritize moving to underloaded partitions
                       if priorities[j] < mean_priority * 1.1:
                           connection_count = sum(1 for n in neighbors if n in other_partition)
                           if connection_count > 0:  # Must have actual neighbors
                               # Calculate benefit: how much this improves balance
                               balance_benefit = (priorities[i] - priorities[j]) / 2
                               # Weight by connection strength
                               total_benefit = balance_benefit * (connection_count / len(neighbors))
                               candidate_moves.append((total_benefit, node, i, j, node_priority, connection_count))
      
       if not candidate_moves:
           break
      
       # Sort by benefit and try best moves
       candidate_moves.sort(reverse=True)
      
       moved = False
       for _, node, from_idx, to_idx, node_prio, connections in candidate_moves[:10]:
           # Make sure move improves balance
           if connections < 1:
               continue
          
           if node not in partitions[from_idx]:
               continue  # Already moved
              
           new_from_priority = priorities[from_idx] - node_prio
           new_to_priority = priorities[to_idx] + node_prio
          
           # Check if move improves overall balance
           old_imbalance = abs(priorities[from_idx] - mean_priority) + abs(priorities[to_idx] - mean_priority)
           new_imbalance = abs(new_from_priority - mean_priority) + abs(new_to_priority - mean_priority)
          
           if new_imbalance < old_imbalance * 1.05:  # Allow slight increase for clustering
               partitions[from_idx].remove(node)
               partitions[to_idx].add(node)
               priorities[from_idx] = new_from_priority
               priorities[to_idx] = new_to_priority
               moved = True
               break
      
       if not moved:
           break
  
   return partitions




def merge_small_partitions(G, partitions, node_priorities, min_priority_threshold=0.5):
   """
   Merge partitions that are too small into their most connected neighbors.
  
   Args:
       G: networkx graph
       partitions: list of node sets
       node_priorities: dict of node priorities
       min_priority_threshold: minimum priority as fraction of mean
  
   Returns:
       merged list of node sets
   """
   priorities = [calculate_partition_priority(G, p) for p in partitions]
   mean_priority = np.mean(priorities)
   threshold = mean_priority * min_priority_threshold
  
   partitions = [set(p) for p in partitions]
  
   changed = True
   while changed:
       changed = False
       priorities = [calculate_partition_priority(G, p) for p in partitions]
      
       for i, partition in enumerate(partitions):
           if len(partition) == 0:
               continue
              
           if priorities[i] < threshold:
               # Find most connected neighboring partition
               best_neighbor = -1
               best_connection_count = 0
              
               for j, other_partition in enumerate(partitions):
                   if i == j or len(other_partition) == 0:
                       continue
                  
                   # Count edges between partitions
                   connection_count = 0
                   for node in partition:
                       for neighbor in G.neighbors(node):
                           if neighbor in other_partition:
                               connection_count += 1
                  
                   if connection_count > best_connection_count:
                       best_connection_count = connection_count
                       best_neighbor = j
              
               if best_neighbor >= 0:
                   # Merge partition i into best_neighbor
                   partitions[best_neighbor].update(partitions[i])
                   partitions[i] = set()
                   changed = True
                   break
  
   # Remove empty partitions
   return [p for p in partitions if len(p) > 0]




def split_large_partitions(G, partitions, node_priorities, rng, target_count=8):
   """
   Split partitions that are too large to reach target count.
  
   Args:
       G: networkx graph
       partitions: list of node sets
       node_priorities: dict of node priorities
       rng: random generator
       target_count: target number of partitions
  
   Returns:
       list of node sets
   """
   while len(partitions) < target_count:
       priorities = [calculate_partition_priority(G, p) for p in partitions]
      
       # Find largest partition by priority
       max_idx = np.argmax(priorities)
       largest_partition = partitions[max_idx]
      
       if len(largest_partition) <= 1:
           break  # Can't split further
      
       # Split this partition
       part1, part2 = spectral_partition_balanced(G, largest_partition, node_priorities, rng)
      
       # Replace the large partition with the two splits
       partitions = partitions[:max_idx] + [part1, part2] + partitions[max_idx+1:]
  
   return partitions




def create_partition_set(G, random_seed=0):
   """
   Create one set of 8 spatially-clustered, priority-balanced partitions.
  
   Args:
       G: networkx graph
       random_seed: random seed for reproducibility and diversity
  
   Returns:
       list of 8 node sets
   """
   rng = np.random.default_rng(random_seed)
  
   # Compute node priorities
   node_priorities = compute_node_priorities(G)
  
   all_nodes = set(G.nodes())
  
   # Create initial partition using spectral method with adaptive subdivision
   print(f"  Initial partitioning...")
   partitions = recursive_balanced_partition(G, all_nodes, node_priorities, 8, rng)
  
   # Remove any empty partitions
   partitions = [p for p in partitions if len(p) > 0]
   priorities = [calculate_partition_priority(G, p) for p in partitions]
   print(f"  Created {len(partitions)} partitions with priorities: {[f'{p:.2f}' for p in priorities]}")
  
   # Merge very small partitions
   print(f"  Merging small partitions...")
   partitions = merge_small_partitions(G, partitions, node_priorities, min_priority_threshold=0.3)
   priorities = [calculate_partition_priority(G, p) for p in partitions]
   print(f"  After merging: {len(partitions)} partitions with priorities: {[f'{p:.2f}' for p in priorities]}")
  
   # Split large partitions if we have too few
   if len(partitions) < 8:
       print(f"  Splitting large partitions to reach 8...")
       partitions = split_large_partitions(G, partitions, node_priorities, rng, target_count=8)
       priorities = [calculate_partition_priority(G, p) for p in partitions]
       print(f"  After splitting: {len(partitions)} partitions with priorities: {[f'{p:.2f}' for p in priorities]}")
  
   # Remove outliers aggressively
   print(f"  Removing outliers...")
   partitions = remove_outliers_from_partitions(G, partitions, node_priorities, max_iterations=30)
  
   priorities = [calculate_partition_priority(G, p) for p in partitions]
   print(f"  After outlier removal, priorities: {[f'{p:.2f}' for p in priorities]}")
  
   # Refine for better balance while maintaining clustering
   print(f"  Refining balance...")
   partitions = refine_partition_balance(G, partitions, node_priorities, max_iterations=300)
  
   priorities = [calculate_partition_priority(G, p) for p in partitions]
   print(f"  After refinement, priorities: {[f'{p:.2f}' for p in priorities]}")
  
   # Final check: if still very imbalanced, do one more merge-split cycle
   mean_priority = np.mean(priorities)
   if np.std(priorities) / mean_priority > 0.3:
       print(f"  Still imbalanced, doing second merge-split cycle...")
       partitions = merge_small_partitions(G, partitions, node_priorities, min_priority_threshold=0.4)
       partitions = split_large_partitions(G, partitions, node_priorities, rng, target_count=8)
       partitions = remove_outliers_from_partitions(G, partitions, node_priorities, max_iterations=20)
       partitions = refine_partition_balance(G, partitions, node_priorities, max_iterations=200)
       priorities = [calculate_partition_priority(G, p) for p in partitions]
       print(f"  After second cycle, priorities: {[f'{p:.2f}' for p in priorities]}")
  
   return [set(p) for p in partitions]




def analyze_partitions(G, partitions):
   """Analyze quality of partitions."""
   stats = {
       "num_partitions": len(partitions),
       "partition_sizes": [],
       "partition_priorities": [],
       "partition_edges": [],
       "partition_total_length": [],
   }
  
   for partition in partitions:
       subgraph = G.subgraph(partition)
       priority = calculate_partition_priority(G, partition)
       total_length = sum(
           data.get("length_feet", 0) for u, v, data in subgraph.edges(data=True)
       )
      
       stats["partition_sizes"].append(len(partition))
       stats["partition_priorities"].append(priority)
       stats["partition_edges"].append(subgraph.number_of_edges())
       stats["partition_total_length"].append(total_length)
  
   return stats




def main():
   print("[INFO] Loading road graph...")
   with open("road_graph.gpickle", "rb") as f:
       G = pickle.load(f)
   print(f"[INFO] Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
  
   # Create multiple partition sets
   num_partition_sets = 3
   partition_sets = []
  
   print(f"\n[INFO] Creating {num_partition_sets} partition sets...")
  
   for i in range(num_partition_sets):
       print(f"\n[INFO] Creating partition set {i+1}/{num_partition_sets}...")
       partitions = create_partition_set(G, random_seed=i)
      
       # Analyze quality
       stats = analyze_partitions(G, partitions)
      
       print(f"  Partition sizes: {stats['partition_sizes']}")
       print(f"  Partition priorities: {[f'{p:.2f}' for p in stats['partition_priorities']]}")
       print(f"  Priority balance (std/mean): {np.std(stats['partition_priorities'])/np.mean(stats['partition_priorities']):.3f}")
       print(f"  Priority range: {min(stats['partition_priorities']):.2f} - {max(stats['partition_priorities']):.2f}")
      
       partition_sets.append(partitions)
  
   # Save partitions
   output_file = "partitions.json"
   partition_sets_serializable = [
       [[int(node) for node in p] for p in partition_set]
       for partition_set in partition_sets
   ]
  
   with open(output_file, "w") as f:
       json.dump(partition_sets_serializable, f, indent=2)
  
   print(f"\n[INFO] Saved {num_partition_sets} partition sets to {output_file}")
  
   # Save detailed statistics
   summary_rows = []
   for i, partitions in enumerate(partition_sets):
       stats = analyze_partitions(G, partitions)
       for j in range(stats["num_partitions"]):
           summary_rows.append({
               "partition_set_id": i,
               "partition_id": j,
               "num_nodes": stats["partition_sizes"][j],
               "total_priority": stats["partition_priorities"][j],
               "num_edges": stats["partition_edges"][j],
               "total_length_feet": stats["partition_total_length"][j],
           })
  
   summary_df = pd.DataFrame(summary_rows)
   summary_df.to_csv("partition_statistics.csv", index=False)
   print(f"[INFO] Saved partition statistics to partition_statistics.csv")
  
   return partition_sets




if __name__ == "__main__":
   main()


#--------- Graph Visualization --------
import networkx as nx
import matplotlib.pyplot as plt
import json
import pickle
import numpy as np
from matplotlib.patches import Polygon, Patch
from scipy.spatial import ConvexHull


# Load the saved graph and partition data
with open("road_graph.gpickle", "rb") as f:
   G = pickle.load(f)


with open("partitions.json", "r") as f:
   partition_sets = json.load(f)


# Pick the first partition set to visualize
partitions = partition_sets[0]


# Map node → partition index
node_to_partition = {}
for i, partition in enumerate(partitions):
   for node in partition:
       node_to_partition[node] = i


# Choose a color palette (adjust as needed)
num_partitions = len(partitions)
colors = plt.cm.get_cmap("tab10", num_partitions)


# Use node coordinates if available; otherwise, use a layout
if "x" in list(G.nodes(data=True))[0][1]:
   pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
else:
   pos = nx.spring_layout(G, seed=42)


# Create figure and axis
plt.figure(figsize=(10, 10))
ax = plt.gca()


# 1️⃣ Draw faint gray edges (no arrows)
nx.draw_networkx_edges(
   G, pos, edge_color="lightgray", alpha=0.6, width=0.8, arrows=False
)


# 2️⃣ Draw rigid shaded polygons for each partition
legend_handles = []
for i, partition in enumerate(partitions):
   color = np.array(colors(i))  # RGBA
   points = np.array([pos[n] for n in partition if n in pos])


   if len(points) >= 3:  # Need at least 3 points for a hull
       try:
           hull = ConvexHull(points)
           polygon = Polygon(
               points[hull.vertices],
               closed=True,
               facecolor=color,
               alpha=0.15,  # Shading transparency
               edgecolor="none",
           )
           ax.add_patch(polygon)


           # Add this zone to the legend
           legend_handles.append(
               Patch(facecolor=color, edgecolor=color, label=f"Zone {i + 1}")
           )


       except Exception as e:
           print(f"Warning: Convex hull failed for zone {i + 1}: {e}")


# 3️⃣ Draw nodes (colored by partition)
nx.draw_networkx_nodes(
   G,
   pos,
   node_color=[colors(node_to_partition[n]) for n in G.nodes()],
   node_size=15,
   alpha=0.9,
)


# 4️⃣ Add legend on the right side
ax.legend(
   handles=legend_handles,
   title="Zones",
   loc="center left",
   bbox_to_anchor=(1.05, 0.5),
   frameon=False,
)


# 5️⃣ Styling and output
plt.title("Graph Partitioning of Ithaca into 8 Zones", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.savefig(
   "partition_visualization_shaded_with_legend.png",
   dpi=300,
   bbox_inches="tight"
)
plt.show()



