# snow plow route optimization system

comprehensive optimization pipeline for ithaca snow plow operations during winter graduation event.

## overview

this system implements a graph-based approach to optimize snow plow routes by:
1. calculating priority scores based on emergency routes, graduation routes, parking, road class, and steepness
2. partitioning the road network into 8 balanced subgraphs
3. generating candidate plow cycles using smart sampling (greedy, random walks, chinese postman)
4. evaluating cycles based on accumulated snow cost
5. optimizing routes through local search with perturbations

## priority equation

```
priority(e) = 0.31×sEM + 0.25×sGR + 0.23×sGP + 0.13×sRC + 0.08×sST
```

where:
- sEM = emergency route score (1.0 for principal arterials, 0 otherwise)
- sGR = graduation route score (1.0 campus, 0.6 nearby, 0 otherwise)
- sGP = graduation parking score (1.0 within 2km of barton hall, 0 otherwise)
- sRC = road class score (0.25 local, 0.5 collector, 0.7 minor arterial, 0.85 principal arterial)
- sST = steepness score (0.2 for 0-5%, 0.4 for 5-10%, 0.6 for 10-15%, 0.8 for 15-20%, 1.0 for >20%)

## cost model

cost is calculated as the integral of priority × snow depth over time:

- **without plow**: `cost = priority × snowfall_rate × (time²/2)`
- **with plow**: cost accumulates between plow visits
- **benefit**: cost_without - cost_with

## installation

```bash
pip install -r requirements.txt
```

## usage

### run full pipeline

```bash
python3 main_pipeline.py
```

options:
- `--num-partitions N`: number of partition sets (default: 3)
- `--cycles-per-subgraph N`: cycles per subgraph (default: 50)
- `--snowfall-rate R`: inches per hour (default: 1.0)
- `--storm-duration H`: hours (default: 12)
- `--skip-to STEP`: skip to specific step
- `--skip-classification`: use existing classified roads

### run individual steps

1. **classify roads**:
   ```bash
   python3 classify_roads.py
   ```

2. **calculate priorities**:
   ```bash
   python3 calculate_priority.py
   ```

3. **build graph**:
   ```bash
   python3 build_graph.py
   ```

4. **partition graph**:
   ```bash
   python3 partition_graph.py
   ```

5. **generate cycles**:
   ```bash
   python3 generate_cycles.py
   ```

6. **evaluate cycles**:
   ```bash
   python3 evaluate_cycles.py
   ```

7. **sample best cycles**:
   ```bash
   python3 sample_cycles.py
   ```

8. **optimize routes**:
   ```bash
   python3 optimize_routes.py
   ```

9. **visualize results**:
   ```bash
   python3 visualize_results.py
   ```

## output files

### data files
- `classified_roads.csv`: roads with classification scores
- `classified_roads.geojson`: roads with geometry
- `roads_with_priority.csv`: roads with priority scores
- `road_graph.gpickle`: networkx graph structure
- `partitions.json`: graph partitions
- `candidate_cycles.json`: generated cycles
- `cycle_evaluations.json`: evaluated cycles with costs
- `selected_cycles.json`: top cycles per partition
- `optimized_cycles.json`: locally optimized cycles
- `operational_plans.json`: driver assignments

### reports
- `optimization_report.md`: comprehensive summary report
- `graph_statistics.csv`: graph metrics
- `partition_statistics.csv`: partition balance metrics
- `cycle_evaluations.csv`: cycle performance metrics

### visualizations
- `plots/priority_map.png`: road priority heatmap
- `plots/partition_map.png`: graph partitions
- `plots/benefit_by_method.png`: benefit distribution by method
- `plots/partition_balance.png`: priority balance across partitions
- `plots/cycle_time_distribution.png`: cycle time histogram

## key results

based on default parameters (1 inch/hour snowfall, 12-hour storm):

- **total road network**: 145 miles, 587 intersections
- **8 partitions** balanced by priority
- **1200 candidate cycles** generated using 3 methods
- **greedy method** produces best cycles (~91% of top cycles)
- **average cycle time**: ~51 minutes
- **local optimization**: +6.2% improvement
- **best plan (partition set 2)**: total benefit = 159,904, avg benefit/min = 19,667

### operational recommendations

1. use partition set 2 for highest total benefit
2. assign 8 drivers, one per partition
3. follow optimized greedy cycles
4. plan for multiple passes during 12-hour storm
5. prioritize emergency routes and campus access
6. coordinate salt usage based on temperature
7. ensure driver breaks comply with federal regulations

## algorithm details

### partitioning
- recursive spectral partitioning using laplacian eigenvectors
- balance by total priority with local refinement
- maintains connectivity within partitions

### cycle generation
1. **greedy**: always choose highest priority unvisited edge
2. **random walk**: probability-weighted by priority with bonus for unvisited
3. **chinese postman**: approximate eulerian tour covering all edges

### cycle evaluation
- cost = ∫(priority × snow_depth) dt
- linear snow accumulation between plow passes
- benefit = reduction in cost compared to no plowing

### local optimization
- perturbation methods: edge swap, vertex insert/remove, 2-opt
- accept improvements (hill climbing)
- converge when no improvement for 5 iterations

## computational complexity

- **graph construction**: o(n) where n = number of roads
- **partitioning**: o(n log n) for spectral decomposition
- **cycle generation**: o(m × k) where m = edges, k = cycles per partition
- **evaluation**: o(total cycles × avg cycle length)
- **optimization**: o(cycles × iterations × edges per cycle)

## future improvements

1. multi-objective optimization (cost, equity, robustness)
2. dynamic replanning based on real-time conditions
3. stochastic snowfall models
4. driver fatigue and break scheduling
5. salt supply constraints
6. integration with real-time traffic data
7. machine learning for cycle quality prediction

## references

- cornell university ithaca campus data
- ithaca road network shapefile
- cugir 2ft contour data for elevation
- ithaca dpw snow removal policies

## contact

for questions or issues, please refer to the optimization_report.md for detailed results.

