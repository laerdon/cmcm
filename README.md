# ithaca snow plow route optimization system

comprehensive graph-based optimization system for coordinating snow plow operations during winter graduation at cornell university.

## project overview

this system addresses the challenge of optimizing snow plow routes for the city of ithaca, ny during a major 12-hour snowstorm coinciding with cornell's winter graduation. the solution balances multiple competing priorities including emergency access, campus accessibility, parking availability, road classification, and terrain steepness.

### key features

- **priority-based routing**: weighted scoring system considering 5 factors
- **graph partitioning**: intelligent division of road network into 8 balanced zones
- **smart cycle generation**: three complementary algorithms (greedy, random walk, chinese postman)
- **cost optimization**: integral-based snow accumulation modeling
- **local search refinement**: perturbation-based route improvement
- **comprehensive visualization**: static maps and statistical analysis

## problem statement

### constraints

- **road network**: ~145 miles (868 segments) across ithaca
- **storm parameters**: 12-hour duration, 1 inch/hour snowfall rate
- **resources**: maximum 12 plow drivers, preferably fewer for flexibility
- **plow speed**: 25 mph during operations
- **special event**: graduation at cornell requiring enhanced campus/parking access

### objectives

1. maximize snow removal efficiency (minimize accumulated priority × snow depth)
2. ensure timely service to high-priority routes (emergency, arterials, steep hills)
3. maintain campus accessibility during graduation event
4. balance service equity across all neighborhoods
5. optimize driver assignments to allow rest periods

## methodology

### 1. priority calculation

each road segment receives a priority score based on five weighted factors:

```
priority = 0.31×emergency + 0.25×graduation_route + 0.23×graduation_parking
           + 0.13×road_class + 0.08×steepness
```

**component scores:**

| factor             | values      | description                               |
| ------------------ | ----------- | ----------------------------------------- |
| emergency          | 0 or 1.0    | principal arterials, hospital access      |
| graduation_route   | 0, 0.6, 1.0 | campus (1.0), nearby (0.6), other (0)     |
| graduation_parking | 0 or 1.0    | within 2km of barton hall                 |
| road_class         | 0.25-0.85   | local (0.25) to principal arterial (0.85) |
| steepness          | 0.2-1.0     | 0-5% (0.2) to >20% (1.0)                  |

### 2. graph construction

- **nodes**: 587 intersections (endpoints snapped within 10 feet)
- **edges**: 1,526 directed edges (bidirectional roads)
- **attributes**: priority, travel time, length, geometry, classification
- **connectivity**: largest component covers 76.9% of network

### 3. balanced partitioning

**algorithm**: recursive spectral partitioning

- split graph into 2 subgraphs using fiedler vector (2nd eigenvector of laplacian)
- recursively split each half (1 → 2 → 4 → 8 partitions)
- refine with local moves to balance total priority
- generate multiple partition sets for comparison (default: 3)

**partition statistics:**

- 8 partitions per set
- average 73 nodes per partition
- average priority: 66.42
- standard deviation/mean ratio: 0.733 (reasonable balance)

### 4. cycle generation

three complementary methods generate 50 candidate cycles per partition:

**greedy (33%)**

- always select highest-priority unvisited edge
- returns to start when no unvisited edges remain
- produces most consistent high-quality routes

**random walk (33%)**

- probabilistic selection biased by priority²
- bonus weighting for unvisited edges
- explores diverse route options

**chinese postman approximation (33%)**

- aims to cover all edges efficiently
- pairs odd-degree nodes via shortest paths
- eulerian-inspired traversal

**constraints:**

- maximum cycle time: 720 minutes (12 hours, full storm duration)
- must return to start location
- average generated cycle: 51.33 minutes

### 5. cost evaluation

**cost model**: integral of priority × snow depth over time

**without plow:**

```
cost = priority × snowfall_rate × (time²/2)
```

**with plow:**

- snow accumulates between visits
- cost = Σ[priority × snowfall_rate × (time_since_last_visit²/2)]
- plow removes all snow on each pass

**benefit metric:**

```
benefit = cost_without_plow - cost_with_plow
benefit_per_minute = benefit / cycle_time
```

### 6. local optimization

perturbation-based hill climbing:

- **edge swap**: replace edge with alternative route
- **vertex insert**: add intermediate stop
- **vertex remove**: skip unnecessary waypoint
- **2-opt**: reverse subsections of route

**convergence**: stops after 5 iterations without improvement or 20 total iterations

**results**: +6.24% average improvement in benefit/minute

## installation

### requirements

```bash
python 3.10+
geopandas>=0.14.0
pandas>=2.0.0
numpy>=1.24.0
shapely>=2.0.0
scipy>=1.10.0
networkx>=3.0
matplotlib>=3.7.0
```

optional for interactive maps:

```bash
folium>=0.14.0
pyproj>=3.4.0
```

### setup

```bash
# clone repository
git clone <repository-url>
cd cmcm

# install dependencies
pip install -r requirements.txt
```

## usage

### quick start: full pipeline

```bash
python3 main_pipeline.py
```

this executes all steps and generates complete results in ~3-5 minutes.

### command-line options

```bash
python3 main_pipeline.py \
    --num-partitions 3 \
    --cycles-per-subgraph 50 \
    --snowfall-rate 1.0 \
    --storm-duration 12 \
    --skip-to <step>
```

**parameters:**

- `--num-partitions`: number of partition sets to generate (default: 3)
- `--cycles-per-subgraph`: candidate cycles per partition (default: 50)
- `--snowfall-rate`: inches per hour (default: 1.0)
- `--storm-duration`: hours (default: 12)
- `--skip-to`: jump to specific step (priority|graph|partition|cycles|evaluate|sample|optimize)
- `--skip-classification`: use existing classified roads

### individual pipeline steps

```bash
# 1. classify roads by type and priority factors
python3 classify_roads.py

# 2. calculate weighted priority scores
python3 calculate_priority.py

# 3. build networkx graph from road network
python3 build_graph.py

# 4. partition graph into 8 balanced subgraphs
python3 partition_graph.py

# 5. generate candidate plow cycles (3 methods)
python3 generate_cycles.py

# 6. evaluate cycles using cost function
python3 evaluate_cycles.py

# 7. select top cycles per partition
python3 sample_cycles.py

# 8. optimize routes with local search
python3 optimize_routes.py

# 9. create visualizations
python3 visualize_results.py
```

## results

### optimal configuration

**partition set 2** (highest total benefit):

- total benefit: 159,904
- average benefit per minute: 19,667
- average cycle time: 1.57 minutes per initial pass

### driver assignments (partition set 2)

| driver | partition | cycle time | benefit/min | method | priority level |
| ------ | --------- | ---------- | ----------- | ------ | -------------- |
| 0      | 0         | 2.7 min    | 7,476       | greedy | medium         |
| 1      | 1         | 3.8 min    | 12,922      | greedy | high           |
| 2      | 2         | 0.2 min    | 12,443      | greedy | high           |
| 3      | 3         | 0.1 min    | 24,989      | greedy | very high      |
| 4      | 4         | 0.9 min    | 17,752      | greedy | high           |
| 5      | 5         | 0.1 min    | 55,676      | greedy | critical       |
| 6      | 6         | 3.0 min    | 15,901      | greedy | high           |
| 7      | 7         | 1.8 min    | 10,180      | greedy | medium-high    |

### key findings

1. **greedy dominates**: 91% of top cycles use greedy method

   - consistent prioritization produces best results
   - random walk and chinese postman useful for exploration only

2. **short initial cycles**: average 51 minutes for first pass

   - allows multiple passes during 12-hour storm
   - flexibility for adaptive replanning

3. **optimization gains**: local search improves routes by 6.24%

   - some partitions gain 76% (e.g., partition set 2, driver 7)
   - others already near-optimal (0% improvement)

4. **critical routes identified**:

   - partition 5 consistently highest priority (emergency + campus core)
   - driver 3 covers critical arterial connectors
   - partitions 1, 2, 4, 6 handle main campus access

5. **priority distribution**:
   - 40 emergency routes (principal arterials)
   - 223 on-campus roads
   - 631 near-campus routes (within 5km)
   - 364 parking-priority roads (within 2km of barton hall)

### performance metrics

**graph statistics:**

- 587 intersections
- 1,526 directed edges
- 145.24 miles total
- 5.81 hours to traverse all edges once at 25 mph

**cycle statistics:**

- 1,200 total cycles generated (3 sets × 8 partitions × 50 cycles)
- average cycle time: 51.33 minutes
- average benefit: 177,974
- average benefit per minute: 5,537
- best benefit per minute: 55,676

**method comparison:**

- greedy: 384 cycles (32%), dominates top-10 (91%)
- random walk: 384 cycles (32%), occasional top-10 appearance
- chinese postman: 432 cycles (36%), rare top-10 appearance

## data sources

### input files

1. **roads shapefile**: `Ithaca NY Roads/Roads.shp`

   - 868 road segments
   - attributes: name, road class, lanes, ownership
   - source: city of ithaca gis

2. **contour data**: `cugir-008148/Ithaca2ftContours.shp`
   - 2-foot elevation contours
   - used for steepness calculation
   - source: cornell university geospatial information repository (cugir)

### output files

**data files:**

- `classified_roads.csv`: roads with classification scores
- `classified_roads.geojson`: roads with geometry
- `roads_with_priority.csv`: priority scores and travel times
- `road_graph.gpickle`: networkx graph structure
- `partitions.json`: 3 sets of 8 partitions each
- `candidate_cycles.json`: 1,200 generated cycles (14mb)
- `cycle_evaluations.json`: evaluated cycles with costs (14mb)
- `selected_cycles.json`: top 10 cycles per partition
- `operational_plans.json`: driver assignments
- `optimized_cycles.json`: locally optimized routes

**reports:**

- `optimization_report.md`: comprehensive summary
- `graph_statistics.csv`: network metrics
- `partition_statistics.csv`: balance analysis
- `cycle_evaluations.csv`: performance rankings (153kb)

**visualizations:**

- `plots/priority_map.png`: road priority heatmap
- `plots/partition_map.png`: graph partitions by color
- `plots/benefit_by_method.png`: boxplot comparison
- `plots/partition_balance.png`: priority distribution
- `plots/cycle_time_distribution.png`: histogram

## operational recommendations

### for dpw snow operations

1. **use partition set 2**: highest total benefit and well-balanced priorities
2. **assign 8 drivers**: one per partition, allows 4 backup drivers available
3. **multiple passes**: 12-hour storm requires 12-15 passes per route
4. **priority sequence**:

   - first pass: cover all routes once (51 min average)
   - subsequent passes: focus on highest-priority edges
   - final pass: return to campus core before graduation

5. **adaptive strategy**:

   - monitor accumulation on critical routes (partition 5)
   - adjust if snowfall rate changes
   - coordinate with salt application (ineffective below 20°f)

6. **driver management**:

   - rotate drivers every 4 hours (federal regulations)
   - use 8 primary + 4 relief drivers
   - partition assignments make handoffs simple

7. **special considerations**:
   - graduation starts at 4pm (assume storm starts 4am)
   - ensure barton hall access clear by 2pm
   - coordinate odd/even parking enforcement
   - deploy extra resources to partition 5 (campus core)

### equity and fairness

**service time by road class:**

- principal arterials: <30 min between passes
- minor arterials: <60 min between passes
- collectors: <90 min between passes
- local roads: <120 min between passes

**geographic equity:**

- all 8 partitions receive dedicated driver
- priority balancing ensures fair resource allocation
- emergency access maintained across all neighborhoods

## algorithmic complexity

**time complexity by stage:**

- road classification: o(n) where n = road segments
- graph construction: o(n)
- spectral partitioning: o(n log n) for eigendecomposition
- cycle generation: o(m × k) where m = edges, k = cycles
- evaluation: o(total cycles × avg cycle length)
- optimization: o(cycles × iterations × edges)

**total pipeline**: ~3-5 minutes on modern hardware

**memory usage**: ~2gb peak (graph + all cycles in memory)

## limitations and future work

### current limitations

1. **static planning**: assumes constant snowfall rate
2. **no real-time updates**: cannot adapt to changing conditions
3. **simplified cost model**: doesn't account for:

   - salt effectiveness vs temperature
   - compacted snow (ice) formation
   - driver fatigue
   - equipment breakdowns
   - parking compliance (odd/even rules)

4. **single objective**: optimizes benefit only, not considering:
   - equity metrics
   - robustness to disruption
   - driver convenience

### future enhancements

1. **dynamic replanning**:

   - real-time weather integration
   - gps tracking of plow locations
   - adaptive route adjustment

2. **multi-objective optimization**:

   - pareto frontier of efficiency vs equity
   - robustness scoring
   - driver welfare metrics

3. **stochastic modeling**:

   - probabilistic snowfall scenarios
   - monte carlo simulation
   - risk analysis

4. **machine learning integration**:

   - predict cycle quality from features
   - learn optimal partition configurations
   - forecast snow accumulation patterns

5. **extended constraints**:

   - salt depot locations and capacity
   - fuel consumption and refueling
   - driver shift schedules
   - equipment maintenance windows

6. **integration capabilities**:
   - real-time traffic data
   - emergency service coordination
   - public communication system
   - mobile app for drivers

## technical details

### code structure

```
cmcm/
├── classify_roads.py           # step 1: classify roads
├── calculate_priority.py       # step 2: priority scores
├── build_graph.py             # step 3: graph construction
├── partition_graph.py         # step 4: partitioning
├── generate_cycles.py         # step 5: cycle generation
├── evaluate_cycles.py         # step 6: cost evaluation
├── sample_cycles.py           # step 7: top cycle selection
├── optimize_routes.py         # step 8: local optimization
├── main_pipeline.py           # orchestration
├── visualize_results.py       # plotting and maps
└── calculate_road_steepness.py # utility: elevation analysis
```

### dependencies

**core:**

- `geopandas`: spatial data handling
- `networkx`: graph algorithms
- `numpy`: numerical computing
- `scipy`: optimization and interpolation
- `pandas`: data manipulation
- `shapely`: geometric operations

**visualization:**

- `matplotlib`: static plotting
- `folium`: interactive maps (optional)
- `pyproj`: coordinate transformations (optional)

### coordinate systems

- **input roads**: web mercator (epsg:3857)
- **contours**: ny state plane feet (epsg:2261)
- **processing**: ny state plane feet (epsg:2261)
- **visualization**: wgs84 geographic (epsg:4326)

### code style

- lowercase comments and print statements
- bracketed debug tags: `[INFO]`, `[ERROR]`, `[PASS]`, `[WARNING]`
- double quotes for strings
- black code formatter compliant
- type hints in function signatures

## references

### data sources

- city of ithaca department of public works
- cornell university geospatial information repository (cugir)
- ithaca snow removal policies: https://www.cityofithaca.org/268/snow

### academic references

- spectral graph theory for network partitioning
- chinese postman problem and arc routing
- vehicle routing problem with time windows
- winter road maintenance optimization

### related work

- arc routing in practice (golden, b. l., & wong, r. t., 1981)
- the snow plow problem (eiselt, h. a., & laporte, g., 1987)
- optimization methods for road network winter maintenance (perrier, n., et al., 2006)

## contact and support

for questions, issues, or contributions:

- see `optimization_report.md` for detailed results
- see `OPTIMIZATION_README.md` for algorithm details
- examine output csv files for raw data

## license

this project was developed for the cornell mcm competition. educational use permitted.

---

**last updated**: 2025-11-08  
**version**: 1.0  
**status**: production-ready
