# snow plow optimization system - summary report
generated: 2025-11-07 23:10:09.856249

## parameters
- partition sets: 3
- cycles per subgraph: 50
- snowfall rate: 1.0 inches/hour
- storm duration: 12 hours

## graph statistics
- num_nodes: 587
- num_edges: 1526
- num_components: 1
- total_priority: 558.3490000000022
- total_length_feet: 766850.32 feet (145.24 miles)
- total_travel_time_minutes: 348.57 minutes (5.81 hours)

## partition statistics
- total partitions: 24
- partition sets: 3
- partitions per set: 8
- avg nodes per partition: 73.4
- avg priority per partition: 66.42

## cycle generation and evaluation
- total cycles generated: 1200
- avg cycle time: 51.33 minutes
- avg benefit: 177973.74
- avg benefit per minute: 5537.43
- best benefit per minute: 55675.99

### method distribution
- chinese_postman: 432 (36.0%)
- greedy: 384 (32.0%)
- random_walk: 384 (32.0%)

## operational plans

### partition set 0
- drivers: 8
- total benefit: 141392.53
- avg cycle time: 1.43 minutes
- avg benefit per minute: 18006.02

#### driver assignments
- driver 0: partition 0, time=2.7min, benefit/min=7476.16, method=greedy
- driver 1: partition 1, time=0.1min, benefit/min=16370.57, method=greedy
- driver 2: partition 2, time=0.5min, benefit/min=9634.18, method=greedy
- driver 3: partition 3, time=0.2min, benefit/min=12793.97, method=greedy
- driver 4: partition 4, time=4.4min, benefit/min=14319.49, method=greedy
- driver 5: partition 5, time=0.1min, benefit/min=55675.99, method=greedy
- driver 6: partition 6, time=0.8min, benefit/min=16090.32, method=greedy
- driver 7: partition 7, time=2.7min, benefit/min=11687.48, method=greedy


### partition set 1
- drivers: 8
- total benefit: 78740.47
- avg cycle time: 0.72 minutes
- avg benefit per minute: 19910.56

#### driver assignments
- driver 0: partition 0, time=0.6min, benefit/min=9043.77, method=greedy
- driver 1: partition 1, time=0.1min, benefit/min=16370.57, method=greedy
- driver 2: partition 2, time=0.3min, benefit/min=8403.27, method=greedy
- driver 3: partition 3, time=0.1min, benefit/min=24988.79, method=greedy
- driver 4: partition 4, time=1.7min, benefit/min=15935.95, method=greedy
- driver 5: partition 5, time=0.1min, benefit/min=55675.99, method=greedy
- driver 6: partition 6, time=0.2min, benefit/min=17178.68, method=greedy
- driver 7: partition 7, time=2.7min, benefit/min=11687.48, method=greedy


### partition set 2
- drivers: 8
- total benefit: 159904.02
- avg cycle time: 1.57 minutes
- avg benefit per minute: 19667.38

#### driver assignments
- driver 0: partition 0, time=2.7min, benefit/min=7476.17, method=greedy
- driver 1: partition 1, time=3.8min, benefit/min=12922.40, method=greedy
- driver 2: partition 2, time=0.2min, benefit/min=12443.23, method=greedy
- driver 3: partition 3, time=0.1min, benefit/min=24988.79, method=greedy
- driver 4: partition 4, time=0.9min, benefit/min=17751.62, method=greedy
- driver 5: partition 5, time=0.1min, benefit/min=55675.99, method=greedy
- driver 6: partition 6, time=3.0min, benefit/min=15900.58, method=greedy
- driver 7: partition 7, time=1.8min, benefit/min=10180.26, method=greedy

## optimization results
local optimization was applied using perturbation-based search.
perturbation methods: edge swap, vertex insert/remove, 2-opt style moves.

## recommendations
1. use the operational plan for the partition set with highest total benefit
2. drivers should start at designated start nodes in their assigned partitions
3. follow the optimized cycle routes to maximize benefit (priority * snow cleared)
4. plan for multiple passes during the 12-hour storm
5. monitor weather conditions and adjust if snowfall rate changes
6. coordinate salt usage based on temperature and priority routes
7. ensure driver breaks comply with federal regulations

