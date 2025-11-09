# gaussian snowfall validation

## overview

validation script that tests plow route performance under **spatially-varying snowfall** using a 2D gaussian distribution. this provides more realistic testing than uniform snowfall.

## what was implemented

### 1. gaussian snowfall model (`create_gaussian_snowfall`)

**generates spatially-varying snowfall rates:**

- center: defaults to graph centroid (configurable)
- sigma: auto-calculated as 25% of graph diameter (configurable)
- multiplier range: 0.1x to 2.0x base rate
- peak at center: 2.0 inches/hour
- edges at periphery: ~0.1-0.4 inches/hour

**key features:**

- uses edge midpoints for sampling
- gaussian formula: `exp(-distance² / (2σ²))`
- smooth spatial variation across ithaca

### 2. variable snow cost calculation (`calculate_cost_with_variable_snow`)

**modified evaluation logic:**

- each edge uses its own snowfall rate
- otherwise identical to uniform evaluation
- tracks repeated passes over 12-hour storm
- accounts for residual accumulation

**key difference:**

```python
# uniform: same rate for all edges
cost = priority × snowfall_rate × (time²/2)

# gaussian: edge-specific rate
cost = priority × edge_snowfall_rate[edge_id] × (time²/2)
```

### 3. cycle evaluation (`evaluate_cycles_gaussian`)

**evaluates all cycles under gaussian conditions:**

- generates gaussian snowfall map
- calculates benefit for each cycle
- returns same metrics as uniform evaluation
- preserves gaussian parameters for reproducibility

**metrics tracked:**

- cost_without_plow
- cost_with_plow
- benefit & benefit_per_minute
- num_complete_cycles
- coverage_ratio

### 4. comparison analysis (`compare_uniform_vs_gaussian`)

**compares performance under both conditions:**

- calculates % change for each cycle
- partition-level aggregation
- identifies most/least affected routes
- statistical summary

**output includes:**

- average benefit change across all cycles
- per-partition comparison
- top 10 positively affected cycles
- top 10 negatively affected cycles

### 5. visualization (`visualize_gaussian_snowfall`)

**creates spatial map showing:**

- edges colored by snowfall intensity
- gaussian center marked with star
- contour lines at 0.1, 0.3, 0.5, 0.7, 0.9
- colorbar indicating snowfall rate

**file:** `gaussian_snowfall_map.png`

## how to use

### basic usage:

```bash
python validate_gaussian_snowfall.py
```

### what it does:

1. loads road graph and uniform evaluation results
2. creates gaussian snowfall distribution
3. re-evaluates all cycles under gaussian conditions
4. compares uniform vs gaussian performance
5. generates visualization

### output files:

- `cycle_evaluations_gaussian.json` - full gaussian evaluation results
- `gaussian_comparison.csv` - comparison dataframe
- `gaussian_snowfall_map.png` - spatial visualization

## key insights from design

### why this matters

**uniform snowfall assumption is unrealistic:**

- actual snowstorms have spatial variation
- lake effect, terrain, wind patterns create gradients
- routes optimized for uniform may fail under realistic conditions

**gaussian model tests route robustness:**

- routes covering high-snow zones get more benefit
- routes avoiding high-snow zones get penalized
- reveals which partitioning strategies are spatially sensitive

### expected behavior

**cycles that improve under gaussian:**

- routes concentrated in high-snow zone (near center)
- compact routes that stay in favorable areas
- partitions that happen to align with snow distribution

**cycles that worsen under gaussian:**

- routes far from gaussian center
- routes spanning high and low snow zones
- long connector paths through low-snow areas

## example output

```
GAUSSIAN SNOWFALL VALIDATION RESULTS
=====================================

average benefit change: -3.2%
std dev of change: 8.7%

partition-level summary:
partition set 0, partition 0: uniform=182.5, gaussian=165.3 (-9.4%)
partition set 0, partition 1: uniform=325.1, gaussian=298.7 (-8.1%)
partition set 0, partition 4: uniform=893.8, gaussian=945.2 (+5.8%)
...

top 10 cycles most negatively affected:
ps=0, p=7, method=greedy: -18.3%
...

top 10 cycles most positively affected:
ps=1, p=4, method=random_walk_bias_2.0: +12.5%
...
```

## technical details

### coordinate system

- uses state plane feet (same as graph coordinates)
- graph bounds: ~9000 ft x ~5500 ft
- sigma: ~2250 ft (covers ~25% of ithaca)

### gaussian parameters

- **default center**: graph centroid
- **default sigma**: 0.25 × graph diameter
- **multiplier range**: [0.1, 2.0]
- **base rate**: 1.0 inches/hour

### performance

- evaluates 1200 cycles in ~5-10 minutes
- comparable to uniform evaluation
- most time spent in shortest path calculations

## validation checks

✅ gaussian integrates reasonably (total snow volume comparable)
✅ edge rates vary smoothly across space
✅ cost calculations preserve cycle closure
✅ results reproducible with same parameters
✅ no linter errors
✅ handles all cycle types (greedy, random walk)

## future enhancements

potential improvements:

- **moving gaussian**: storm tracks across ithaca over time
- **multiple centers**: simulate multiple storm cells
- **temporal variation**: intensity changes during 12-hour period
- **anisotropic gaussian**: elliptical patterns for wind effects
- **real weather data**: use historical snowfall patterns
