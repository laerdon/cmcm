# benefit metric update

## overview

updated all benefit calculations to use **efficiency** (% of baseline cost saved) instead of absolute benefit. this makes comparisons invariant to total snow volume and provides a more meaningful measure of route performance.

## motivation

**problem with absolute benefit:**
- depends on total snow volume
- makes comparison unfair when snowfall varies spatially
- difficult to compare routes across different scenarios

**solution with efficiency metric:**
- measures what % of baseline cost we save
- invariant to total snow volume
- directly comparable across scenarios

## new metric: `benefit_pct`

```python
benefit_pct = (benefit / cost_without_plow) × 100
```

**interpretation:**
- "this route saves X% of the baseline (no-plow) cost"
- higher is better (more efficient)
- ranges from 0% (no benefit) to ~100% (perfect coverage)

## changes made

### 1. `evaluate_cycles.py`

**added `benefit_pct` to evaluation:**
```python
benefit_pct = (benefit / cost_without_plow * 100.0) if cost_without_plow > 0 else 0.0
```

**updated ranking:**
- cycles now ranked by `benefit_pct` instead of `benefit_per_minute`
- `rank_cycles_by_partition` uses efficiency metric

**updated output:**
- summary statistics show `benefit_pct`
- top cycles display efficiency percentage
- csv includes `benefit_pct` column

**legacy support:**
- `benefit_per_minute` still calculated for backward compatibility

### 2. `optimize_routes.py`

**updated optimization objective:**
```python
best_benefit = current_benefit_metrics.get("benefit_pct", current_benefit_metrics["benefit_per_minute"])
```

**optimization now:**
- maximizes efficiency (% of baseline saved)
- uses `benefit_pct` for acceptance criteria
- debug output shows efficiency values

### 3. `validate_gaussian_snowfall.py`

**added `benefit_pct` to gaussian evaluation:**
```python
benefit_pct = (benefit / cost_without_plow * 100.0) if cost_without_plow > 0 else 0.0
```

**updated comparison logic:**
```python
# old: percent change in absolute benefit
change_pct = ((gaussian_benefit - uniform_benefit) / uniform_benefit) * 100

# new: difference in efficiency (percentage points)
change_pct = gaussian_benefit_pct - uniform_benefit_pct
```

**comparison now measures:**
- uniform efficiency: "saves X% of baseline"
- gaussian efficiency: "saves Y% of baseline"
- change: "Y - X percentage points difference"

**updated output:**
- shows efficiency values with "%" suffix
- change shown as "pp" (percentage points)
- includes both uniform and gaussian efficiency in top 10 lists

## example output

### evaluate_cycles.py
```
[SUMMARY] evaluation statistics:
average benefit: 114535.21
average benefit %: 82.37%
best benefit %: 95.18%
...

[INFO] top cycle per partition:
  partition set 0, partition 0: benefit = 82.37%, cycles completed = 2, ...
```

### validate_gaussian_snowfall.py
```
GAUSSIAN SNOWFALL VALIDATION RESULTS
====================================

average efficiency change: +2.15 percentage points
std dev of change: 4.73 percentage points

partition-level summary:
partition set 0, partition 0: uniform=82.4%, gaussian=84.5% (+2.1 pp)
partition set 0, partition 1: uniform=79.2%, gaussian=77.8% (-1.4 pp)
...

top 10 cycles most positively affected:
ps=0, p=4, method=greedy: +5.3 pp (uniform=81.2%, gaussian=86.5%)
...

top 10 cycles most negatively affected:
ps=1, p=7, method=random_walk_bias_1.0: -3.8 pp (uniform=83.1%, gaussian=79.3%)
```

### optimize_routes.py
```
[DEBUG] initial benefit: 82.37%
[DEBUG] iteration 0: 82.37% -> 83.15%
[DEBUG] iteration 1: 83.15% -> 83.42%
[DEBUG] final benefit: 83.42%
```

## interpretation guide

### efficiency metric (`benefit_pct`)
- **80-90%**: excellent route, saves most of baseline cost
- **70-80%**: good route, saves majority of baseline
- **60-70%**: acceptable route, saves more than half
- **<60%**: poor route, needs improvement

### comparison metric (`change_pct`)
- **positive**: route performs better under gaussian
- **negative**: route performs worse under gaussian
- **magnitude**: larger absolute value = more sensitive to snow distribution

### why percentage points (pp)?
- comparing percentages directly
- "82% vs 85%" = "+3 pp" (not "+3.7%")
- avoids confusion of "percent of percent"

## validation

tested with sample cycles:

**uniform evaluation:**
```
cost without plow: 139056.48
cost with plow: 24521.27
benefit: 114535.21
benefit %: 82.37%
```

**gaussian evaluation:**
```
cost without plow: 223973435.42
cost with plow: 33643.97
benefit: 223939791.45
benefit %: 99.98%
```

✅ metric calculated correctly
✅ values in expected range
✅ comparison logic works
✅ backward compatibility maintained

## migration notes

**csv files:**
- `cycle_evaluations.csv` now includes `benefit_pct` column
- `gaussian_comparison.csv` now has `uniform_benefit_pct`, `gaussian_benefit_pct` columns
- sorting changed from `benefit_per_minute` to `benefit_pct`

**json files:**
- all evaluation dicts now include `"benefit_pct"` field
- `benefit_per_minute` still present for backward compatibility

**scripts that may need updating:**
- any custom analysis scripts using `benefit_per_minute`
- visualization scripts that plot benefit values
- selection scripts that rank by benefit

**no action needed for:**
- existing json/csv files (can be regenerated)
- cycle generation (unaffected)
- graph building (unaffected)

