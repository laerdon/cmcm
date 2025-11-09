# cycle evaluation changes

## key improvements

### 1. full 12-hour storm rollout
**before**: only simulated one pass through each cycle  
**after**: simulates repeated passes throughout the entire 12-hour storm

this automatically penalizes routes that don't cover much ground because:
- short cycles on few edges → lower total benefit (fewer edges plowed)
- long cycles that can't repeat → accumulate cost on unvisited edges

### 2. new metrics tracked
- `num_complete_cycles`: how many times the cycle completes in 12 hours
- `coverage_ratio`: fraction of unique edges visited (currently 100% since cycle visits all its edges)
- `benefit_per_minute`: now calculated per minute of storm duration, not cycle time

### 3. how it works

#### cost without plow:
for each edge in the cycle, snow accumulates linearly for full 12 hours:
```
cost = priority × snowfall_rate × (storm_duration²) / 2
```

#### cost with plow (new):
1. plow repeats the cycle as many times as possible in 12 hours
2. for each edge visit, accumulate cost since last visit
3. after all cycles complete, add residual cost for time from last visit to storm end

#### benefit calculation:
```
benefit = cost_without_plow - cost_with_plow
benefit_per_minute = benefit / storm_duration
```

### 4. what this means for route selection

**good routes** (high benefit):
- cover many high-priority edges
- complete fast enough to repeat multiple times
- balance coverage with efficiency

**penalized routes**:
- only cover a few edges (low total benefit)
- take too long (fewer repetitions, more residual cost)
- skip high-priority roads

## example scenario

imagine two routes in a 720-minute (12-hour) storm:

**route a**: 30 minutes, covers 10 high-priority edges
- completes 24 times
- each edge gets plowed every 30 minutes
- low residual cost

**route b**: 90 minutes, covers 30 medium-priority edges  
- completes 8 times
- each edge gets plowed every 90 minutes
- higher residual cost

the evaluation now properly accounts for the repeated passes and will favor whichever provides more total benefit over the full storm duration.

