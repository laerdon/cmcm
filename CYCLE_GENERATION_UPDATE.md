# cycle generation update

## changes made

### 1. removed chinese postman algorithm
**reason**: the implementation was broken and didn't properly handle non-eulerian graphs

**issues with chinese postman**:
- computed edge duplications but never used them
- ran hierholzer's algorithm on original graph, not augmented eulerian graph
- only covered partial edges in non-eulerian graphs
- theoretical optimality doesn't match our repeated-pass cost model

### 2. replaced with multiple priority bias values

**new approach**: test three different priority bias values for random walk

**bias values tested**:
- `1.5` - **low bias**: more exploration, less priority-focused
- `2.0` - **medium bias**: balanced approach (original default)
- `3.0` - **high bias**: strongly favors high-priority edges

**method names**:
- `greedy` - deterministic highest-priority-first approach
- `random_walk_bias_1.5` - low priority bias
- `random_walk_bias_2.0` - medium priority bias
- `random_walk_bias_3.0` - high priority bias

### 3. updated cycle distribution

**before** (50 cycles per partition):
- 16 greedy
- 16 random walk (bias=2.0)
- 18 chinese postman (broken)

**after** (50 cycles per partition):
- 12 greedy
- 12 random walk (bias=1.5)
- 12 random walk (bias=2.0)
- 12 random walk (bias=3.0)
- **total: 48 cycles** (2 fewer due to division by 4)

### 4. increased max cycle time

**before**: 120 minutes (2 hours)  
**after**: 360 minutes (6 hours)

**rationale**: 
- allows complete coverage of larger partitions
- some partitions have 372+ edges requiring longer cycles
- still fits within 12-hour storm with multiple repetitions

## bias parameter explanation

the priority bias controls how strongly the random walk favors high-priority edges:

```python
# edge selection probability
probability ∝ priority^bias

# examples for priority=10 vs priority=1:
bias=1.0: 10:1 ratio (10x more likely)
bias=1.5: 31.6:1 ratio
bias=2.0: 100:1 ratio
bias=3.0: 1000:1 ratio
```

**low bias (1.5)**:
- more uniform exploration
- better coverage of low-priority edges
- useful for ensuring complete street coverage

**high bias (3.0)**:
- strongly focuses on high-priority roads
- may skip low-priority areas
- good for limited-time scenarios

## expected outcomes

testing multiple bias values will help identify:
1. **optimal exploration-exploitation tradeoff** for repeated cycles
2. whether **high-priority focus** (high bias) or **complete coverage** (low bias) performs better
3. how bias affects benefit per minute in 12-hour rollout evaluation

## next steps

1. run `python generate_cycles.py` to generate new candidate cycles
2. run `python evaluate_cycles.py` to evaluate with 12-hour rollout
3. compare performance across bias values
4. select best approach for final route optimization

