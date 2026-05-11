---
name: cost-functions
description: Use when modeling the real expense of moving between gpCAM measurement points — motor travel time, settling, directional costs, sample damage, beam time, or zone-based penalties.
---

# Skill: gpCAM Cost Functions

Design cost functions that account for the real expense of moving between measurement points — motor travel time, sample damage, beam time, etc.

## When to Use

- Moving motors is slow and travel time matters (long-range stage moves)
- Some regions of the parameter space are more expensive to reach
- You want to avoid unnecessary large jumps between measurements
- Cost varies by direction (e.g., moving up is faster than moving down)
- Sample damage accumulates with exposure and should be minimized

## How Cost Functions Work in gpCAM

The cost function modifies the acquisition score:

```
effective_score(x) = acquisition_score(x) / cost(origin, x)
```

Points that are expensive to reach get penalized. The optimizer still picks high-value points, but prefers nearby high-value points over distant ones.

## Cost Function Contract

```python
def my_cost(origin, x, arguments=None):
    """
    Parameters
    ----------
    origin : np.ndarray, shape (V, D)
        The current position (where we are now).
    x : np.ndarray, shape (V, D)
        The candidate destination positions.
    arguments : dict or None
        Optional extra parameters (passed via GPOptimizer constructor).
    
    Returns
    -------
    cost : np.ndarray, shape (V,)
        Cost of moving from origin to each point in x.
        Must be > 0. Higher cost = less desirable to visit.
    """
```

**Key rules:**
- Cost must be **positive** (> 0) — zero cost causes division by zero
- Cost is used as a **divisor** — higher cost = lower effective acquisition score
- The function receives batches of points, not single points

## Recipes

### L2 (Euclidean) Distance Cost
Simple travel time proportional to straight-line distance:
```python
def l2_cost(origin, x, arguments=None):
    """Cost proportional to Euclidean distance."""
    offset = 1.0   # minimum cost (prevents div-by-zero, represents measurement time)
    speed = 1.0     # cost per unit distance
    distance = np.linalg.norm(x - origin, axis=1)
    return offset + speed * distance
```

### L1 (Manhattan) Distance Cost
For stage systems that move one axis at a time:
```python
def l1_cost(origin, x, arguments=None):
    """Cost proportional to Manhattan distance (axis-by-axis motion)."""
    offset = 1.0
    speed = 1.0
    distance = np.sum(np.abs(x - origin), axis=1)
    return offset + speed * distance
```

### Anisotropic Cost
Different axes have different speeds:
```python
def anisotropic_cost(origin, x, arguments=None):
    """
    Different cost per axis.
    arguments["speeds"] = [speed_dim0, speed_dim1, ...]
    """
    offset = 1.0
    speeds = arguments.get("speeds", np.ones(x.shape[1]))
    weighted_dist = np.sum(np.abs(x - origin) * speeds, axis=1)
    return offset + weighted_dist
```

### Directional Cost
Moving in one direction is cheaper (e.g., gravity-assisted, or always-increasing scans):
```python
def directional_cost(origin, x, arguments=None):
    """Cheaper to move in +x direction than -x."""
    offset = 1.0
    diff = x - origin
    # Forward motion (positive) is cheap, backward is expensive
    forward_cost = np.maximum(diff[:, 0], 0) * 1.0   # cost going forward
    backward_cost = np.maximum(-diff[:, 0], 0) * 5.0  # 5x cost going backward
    lateral_cost = np.sum(np.abs(diff[:, 1:]), axis=1) * 1.0
    return offset + forward_cost + backward_cost + lateral_cost
```

### Settling Time Cost
Fast moves need more settling time:
```python
def settling_cost(origin, x, arguments=None):
    """
    Short moves are fast; long moves need extra settling time.
    Models: cost = base + travel + settling * (distance > threshold)
    """
    base = 1.0
    travel_rate = 0.5
    settle_time = 3.0
    settle_threshold = 2.0  # distance above which settling kicks in
    
    distance = np.linalg.norm(x - origin, axis=1)
    settling = np.where(distance > settle_threshold, settle_time, 0.0)
    return base + travel_rate * distance + settling
```

### Zone-Based Cost
Some regions of the parameter space are more expensive:
```python
def zone_cost(origin, x, arguments=None):
    """
    Higher cost to measure in certain zones.
    E.g., cryogenic sample region requires cooldown.
    """
    base = 1.0
    distance = np.linalg.norm(x - origin, axis=1)
    
    # Expensive zone: x[:, 0] > 8.0
    zone_penalty = np.where(x[:, 0] > 8.0, 10.0, 0.0)
    
    return base + distance + zone_penalty
```

## Usage

```python
gpo = GPOptimizer(
    x_data=x_data,
    y_data=y_data,
    cost_function=l2_cost,
    # args={"speeds": np.array([1.0, 2.0])}  # for anisotropic_cost
)
```

The cost function is automatically applied during `ask()` — no extra code needed in the experiment loop.

## Parameterized Cost with `arguments`

Pass parameters via the `args` dict at construction:

```python
gpo = GPOptimizer(
    x_data=x_data,
    y_data=y_data,
    cost_function=anisotropic_cost,
    args={"speeds": np.array([1.0, 3.0, 0.5])},
)
```

The `arguments` parameter in the cost function receives this dict.

## Cost Functions Do NOT Add Hyperparameters

Unlike kernel/mean/noise functions, cost functions have **fixed parameters** — they are not optimized during training. If you need to tune cost parameters, do it manually or via the `arguments` dict.

## Calibrating a Cost Function From Observed Moves

If you don't know the cost parameters ahead of time, record observed costs during the experiment and fit `offset` / `slope` by nonlinear least squares. gpCAM ships templates (`_update_cost_function`, `update_l1_cost_function`, `update_l2_cost_function`) that use SciPy's `differential_evolution`:

```python
# Collect observations as a list of dicts during the experiment:
observations = [
    {"origin": np.array([0.0, 0.0]), "point": np.array([0.5, 0.2]), "cost": 2.7},
    {"origin": np.array([0.5, 0.2]), "point": np.array([0.8, 0.9]), "cost": 3.4},
    # ...
]

# Then fit:
from gpcam.cost_functions import update_l2_cost_function   # or update_l1_cost_function
new_args = update_l2_cost_function(observations, parameters={"offset": 1.0, "slope": 1.0})
# Plug `new_args` back into the cost-function `arguments` dict.
```

The updater drops outliers beyond ±2σ of the per-move cost-per-distance before fitting, so a few anomalous moves (e.g. instrument glitches) won't distort the cost model.

## Common Pitfalls

1. **Zero cost**: Causes division by zero in acquisition. Always add a positive offset (minimum cost ≥ 1.0).
2. **Cost too high**: If cost dominates, the optimizer just measures nearby points and never explores. Balance cost magnitude with acquisition scores.
3. **Wrong shape**: Must return 1D array of length `V`, matching the number of candidate points.
4. **Forgetting `origin`**: The cost depends on where you currently are, not just where you're going.
5. **Not vectorized**: `origin` and `x` are batches — use numpy operations, not loops.
