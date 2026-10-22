---
name: cost-functions
description: Use when modeling the real expense of moving between gpCAM measurement points — motor travel time, settling, directional costs, sample damage, beam time, or zone-based penalties.
gpcam_version: "8.4.x"
fvgp_version: "4.8.x"
last_verified: "2026-07-23 (gpCAM dadeb65)"
---

# Skill: gpCAM Cost Functions

*Verified against gpCAM 8.4.x / fvgp 4.8.x — last checked 2026-07-23 (gpCAM `dadeb65`).*

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

gpCAM calls the cost function with **exactly two positional arguments**:

```python
def my_cost(origin, x):
    """
    Parameters
    ----------
    origin : np.ndarray, shape (D,) or (1, D)
        The current position (where we are now), as passed to ask(position=...).
    x : np.ndarray, shape (V, D)
        The candidate destination positions.
    
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
- The function receives a batch of candidate points, not a single point
- **The cost function does NOT receive the `args` dict.** Unlike kernel / prior mean /
  noise callables, there is no arity-based dispatch here — the call site is literally
  `cost_function(origin, x)`. A signature like `my_cost(origin, x, arguments=None)`
  will run without error but `arguments` is always `None`. To parameterize a cost
  function, use a closure or `functools.partial` (see below).
- **`cost_function` only takes effect when `origin` is not None**, i.e. when you call
  `ask(..., position=current_position)`. Without `position=`, the cost is ignored
  entirely and the acquisition is used unmodified. This is the single most common
  reason a cost function "does nothing".

## Recipes

### L2 (Euclidean) Distance Cost
Simple travel time proportional to straight-line distance:
```python
def l2_cost(origin, x):
    """Cost proportional to Euclidean distance."""
    offset = 1.0   # minimum cost (prevents div-by-zero, represents measurement time)
    speed = 1.0     # cost per unit distance
    distance = np.linalg.norm(x - origin, axis=1)
    return offset + speed * distance
```

### L1 (Manhattan) Distance Cost
For stage systems that move one axis at a time:
```python
def l1_cost(origin, x):
    """Cost proportional to Manhattan distance (axis-by-axis motion)."""
    offset = 1.0
    speed = 1.0
    distance = np.sum(np.abs(x - origin), axis=1)
    return offset + speed * distance
```

### Anisotropic Cost
Different axes have different speeds. Since the cost function receives no `args`,
bind the speeds with a factory:
```python
def make_anisotropic_cost(speeds, offset=1.0):
    """
    Returns a two-argument cost function with per-axis speeds captured.
    speeds : array of shape (D,) — cost per unit distance along each axis.
    """
    speeds = np.asarray(speeds, dtype=float)

    def anisotropic_cost(origin, x):
        return offset + np.sum(np.abs(x - origin) * speeds, axis=1)
    return anisotropic_cost
```

### Directional Cost
Moving in one direction is cheaper (e.g., gravity-assisted, or always-increasing scans):
```python
def directional_cost(origin, x):
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
def settling_cost(origin, x):
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
def zone_cost(origin, x):
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
    # cost_function=make_anisotropic_cost([1.0, 3.0, 0.5]),   # parameterized variant
)
```

**You must pass `position=` to `ask()` for the cost to be applied:**

```python
current_position = np.array([[0.0, 0.0]])       # where the instrument is now

for i in range(N_ITERATIONS):
    next_x = gpo.ask(
        input_set=parameter_bounds,
        position=current_position,               # REQUIRED — without it cost is ignored
        acquisition_function="variance",
    )["x"]
    y = measure(next_x)
    gpo.tell(next_x, y)
    current_position = next_x                    # advance the origin
```

Forgetting `position=` is the usual reason a cost function appears to have no effect:
gpCAM applies the cost only when `origin is not None`, and silently skips it otherwise.

## Parameterizing a Cost Function

The `args` dict passed to `GPOptimizer(..., args=...)` reaches kernel, prior mean, and
noise functions — **not** the cost function. Use a closure (as in
`make_anisotropic_cost` above) or `functools.partial`:

```python
from functools import partial

def scaled_l2_cost(origin, x, offset, speed):
    return offset + speed * np.linalg.norm(x - origin, axis=1)

gpo = GPOptimizer(x_data, y_data,
                  cost_function=partial(scaled_l2_cost, offset=1.0, speed=2.5))
```

`partial` of a module-level function pickles by reference, so this form survives
`pickle.dumps(gpo)` for checkpointing; a closure defined inside another function does
not.

## Cost Functions Do NOT Add Hyperparameters

Unlike kernel/mean/noise functions, cost function parameters are **fixed** — they are
not optimized during training. To tune them, refit them yourself and rebuild the
callable.

## Calibrating a Cost Function From Observed Moves

If you don't know the cost parameters ahead of time, record the real cost of each move
during the experiment and fit them offline. There is no gpCAM helper for this — write
the fit yourself; it is a few lines:

```python
import numpy as np
from scipy.optimize import curve_fit

# Record every move as you go:
observations = []   # (origin, destination, measured_cost) triples
# ... inside the loop: observations.append((current_position[0], next_x[0], elapsed_seconds))

origins = np.array([o for o, _, _ in observations])
dests = np.array([d for _, d, _ in observations])
costs = np.array([c for _, _, c in observations])
distances = np.linalg.norm(dests - origins, axis=1)

# Drop outliers beyond 2 sigma of cost-per-distance (instrument glitches)
rate = costs / np.maximum(distances, 1e-9)
keep = np.abs(rate - rate.mean()) <= 2.0 * rate.std()

(offset, speed), _ = curve_fit(lambda d, a, b: a + b * d,
                               distances[keep], costs[keep], p0=[1.0, 1.0])
print(f"offset={offset:.3f} s, speed={speed:.3f} s per unit distance")

gpo.cost_function = partial(scaled_l2_cost, offset=offset, speed=speed)
```

Fitting on measured seconds makes the cost units meaningful, which matters because the
acquisition is divided by it — see the balance pitfall below.

## Common Pitfalls

1. **Forgetting `position=` in `ask()`**: The most common failure. Without it, `origin` is `None` and gpCAM skips the cost function entirely — silently.
2. **Expecting `args` to reach the cost function**: It does not. Use a closure or `functools.partial`.
3. **Zero cost**: Causes division by zero in acquisition. Always add a positive offset (minimum cost ≥ 1.0).
4. **Cost too high**: The acquisition is *divided* by the cost, so if cost dominates, the optimizer measures only nearby points and never explores. Compare typical cost magnitudes against typical acquisition magnitudes (`gpo.evaluate_acquisition_function(x_grid)`) and rescale so neither swamps the other.
5. **Wrong shape**: Must return 1D array of length `V`, matching the number of candidate points.
6. **Not advancing `origin`**: Update the position you pass to `ask()` after every move, or the cost is computed from a stale location.
7. **Not vectorized**: `x` is a batch — use numpy operations, not loops.
