---
name: cost-functions
description: Use when modeling the real expense of moving between gpCAM measurement points — motor travel time, settling, directional costs, sample damage, beam time, or zone-based penalties.
gpcam_version: "8.7.x"
fvgp_version: "4.11.x"
last_verified: "2026-10-28 (gpCAM 8.7.0, a21385a)"
---

# Skill: gpCAM Cost Functions

*Verified against gpCAM 8.7.0 / fvgp 4.11.1 — last checked 2026-10-28 (gpCAM a21385a).*

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
- Cost must be **strictly positive** (`> 0`) and return a 1-D array of shape `(V,)` — it is used as a divisor (`surrogate_model.py:265`), so zero cost causes division by zero. Always include a positive base offset (e.g., measurement overhead time).
- **Pass `position=current_position` to `ask()`:** gpCAM only evaluates `cost_function(origin, x)` when `origin is not None` (`surrogate_model.py:264`). Omitting `position=` in `ask()` silently disables the cost function.
- **No `args` dict or trainable hyperparameters:** `cost_function` is always called with 2 arguments `cost_function(origin, x)` and is not optimized during `train()`. Parameterize it with `functools.partial` (which survives `pickle.dumps(gpo)` for checkpointing; see the `experiment-designer` skill for the full `args` dispatch table).

## Recipes

### Euclidean / Anisotropic Travel Cost (with `functools.partial`)
```python
from functools import partial
import numpy as np

def travel_cost(origin, x, offset=1.0, weights=1.0):
    """
    Base overhead `offset` (> 0) plus weighted Euclidean travel time.
    Pass `weights` as a scalar or shape-(D,) per-axis array via functools.partial.
    """
    diff = (x - origin) * np.asarray(weights, dtype=float)
    return offset + np.linalg.norm(diff, axis=1)

my_cost = partial(travel_cost, offset=1.0, weights=[1.0, 3.0])
```
For axis-by-axis (Manhattan) stages, replace `np.linalg.norm(diff, axis=1)` with `np.sum(np.abs(diff), axis=1)`.

### Directional, Settling, or Zone-Based Cost
Combine asymmetric step costs, step-function settling times, or expensive-region penalties into the same positive 1-D return vector:
```python
def stage_cost(origin, x):
    """Asymmetric axis-0 motion + long-move settling time + zone penalty."""
    base = 1.0
    diff = x - origin
    axis0_cost = np.maximum(diff[:, 0], 0.0) * 1.0 + np.maximum(-diff[:, 0], 0.0) * 5.0
    settling = np.where(np.linalg.norm(diff, axis=1) > 2.0, 3.0, 0.0)
    zone_penalty = np.where(x[:, 0] > 8.0, 10.0, 0.0)
    return base + axis0_cost + settling + zone_penalty
```

## Usage in the Loop

```python
gpo = GPOptimizer(
    x_data=x_data,
    y_data=y_data,
    cost_function=my_cost,
)

current_position = np.array([[0.0, 0.0]])       # where the instrument is now

for i in range(N_ITERATIONS):
    next_x = gpo.ask(
        input_set=parameter_bounds,
        position=current_position,               # REQUIRED — sets `origin`
        acquisition_function="variance",
    )["x"]
    y = measure(next_x)
    gpo.tell(next_x, y)
    current_position = next_x                    # advance the origin
```

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

gpo.cost_function = partial(travel_cost, offset=offset, weights=speed)
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
