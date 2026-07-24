---
name: acquisition-functions
description: Use when designing custom acquisition functions for gpCAM that encode experimental priorities — exploration vs exploitation balance, multi-objective targets, constrained search regions, cost-aware moves, UCB/LCB, or probability-of-improvement criteria.
gpcam_version: "8.4.x"
fvgp_version: "4.8.x"
last_verified: "2026-07-23 (gpCAM dadeb65)"
---

# Skill: gpCAM Acquisition Functions

*Verified against gpCAM 8.4.x / fvgp 4.8.x — last checked 2026-07-23 (gpCAM `dadeb65`).*

Design custom acquisition functions that control where gpCAM measures next.

## When to Use

When a user needs acquisition behavior beyond the built-in options:
- Balancing exploration and exploitation
- Multi-objective optimization
- Constrained search (avoid forbidden regions)
- Cost-aware acquisition (expensive moves)
- Upper/lower confidence bounds
- Probability of improvement

## Acquisition Function Contract

```python
def my_acquisition(x, gp_optimizer):
    """
    Parameters
    ----------
    x : np.ndarray, shape (V, D)
        Candidate points to evaluate.
    gp_optimizer : GPOptimizer
        The GP model. Use its posterior_mean() and posterior_covariance() methods.
    
    Returns
    -------
    scores : np.ndarray, shape (V,)
        Score for each candidate. HIGHER = more desirable (function is MAXIMIZED).
    """
```

**Key rule:** The acquisition function is **maximized**. Return higher values for points you want to measure.

### Reading the GP's data inside an acquisition function

`gpo.y_data` and `gpo.x_data` are public properties and are the right thing to use
for an incumbent value (`y_best`). They return the data **in the GP's own modeling
space**, which is the same space `posterior_mean()` returns — so the two are always
directly comparable.

`get_data()` returns the same arrays in a dict along with hyperparameters and
metadata. Use `get_data()` for reporting and inspection; use the `x_data` / `y_data`
properties inside acquisition functions, where you want the cheap attribute access
and guaranteed space-consistency.

> **With a transformed optimizer** (`LogGPOptimizer`, `LogitGPOptimizer`), `gpo.y_data`
> holds the *transformed* observations — `log(y)` or `logit(y)` — matching
> `posterior_mean()`. Every recipe below therefore stays internally consistent with no
> changes. What you must **not** do is mix spaces: `get_data()["original y data"]`
> is on the original scale and comparing it against `posterior_mean()` is a silent
> units bug. Any **absolute constant** you hardcode — a threshold, a target value, a
> forbidden output level — is also in latent space and must be transformed
> (`np.log(threshold)`, `scipy.special.logit(threshold)`). See the
> `transformed-optimizers-advanced` skill.

## Built-in Options

Pass these as strings to `gpo.ask(acquisition_function=...)`:

| Name | String key | Best for |
|------|-----------|----------|
| Variance | `"variance"` | Pure exploration / mapping |
| Expected Improvement | `"expected improvement"` | Optimization (find max) |
| Probability of Improvement | `"probability of improvement"` | Optimization, risk-averse |
| Upper Confidence Bound | `"ucb"` | Maximization with tunable exploration/exploitation |
| Lower Confidence Bound | `"lcb"` | Minimization with tunable exploration/exploitation |
| Predicted Maximum | `"maximum"` | Pure exploitation — mean only, no uncertainty |
| Predicted Minimum | `"minimum"` | Pure exploitation for minimization |
| Gradient | `"gradient"` | Seek steepest regions of the posterior mean |
| Target Probability | `"target probability"` | Find points with output near a target value |
| Relative Information Entropy | `"relative information entropy"` | Information-theoretic exploration |
| Relative Information Entropy Set | `"relative information entropy set"` | Batch acquisition |
| Total Correlation | `"total correlation"` | Batch acquisition |

To sanity-check any built-in or custom acquisition on a grid of candidates without calling `ask()`:
```python
scores = gpo.evaluate_acquisition_function(x_grid, acquisition_function="ucb")
```

## Custom Acquisition Recipes

### Upper Confidence Bound (UCB)
Available as the built-in string `"ucb"` — pass directly to `gpo.ask(acquisition_function="ucb")`. Write the callable form below only when you need to tune `beta` or otherwise customize the score:
```python
def ucb(x, gpo):
    """
    beta controls exploration/exploitation tradeoff:
      beta=0: pure exploitation (just go to predicted max)
      beta=1: mild exploration
      beta=3: strong exploration (~95% confidence)
    """
    beta = 2.0  # TUNE THIS
    mean = gpo.posterior_mean(x)["m(x)"]
    var = gpo.posterior_covariance(x, variance_only=True)["v(x)"]
    return mean + beta * np.sqrt(var)
```

### Lower Confidence Bound (for minimization)
gpCAM maximizes acquisition, so flip the sign for minimization:
```python
def lcb(x, gpo):
    """Find the minimum of the function."""
    beta = 2.0
    mean = gpo.posterior_mean(x)["m(x)"]
    var = gpo.posterior_covariance(x, variance_only=True)["v(x)"]
    return -(mean - beta * np.sqrt(var))  # note the negation
```

### Expected Improvement (custom version with minimization)
```python
from scipy.stats import norm

def expected_improvement_minimize(x, gpo):
    """Expected improvement for finding the MINIMUM."""
    mean = gpo.posterior_mean(x)["m(x)"]
    var = gpo.posterior_covariance(x, variance_only=True)["v(x)"]
    std = np.sqrt(np.maximum(var, 1e-10))
    
    y_best = np.min(gpo.y_data)  # current best (minimum)
    z = (y_best - mean) / std
    ei = std * (z * norm.cdf(z) + norm.pdf(z))
    return ei
```

### Probability of Improvement
```python
from scipy.stats import norm

def probability_of_improvement(x, gpo):
    """Probability that measurement improves on current best."""
    mean = gpo.posterior_mean(x)["m(x)"]
    var = gpo.posterior_covariance(x, variance_only=True)["v(x)"]
    std = np.sqrt(np.maximum(var, 1e-10))
    
    y_best = np.max(gpo.y_data)
    z = (mean - y_best) / std
    return norm.cdf(z)
```

### Constrained Acquisition (Avoid Regions)
```python
def constrained_variance(x, gpo):
    """Explore but avoid a circular forbidden zone."""
    var = gpo.posterior_covariance(x, variance_only=True)["v(x)"]
    
    # Forbidden zone: circle at (5, 5) with radius 1
    center = np.array([5.0, 5.0])
    dist = np.linalg.norm(x - center, axis=1)
    penalty = np.where(dist < 1.0, -1e6, 0.0)
    
    return var + penalty
```

### Multi-Objective (Weighted)

Combining terms with different units requires normalization — but **normalize against
fixed scales derived from the data, never against the current batch of candidates.**
Min-max scaling over `x` makes a point's score depend on which other candidates
happened to be evaluated alongside it, so the argmax shifts between calls and the
inner optimizer is chasing a moving target.

```python
def multi_objective(x, gpo):
    """
    Balance finding the max with reducing uncertainty.

    Both terms are divided by std(y_data), a fixed scale that does not depend on
    the candidate batch, so a given point always receives the same score.
    """
    mean = gpo.posterior_mean(x)["m(x)"]
    var = gpo.posterior_covariance(x, variance_only=True)["v(x)"]

    w_exploit = 0.7   # weight on exploitation
    w_explore = 0.3   # weight on exploration

    scale = np.std(gpo.y_data) + 1e-10        # fixed reference scale
    mean_term = (mean - np.mean(gpo.y_data)) / scale   # dimensionless z-score
    std_term = np.sqrt(var) / scale                    # dimensionless

    return w_exploit * mean_term + w_explore * std_term
```

Both terms are now in units of "standard deviations of the observed signal", which
also makes the weights interpretable: `w_exploit=0.7, w_explore=0.3` genuinely means
70/30. This is UCB with `beta = w_explore / w_exploit` up to a positive rescaling —
if that is all you need, prefer the built-in `"ucb"`.

If you must recompute the scale as data accumulates, cache it and refresh it only on
retraining, not on every acquisition evaluation.

### Threshold Finder (Find Boundary)
Useful when searching for where a signal crosses a threshold:
```python
def threshold_finder(x, gpo):
    """
    Find the boundary where f(x) = threshold.

    The softening epsilon is scaled to the signal, not hardcoded. A fixed 0.01 is
    meaningful only when y is O(1): on detector counts it is negligible and the
    acquisition becomes a near-singular spike; on data of order 1e-3 it dominates
    and the acquisition goes flat. Both degrade silently.
    """
    threshold = 0.5  # EDIT THIS — in the GP's modeling space (see the transform note above)

    mean = gpo.posterior_mean(x)["m(x)"]
    var = gpo.posterior_covariance(x, variance_only=True)["v(x)"]
    std = np.sqrt(np.maximum(var, 1e-10))

    eps = 0.01 * np.std(gpo.y_data) + 1e-12    # scale-aware softening
    distance_to_threshold = np.abs(mean - threshold)
    return std / (distance_to_threshold + eps)
```

An equivalent formulation that needs no epsilon at all — score by the probability
that the point sits near the threshold:
```python
def threshold_finder_prob(x, gpo):
    """Probability-based boundary search; no scale-dependent constant."""
    threshold = 0.5
    mean = gpo.posterior_mean(x)["m(x)"]
    std = np.sqrt(np.maximum(gpo.posterior_covariance(x, variance_only=True)["v(x)"], 1e-10))
    return norm.pdf((mean - threshold) / std) / std
```

## Usage in the Experiment Loop

```python
# Built-in string or a callable are both accepted:
result = gpo.ask(
    input_set=parameter_bounds,
    acquisition_function=ucb,   # or "ucb", "expected improvement", ...
)
```

### Useful `ask()` options

| Argument | Meaning |
|----------|---------|
| `n=N` | Request `N` points at once (batch). For vectorized single-task, use a batch-aware acquisition like `"relative information entropy set"` or `"total correlation"`. |
| `vectorized=True` (default) | The acquisition function is called once with all candidate points, shape `(V, D)` — required for custom callables written against the contract above. |
| `vectorized=False` | Candidates are evaluated one at a time (list of 1-D arrays). Used for non-vectorizable acquisition or non-Euclidean inputs. |
| `method="global"\|"local"\|"hgdl"\|"hgdlAsync"` | Inner optimizer that searches for the argmax of the acquisition over `input_set`. `hgdl` requires `dask_client=`; `hgdlAsync` starts a background search and returns an `opt_obj` you can poll or `kill_client()`. |
| `dask_client=client` | Distribute the inner optimization across Dask workers. |
| `batch_size=B` | When candidates are a list, evaluate them in chunks of `B` on the cluster. |
| `max_iter`, `pop_size`, `info=True` | Inner optimizer controls. |

`input_set` can be continuous bounds (`np.array([[lo,hi], ...])`), a list of candidate points (discrete finite set), or a list of arbitrary objects (non-Euclidean — strings, graphs — provided your kernel handles them).

## Hyperparameter Coordination

Acquisition functions don't add hyperparameters — they read the GP state via `gpo.posterior_mean()` and `gpo.posterior_covariance()`. However:

- `gpo.y_data` reflects every `tell()` immediately, so `y_best` is current without extra bookkeeping. It is in the GP's modeling space (see the note at the top of this skill).
- Acquisition functions receive no `args` dict — unlike kernel/mean/noise callables, the signature is exactly `f(x, gp_optimizer)`. Bind extra parameters with a closure or `functools.partial`.
- The GP must be trained before acquisition makes sense — always call `train()` first
- For `variance_only=True`: faster, returns just diagonal variances (usually what you want)
- For full covariance: use `variance_only=False` but this is O(V²) memory

## Common Pitfalls

1. **Returning negative scores for points you want**: Remember, acquisition is MAXIMIZED.
2. **Division by zero in std**: Always use `np.maximum(var, 1e-10)` before taking sqrt.
3. **Not handling edge cases**: Early in the loop with few points, the GP posterior can be unreliable.
4. **Expensive acquisition functions**: They're evaluated many times during optimization. Keep them fast.
