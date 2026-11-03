---
name: acquisition-functions
description: Use when choosing or writing an acquisition function for gpCAM — selecting a built-in criterion for exploration, optimization, batch, or multi-task runs, or implementing a custom callable for UCB/LCB, minimization EI, constraints, radical gradient, or threshold finding.
gpcam_version: "8.7.x"
fvgp_version: "4.11.x"
last_verified: "2026-10-28 (gpCAM 8.7.0, a21385a)"
---

# Skill: gpCAM Acquisition Functions

*Verified against gpCAM 8.7.0 / fvgp 4.11.1 — last checked 2026-10-28 (gpCAM a21385a).*

Choose — and where needed, write — the acquisition function that controls where gpCAM measures next.

## When to Use

- Deciding which built-in string or custom callable to pass to `ask(acquisition_function=...)`
- Balancing exploration and exploitation, or switching strategies mid-run
- Minimization, constrained search, multi-objective targets, change-mapping (`"gradient"` / `radical_gradient`), or level-set (`threshold_finder` / `"target probability"`) criteria

## Choose the Acquisition Function *With* the User

**Never pick an acquisition function silently.** Before generating a script:

1. **Name your recommendation and its main tradeoff** (e.g., `"variance"` gives uniform coverage across the space, whereas `"ucb"` concentrates shots near the peak).
2. **Disambiguate "find the best conditions":** ask whether they need the single best operating point (`"ucb"`, `"noisy expected improvement"`, `"knowledge gradient"`) or a trustworthy map of the landscape (`"variance"`, `"gradient"`, `radical_gradient`), and confirm whether they are **maximizing or minimizing**.
3. **Remember the acquisition is an `ask()` argument, not a fixed GP property** — exploring with `"variance"` for the first ~20 points before switching to `"ucb"` or `"noisy expected improvement"` is often the best design.

Do **not** default to `"expected improvement"`: it is maximization-only, anchors to a single noisy observation (`np.max(y_data)`), and in legacy gpCAM ≤ 8.4.2 degenerated to pure exploration below the incumbent.

For per-archetype dialogue prompts ("Ask back...") and loop templates across all 9 experimental archetypes, see [references/choosing-an-acquisition-function.md](references/choosing-an-acquisition-function.md).

## Built-in Options & Decision Guide

Pass these strings to `gpo.ask(acquisition_function=...)`:

| Goal / Scenario | String key (or callable) | Key behavior & tradeoffs |
|---|---|---|
| Pure exploration / replace raster scan | `"variance"` | Uniform posterior confidence everywhere; won't refine peaks |
| Maximize (low noise, fast) | `"ucb"` | `mean + 3.0 * std`; reliable general-purpose maximizer (`beta=3.0` fixed — use a callable to tune `beta`) |
| **Minimize** | `"lcb"` (or `expected_improvement_minimize`) | `-(mean - 3.0 * std)`; **never** use EI, PI, UCB, KG, or NEI for minimization |
| Maximize under **observation noise** | `"noisy expected improvement"` | Integrates over the incumbent posterior (`nei_samples`, `nei_reference_set_size`, `nei_seed` via `args`); slower than `"ucb"` |
| Maximize when greedy EI/UCB stalls | `"knowledge gradient"` | One-step lookahead expected gain in `max posterior_mean` (`kg_reference_set_size`, `kg_seed` via `args`); heaviest built-in |
| Late local refinement (low noise) | `"expected improvement"` | Maximization-only; anchors to `np.max(y_data)` (single-task) or `_incumbent_task_sum` (multi-task) |
| Risk-averse maximization | `"probability of improvement"` | Maximization-only; greedier than EI and stalls easily |
| Map steep transitions / edges | `"gradient"` or `radical_gradient` callable | `"gradient"` scores `‖∇m(x)‖ · σ(x)` (sharpest ridge); `radical_gradient` uses `√‖∇m(x)‖ · σ(x)` to spread across multiple transitions |
| Find where output ∈ `[a, b]` | `"target probability"` | Requires `args={"a": lo, "b": hi}` on `GPOptimizer(...)` or `gpo.ask(..., args=...)` |
| Find level set `f(x) = threshold` | `threshold_finder` callable | Concentrates on one contour; alternate with `"variance"` if a full map is also needed |
| Batch (`n > 1` points jointly) | `"total correlation"` or `"relative information entropy"` | The only two built-ins that score a **set** jointly (forces `vectorized=False`; `"total correlation"` unavailable in `"kronecker"` mode) |
| Multi-task exploration | `"relative information entropy set"` or `"variance"` | Vectorized exploration across `x_out` tasks |
| Multi-task maximization (`Σ_t f(x, t)`) | `"knowledge gradient"` or `"noisy expected improvement"` | Uses exact cross-task covariance (`_scalarized_blocks`), whereas multi-task `"expected improvement"` assumes independent tasks |
| Pure exploitation (no uncertainty) | `"maximum"` / `"minimum"` | Posterior mean only (`σ` ignored) — almost always stalls at one point |

To compare any built-in or custom acquisition on a grid before running the loop:
```python
scores = gpo.evaluate_acquisition_function(x_grid, acquisition_function="ucb")
```
(`evaluate_acquisition_function` flips the internal minimization sign back at `gp_optimizer_base.py:301` so higher scores mean higher priority.)

### Set-valued batch acquisitions (`"total correlation"` vs `"relative information entropy"`)

`"total correlation"` and `"relative information entropy"` are the only built-ins that score a **whole candidate set** with a single number rather than scoring each point independently:
- **`"total correlation"`** measures the KL divergence between the joint distribution over existing data + candidates and the factored distribution with data-to-candidate and candidate-to-candidate correlations removed. Negated for maximization, it penalizes both redundancy with existing data **and** mutual redundancy among candidates in the batch. (Not available when `multi_task_mode="kronecker"`, `gp_optimizer_base.py:443-445`.)
- **`"relative information entropy"`** measures the KL divergence between the prior and posterior at the candidates alone, preferring unexplored regions without explicitly penalizing candidate-to-candidate proximity.

## gpCAM-Specific Behavior You Must Know

**(a) Legacy gpCAM ≤ 8.4.2 EI bug (fixed in #55 / gpCAM ≥ 8.5).**
In `surrogate_model.py:409-412`, `_expected_improvement` standardizes `z = imp / (std + 1e-9)` before evaluating `imp * norm.cdf(z) + std * norm.pdf(z)`. In gpCAM ≤ 8.4.2, `imp` was clipped to 0 *before* dividing by `std`, collapsing EI to `0.3989 · σ(x)` wherever `mean < y_best`. If a user is on an older install, check with:
```python
ei  = gpo.evaluate_acquisition_function(x_grid, acquisition_function="expected improvement")
var = gpo.evaluate_acquisition_function(x_grid, acquisition_function="variance")
below = gpo.posterior_mean(x_grid)["m(x)"] < np.max(gpo.y_data)
print(np.unique(np.round(ei[below] / var[below], 6)))  # [0.398942] on <=8.4.2; spread below 0.398942 on >=8.5
```

**(b) EI, PI, UCB, KG, and NEI are all maximization-only.**
Single-task EI/PI hardcode `last_best = np.max(gp.y_data)` (`surrogate_model.py:306, 313`), multi-task EI maximizes the task sum (`_incumbent_task_sum`, `surrogate_model.py:415-427`), and KG/NEI take `np.max(mu_ref)` (`surrogate_model.py:462, 513`). For minimization, use `"lcb"` or `expected_improvement_minimize` below.

**(c) Single-task plain EI/PI anchor to a noisy observation.**
`np.max(gpo.y_data)` is the largest raw reading, not the posterior mean, so on noisy data plain EI over-exploits noise spikes. Prefer `"noisy expected improvement"` (or `"knowledge gradient"`), which integrates over the posterior at the observed reference points (`surrogate_model.py:495-520`).

**(d) Multi-task plain EI assumes uncorrelated tasks.**
When `x_out` is set (`surrogate_model.py:375-389, 415-427`), multi-task `"expected improvement"` scalarizes to the task-summed objective `g(x) = Σ_t f(x, t)` using `_incumbent_task_sum` (the maximum task-summed **posterior mean** at the observed inputs, so missing `NaN` task entries in `"cartesian"` mode do not poison EI) and `std = sqrt(Σ_t Var[f(x, t)])`. That spread drops cross-task covariance, so it is exact only when tasks are independent. `"knowledge gradient"` and `"noisy expected improvement"` scalarize the same `Σ_t f(x, t)` objective using the full cross-task covariance via `_scalarized_blocks` (`surrogate_model.py:430-452`).

**(e) Batch (`n > 1`) behavior depends on `input_set` type.**
- With a continuous `np.ndarray` bounds `input_set`, `n > 1`, `method != "hgdl"`, and a string acquisition other than `"total correlation"` or `"relative information entropy"`, `gp_optimizer_base.py:583-594` warns and overrides the string to `"total correlation"` (or `"relative information entropy"` when `multi_task_mode == "kronecker"`). To get `n > 1` local optima of a point-wise acquisition over bounds, pass a callable with `method="hgdl"` and a `dask_client`.
- With a discrete **candidate list** and `n > 1`, `"total correlation"` and `"relative information entropy"` select the batch jointly via greedy forward selection (`gp_optimizer_base.py:618-625`), returning a single scalar score in `"f_a(x)"`. All other strings and custom callables are scored point-by-point and truncated to the top `n` (`gp_optimizer_base.py:626-636`), which warns and can pick clustered duplicates.

**(f) `"ucb"` and `"lcb"` fix `beta = 3.0`** (`surrogate_model.py:291, 295, 366, 372`). Write the 5-line callable below to change `beta`.

**(g) KG and NEI cost O((N_ref + V)²) per call.**
Keep `kg_reference_set_size` / `nei_reference_set_size` modest (default `100`) and pass via `GPOptimizer(..., args=...)` or `gpo.ask(..., args=...)`:
```python
gpo.ask(bounds, acquisition_function="noisy expected improvement",
        args={"nei_samples": 128, "nei_reference_set_size": 100, "nei_seed": 0})
```

## Acquisition Function Contract

```python
def my_acquisition(x, gp_optimizer):
    """
    x : np.ndarray, shape (V, D) — candidate points to evaluate.
    gp_optimizer : GPOptimizer — use posterior_mean() and posterior_covariance().
    Returns: scores : np.ndarray, shape (V,) — HIGHER = more desirable (MAXIMIZED).
    """
```

- **Reading GP data:** Use `gpo.x_data` and `gpo.y_data` inside custom acquisitions (not `get_data()["original y data"]`). For `LogGPOptimizer` and `LogitGPOptimizer`, `gpo.y_data` is already in latent (`log` or `logit`) space matching `posterior_mean()`; any hardcoded threshold or target constant must also be transformed into latent space (`np.log(threshold)` or `scipy.special.logit((threshold - lo) / (hi - lo))` — see `transformed-optimizers-advanced`).

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
Required if the scientist is minimizing — the built-in `"expected improvement"` string
is hardcoded to maximize (caveat b). This version forms `z` before clipping, so it is
also the safe choice on gpCAM ≤ 8.4.2, where the built-in degenerates (caveat a).

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

On noisy measurements, swap the incumbent for the model's own estimate so EI doesn't
chase a noise spike (caveat c):

```python
    y_best = np.min(gpo.posterior_mean(gpo.x_data)["m(x)"])
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

### Radical Gradient (Map Where the Signal *Changes*)
The best default when the scientist wants to map how a signal changes across parameter
space rather than find an optimum — phase boundaries, transitions, onset of an effect.

```python
def radical_gradient(x, gpo):
    """Square root of the gradient magnitude, weighted by uncertainty.

    The built-in "gradient" acquisition is  ||grad m(x)|| * sigma(x).
    This takes the square root of the gradient term only:

        ||grad m(x)||           ->   sqrt(||grad m(x)||)

    Softening the gradient weighting lets the uncertainty term carry relatively
    more weight, so sampling spreads across every region that is changing instead
    of piling onto the single steepest ridge. Better coverage of a structured
    landscape; slightly slower to nail any one feature.
    """
    g = gpo.posterior_mean_grad(x)["dm/dx"]
    std = np.sqrt(gpo.posterior_covariance(x, variance_only=True)["v(x)"])
    return np.sqrt(np.linalg.norm(g, axis=1)) * std
```

**Radical gradient vs. built-in `"gradient"`:** both are good for change-mapping.
Use `radical_gradient` when the space has several features of differing steepness and
you want all of them characterized. Use `"gradient"` when there is one dominant
transition and finding it precisely is the point. If unsure, offer the scientist both
and note that swapping them is a one-line change.

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
| `n=N` | Request `N` points at once (batch). Use `"total correlation"` or `"relative information entropy"` — the two that score a set — or the batch is chosen point by point and will cluster. |
| `vectorized=True` (default) | The acquisition function is called once with all candidate points, shape `(V, D)` — required for custom callables written against the contract above. |
| `vectorized=False` | Candidates are evaluated one at a time (list of 1-D arrays). Used for non-vectorizable acquisition or non-Euclidean inputs. |
| `method="global"\|"local"\|"hgdl"\|"hgdlAsync"` | Inner optimizer that searches for the argmax of the acquisition over `input_set`. `hgdl` requires `dask_client=`; `hgdlAsync` starts a background search and returns an `opt_obj` you can poll or `kill_client()`. |
| `dask_client=client` | Distribute the inner optimization across Dask workers. |
| `batch_size=B` | When candidates are a list, evaluate them in chunks of `B` on the cluster. |
| `args={...}` | Updates `gpo.args` before evaluating the acquisition (`gp_optimizer_base.py:373, 536-537, 559`) — used by `"target probability"` (`{'a': lo, 'b': hi}`) and KG/NEI tuning keys. |
| `max_iter`, `pop_size`, `info=True` | Inner optimizer controls. |

`input_set` can be continuous bounds (`np.array([[lo,hi], ...])`), a list of candidate points (discrete finite set), or a list of arbitrary objects (non-Euclidean — strings, graphs — provided your kernel handles them).

## Hyperparameter & `args` Coordination

Acquisition functions don't add hyperparameters — they read the trained GP state via `gpo.posterior_mean(x)` and `gpo.posterior_covariance(x, variance_only=True)` (prefer `variance_only=True` to avoid O(V²) covariance matrices). Custom acquisition callables always have the 2-argument signature `f(x, gp_optimizer)`; bind extra parameters with `functools.partial` or read `gp_optimizer.args` (see the `experiment-designer` skill for the full `args` dispatch table).

## Common Pitfalls

1. **Reaching for `"expected improvement"` by reflex.** It is the best-known acquisition
   function, not the best-suited one. Work the choice out with the scientist using the
   decision table above, and check it against the caveats before recommending it.
2. **Recommending a maximization acquisition to someone minimizing.** `"expected
   improvement"`, `"probability of improvement"`, `"ucb"`, `"maximum"`,
   `"knowledge gradient"`, and `"noisy expected improvement"` all assume
   bigger-is-better. Confirm the direction explicitly — scientists often say "optimize"
   for both.
3. **Returning negative scores for points you want**: Remember, acquisition is MAXIMIZED.
4. **Division by zero in std**: Always use `np.maximum(var, 1e-10)` before taking sqrt.
5. **Not handling edge cases**: Early in the loop with few points, the GP posterior can be unreliable.
6. **Expensive acquisition functions**: They're evaluated many times during optimization. Keep them fast.
7. **Assuming the acquisition is fixed for the run.** It's an `ask()` argument. Switching
   from exploration to exploitation partway through is often the right design.
