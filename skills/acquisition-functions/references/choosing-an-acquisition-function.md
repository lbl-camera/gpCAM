# Choosing an Acquisition Function: Worked Examples

One entry per experiment archetype. Each gives how a scientist typically phrases the
goal, what to ask back before committing, the choice, and the loop snippet.

Read this alongside the caveats in `../SKILL.md` — several built-ins behave differently
from their textbook definitions.

**The meta-rule:** the scientist's opening sentence almost never determines the
acquisition function on its own. One follow-up question usually does.

---

## 1. Peak finding — "find the conditions that maximize my signal"

**Ask back:** *"Do you want the single best point to run production at, or a map of the
landscape that happens to contain the peak? And roughly how noisy is a single
measurement?"*

- Single best point, low noise → `"ucb"`
- Single best point, noisy → `"ucb"` (its `+3σ` term is robust to a noisy incumbent in
  a way EI is not)
- Landscape → `"variance"`, then switch to `"ucb"` once the posterior settles
- They already know roughly where the peak is and want it refined → `"expected improvement"`

**Why not EI by default.** Two reasons specific to gpCAM. Its EI anchors to
`np.max(y_data)` — the largest *noisy observation*, not the posterior mean — so on
noisy data it locks onto a spike. And until the posterior mean somewhere exceeds that
incumbent, gpCAM's EI is algebraically `0.3989 · σ(x)`, i.e. identical in ranking to
`"variance"`. Early in a run it is not exploiting at all.

```python
# Explore first, then exploit — a good default for peak finding.
for i in range(N_ITERATIONS):
    acq = "variance" if i < 20 else "ucb"
    next_x = gpo.ask(input_set=parameter_bounds, acquisition_function=acq)["x"]
    y, nv = measure(next_x)
    gpo.tell(next_x, np.array([y]), noise_variances=np.array([nv]), append=True)
    if (i + 1) % RETRAIN_EVERY == 0:
        gpo.train(hyperparameter_bounds=hp_bounds, method="local", max_iter=100)
```

---

## 2. Mapping / surrogate building — "replace my raster scan"

**Ask back:** *"Do you need uniform quality everywhere, or is some part of the space
more important?"*

- Uniform → `"variance"`
- Information-theoretic variant, slower but sharper → `"relative information entropy"`
- Some regions matter more → `"variance"` with a multiplicative weight (see the
  constrained/multi-objective recipes in `../SKILL.md`)

This is the archetype where an optimization acquisition does real damage: EI or UCB will
concentrate shots near the maximum and leave the rest of the map at raster-quality or
worse, which is the opposite of what was asked for.

```python
next_x = gpo.ask(input_set=parameter_bounds, acquisition_function="variance")["x"]
```

---

## 3. Change mapping — "find where the behavior transitions"

Phase boundaries, onset of an effect, the edge of a feature. The scientist wants the
*derivative* structure, not the extremum.

**Ask back:** *"Is there one transition you need located precisely, or several features
of different sharpness you want all characterized?"*

- One dominant transition → `"gradient"` (built-in: `‖∇m‖ · σ`)
- Several features, differing steepness → `radical_gradient` (`√‖∇m‖ · σ`)

The square root softens the gradient weighting so the uncertainty term carries
relatively more weight. Sampling then spreads across every changing region instead of
piling onto the single steepest ridge.

```python
import numpy as np

def radical_gradient(x, gpo):
    g = gpo.posterior_mean_grad(x)["dm/dx"]
    std = np.sqrt(gpo.posterior_covariance(x, variance_only=True)["v(x)"])
    return np.sqrt(np.linalg.norm(g, axis=1)) * std

next_x = gpo.ask(input_set=parameter_bounds,
                 acquisition_function=radical_gradient)["x"]
```

Swapping between the two is a one-line change — say so, and let the scientist try both
on the same data.

---

## 4. Boundary / threshold finding — "find where the signal crosses X"

**Ask back:** *"Is X a hard number, or a band?"* A band means archetype 5 instead.

```python
def threshold_finder(x, gpo):
    threshold = 0.5  # EDIT
    mean = gpo.posterior_mean(x)["m(x)"]
    std = np.sqrt(np.maximum(
        gpo.posterior_covariance(x, variance_only=True)["v(x)"], 1e-10))
    return std / (np.abs(mean - threshold) + 0.01)

next_x = gpo.ask(input_set=parameter_bounds,
                 acquisition_function=threshold_finder)["x"]
```

Warn them that everything away from the level set stays coarsely sampled. If they also
want a usable map, alternate this with `"variance"`.

---

## 5. Target value — "find conditions giving output ≈ X"

Common in sample synthesis and alignment: not a maximum, a specification.

```python
gpo = GPOptimizer(x_data, y_data, args={"a": 0.95, "b": 1.05})  # target band
next_x = gpo.ask(input_set=parameter_bounds,
                 acquisition_function="target probability")["x"]
```

`args` must carry `'a'` (lower) and `'b'` (upper) or the call raises. For
`LogGPOptimizer` / `LogitGPOptimizer`, pass the bounds already transformed —
`np.log(0.95)`, `np.log(1.05)`.

---

## 6. Minimization — "find the conditions that minimize my background"

**This is where a reflexive EI recommendation is outright wrong.** gpCAM's
`"expected improvement"`, `"probability of improvement"`, `"ucb"`, and `"maximum"` all
hardcode `np.max(gpo.y_data)` or the raw posterior mean, so they hunt maxima with no
error raised.

- Built-in → `"lcb"` (β fixed at 3.0)
- Need a tunable β, or EI semantics → write the callable

```python
def lcb(x, gpo):
    beta = 2.0  # TUNE
    mean = gpo.posterior_mean(x)["m(x)"]
    std = np.sqrt(np.maximum(
        gpo.posterior_covariance(x, variance_only=True)["v(x)"], 1e-10))
    return -(mean - beta * std)   # negated: gpCAM maximizes acquisition
```

An alternative some scientists prefer: negate the measurement in `measure()` and keep
maximizing. Cleaner, but remember to flip the sign back when reporting results.

---

## 7. Batch — "give me 5 points per round, my detector takes them in parallel"

**Ask back:** *"Do the 5 points need to be jointly informative, or is 5 independent
suggestions fine?"*

Jointly informative is the reason batch acquisition exists — 5 copies of the same greedy
suggestion is worthless.

```python
result = gpo.ask(input_set=parameter_bounds, n=5,
                 acquisition_function="total correlation")
next_x = result["x"]        # shape (5, D)
```

**Critical caveat.** With an array `input_set`, `n > 1`, `method != "hgdl"`, and a
*string* acquisition, gpCAM replaces whatever was requested with `"total correlation"`
and emits a warning (`gp_optimizer_base.py:517-525`). So `ask(n=5,
acquisition_function="expected improvement")` does not run EI. Either ask for a
batch-aware acquisition deliberately, or pass a callable with `method="hgdl"` and a
Dask client:

```python
from distributed import Client
client = Client()
result = gpo.ask(input_set=parameter_bounds, n=5,
                 acquisition_function=my_callable,
                 method="hgdl", dask_client=client)
```

---

## 8. Multi-task — "each measurement returns a spectrum"

Use `fvGPOptimizer` with `x_out`. Recommend `"relative information entropy set"` or
`"variance"`.

**Do not recommend EI here.** The `x_out` branch (`surrogate_model.py:306`) sums the
posterior mean across tasks and sums the standard deviations across tasks, then compares
that sum against `np.max(gpo.y_data)` of a scalar. The quantities are not comparable and
the result is not expected improvement of anything.

```python
next_x = gpo.ask(input_set=parameter_bounds,
                 x_out=np.arange(n_tasks),
                 acquisition_function="relative information entropy set",
                 vectorized=True)["x"]
```

If the scientist genuinely wants to optimize one channel of the spectrum, that is a
single-task problem on a derived scalar — extract it in `measure()` and use archetype 1.

---

## 9. Cost-constrained — "moving that motor takes 4 minutes"

The acquisition function stays whatever the science calls for; the *cost* is a separate
argument. `evaluate_acquisition_function` divides the acquisition by the cost function,
so an expensive move must earn its score.

```python
def my_cost(origin, x, arguments=None):
    """Minutes to move from `origin` to each row of `x`. Must be > 0."""
    return 1.0 + 4.0 * np.abs(x[:, 0] - origin[0])

gpo = GPOptimizer(x_data, y_data, cost_function=my_cost)
next_x = gpo.ask(input_set=parameter_bounds,
                 acquisition_function="variance",
                 position=current_motor_position)["x"]
```

`position` must be passed to `ask()` — `evaluate_acquisition_function` only applies the
cost when `origin is not None` (`surrogate_model.py:196`), so omitting it silently
disables cost-awareness. See the `cost-functions` skill for the full contract and for
fitting cost parameters from observed timings.

---

## Quick reference

| Goal | Choice | Never use |
|---|---|---|
| Map the space | `"variance"` | EI, UCB |
| Map changes / transitions | `radical_gradient` or `"gradient"` | `"maximum"` |
| Find the max | `"ucb"` | — |
| Refine a known max | `"expected improvement"` | — |
| Find the min | `"lcb"` | EI, PI, UCB, `"maximum"` |
| Hit a target value | `"target probability"` | EI |
| Find a level set | `threshold_finder` | EI |
| Batch | `"total correlation"`, `"relative information entropy set"` | any other string (silently overridden) |
| Multi-task | `"relative information entropy set"`, `"variance"` | EI |
