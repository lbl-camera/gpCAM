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
- Single best point, **noisy** → `"noisy expected improvement"` (integrates over the
  posterior of the incumbent instead of trusting the largest noisy reading), or `"ucb"`
  if acquisition evaluation cost matters more than statistical correctness
- The loop has stalled — it keeps re-measuring near the same point → `"knowledge gradient"`
- Landscape → `"variance"`, then switch to `"ucb"` once the posterior settles
- They already know roughly where the peak is and want it refined → `"expected improvement"`
  (low noise) or `"noisy expected improvement"`

**Why not plain EI by default.** Its EI anchors to `np.max(y_data)` — the largest
*noisy observation*, not the posterior mean — so on noisy data it locks onto a spike;
this is exactly what NEI was added to fix. (In legacy gpCAM ≤ 8.4.2, plain EI also
clipped `imp` before dividing by `σ`, collapsing to `0.3989 · σ(x)` wherever the
posterior mean sat below the incumbent — fixed in
[#55](https://github.com/lbl-camera/gpCAM/issues/55).)

```python
# Explore first, then exploit — a good default for peak finding.
for i in range(N_ITERATIONS):
    acq = "variance" if i < 20 else "ucb"
    next_x = gpo.ask(input_set=parameter_bounds, acquisition_function=acq)["x"]
    y, nv = measure(next_x)
    gpo.tell(next_x, np.array([y]), noise_variances=np.array([nv]))
    if (i + 1) % RETRAIN_EVERY == 0:
        gpo.train(hyperparameter_bounds=hp_bounds, method="local", max_iter=100, n_restarts=1)
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
    threshold = 0.5  # EDIT — in the GP's modeling space
    mean = gpo.posterior_mean(x)["m(x)"]
    std = np.sqrt(np.maximum(
        gpo.posterior_covariance(x, variance_only=True)["v(x)"], 1e-10))
    eps = 0.01 * np.std(gpo.y_data) + 1e-12  # scale-aware softening
    return std / (np.abs(mean - threshold) + eps)

next_x = gpo.ask(input_set=parameter_bounds,
                 acquisition_function=threshold_finder)["x"]
```

Warn them that everything away from the level set stays coarsely sampled. If they also
want a usable map, alternate this with `"variance"`.

---

## 5. Target value — "find conditions giving output ≈ X"

Common in sample synthesis and alignment: not a maximum, a specification.

```python
next_x = gpo.ask(input_set=parameter_bounds,
                 acquisition_function="target probability",
                 args={"a": 0.95, "b": 1.05})["x"]  # or pass args={...} to GPOptimizer(...)
```

`args` must carry `'a'` (lower) and `'b'` (upper) — passed to `GPOptimizer(..., args=...)` or `gpo.ask(..., args=...)` — or the call raises. For `LogGPOptimizer` / `LogitGPOptimizer`, pass the bounds in latent space (`np.log(a)` for `LogGPOptimizer`; `scipy.special.logit((a - lo) / (hi - lo))` for `LogitGPOptimizer` with `range=(lo, hi)`).

---

## 6. Minimization — "find the conditions that minimize my background"

**This is where a reflexive EI recommendation is outright wrong.** gpCAM's
`"expected improvement"`, `"probability of improvement"`, `"ucb"`, `"maximum"`,
`"knowledge gradient"`, and `"noisy expected improvement"` all hardcode a maximum —
`np.max(gpo.y_data)`, `np.max(mu_ref)`, or the raw posterior mean — so they hunt maxima
with no error raised. There is no minimizing KG or NEI.

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

**Ask back:** *"What should the 5 points do — find the best conditions, or map the
space?"* The batch is chosen under that goal's acquisition function, so the choice is the
same as for one point (sections 1–6); batching itself needs nothing special.

Since gpCAM 8.8, `ask(n > 1)` chooses the points together, greedily: the best point, then
the best given that one is pending, and so on. 5 copies of the same suggestion cannot
happen. With `"noisy expected improvement"` the batch is exact sequential-greedy q-NEI;
other acquisition functions condition on each pick through a pseudo-observation.

```python
result = gpo.ask(input_set=parameter_bounds, n=5,
                 acquisition_function="noisy expected improvement")
next_x = result["x"]        # shape (5, D), in the order picked
```

**Mixed batches.** Some exploitation, some exploration in one round: ask for the second
batch with the first one pending, so it plans around it.

```python
exploit = gpo.ask(parameter_bounds, n=3, acquisition_function="noisy expected improvement")["x"]
explore = gpo.ask(parameter_bounds, n=2, acquisition_function="variance", x_pending=exploit)["x"]
next_x = np.vstack([exploit, explore])
```

The same `x_pending=` keeps an asynchronous loop from re-proposing points whose
measurements are still running.

**gpCAM ≤ 8.7** handled `n > 1` differently: over bounds it silently replaced any string
acquisition function by `"total correlation"`, so `ask(n=5, acquisition_function="expected
improvement")` did not run EI. For that set-valued behavior now, pass
`batch_mode="joint"` with `"total correlation"` or `"relative information entropy"`;
for hgdl's distinct local optima, `batch_mode="independent", method="hgdl"` and a Dask
client.

---

## 8. Multi-task — "each measurement returns a spectrum"

Use `GPOptimizer` with `y_data` of shape `(V, No)` and a `multi_task_mode` (see the
`multi-task-advanced` skill), and `x_out` to pick outputs.

**Ask back:** *"Are you mapping all the channels, or optimizing something derived from
them?"*

- Mapping / exploring → `"relative information entropy set"` or `"variance"`
- Optimizing → `"knowledge gradient"` or `"noisy expected improvement"`

**Why KG or NEI over plain multi-task EI.** The `x_out` branch (`surrogate_model.py:375-389`)
scalarizes to the task-summed objective `g(x) = Σ_t f(x, t)` using the best task-summed
posterior mean at the observed inputs (`_incumbent_task_sum`, `surrogate_model.py:415-427`)
and `std = sqrt(Σ_t Var[f(x, t)])`. That spread drops the cross-task covariance, so plain
multi-task EI is accurate only when tasks are uncorrelated. KG and NEI scalarize the same
task-summed objective `g(x) = Σ_t f(x, t)` using the exact cross-task covariance
(`_scalarized_blocks`, `surrogate_model.py:430-452`).

```python
# Exploring all channels
next_x = gpo.ask(input_set=parameter_bounds,
                 x_out=np.arange(n_tasks),
                 acquisition_function="relative information entropy set",
                 vectorized=True)["x"]

# Optimizing the task-summed objective sum_t f(x, t)
next_x = gpo.ask(input_set=parameter_bounds,
                 x_out=np.arange(n_tasks),
                 acquisition_function="knowledge gradient")["x"]
```

If the scientist wants to optimize *one specific channel* rather than the task sum,
that is a single-task problem on a derived scalar — extract it in `measure()` and use
archetype 1.

---

## 9. Cost-constrained — "moving that motor takes 4 minutes"

The acquisition function stays whatever the science calls for; the *cost* is a separate
argument. `evaluate_acquisition_function` divides the acquisition by the cost function,
so an expensive move must earn its score.

```python
def my_cost(origin, x):
    """Minutes to move from `origin` to each row of `x`. Must be > 0."""
    return 1.0 + 4.0 * np.abs(x[:, 0] - origin[0])

gpo = GPOptimizer(x_data, y_data, cost_function=my_cost)
next_x = gpo.ask(input_set=parameter_bounds,
                 acquisition_function="variance",
                 position=current_motor_position)["x"]
```

`position` must be passed to `ask()` — `evaluate_acquisition_function` only applies the
cost when `origin is not None` (`surrogate_model.py:264-265`), so omitting it silently
disables cost-awareness. See the `cost-functions` skill for the full contract and for
fitting cost parameters from observed timings.

---

## Quick reference

| Goal | Choice | Never use |
|---|---|---|
| Map the space | `"variance"` | EI, UCB |
| Map changes / transitions | `radical_gradient` or `"gradient"` | `"maximum"` |
| Find the max, low noise | `"ucb"` | — |
| Find the max, noisy | `"noisy expected improvement"` | plain EI (anchors to a noise spike) |
| Find the max, loop has stalled | `"knowledge gradient"` | `"maximum"` |
| Refine a known max | `"expected improvement"` | — |
| Find the min | `"lcb"` | EI, PI, UCB, KG, NEI, `"maximum"` |
| Hit a target value | `"target probability"` | EI |
| Find a level set | `threshold_finder` | EI |
| Batch | `"total correlation"`, `"relative information entropy set"` | any other string (silently overridden) |
| Multi-task, exploring | `"relative information entropy set"`, `"variance"` | plain EI |
| Multi-task, optimizing | `"knowledge gradient"`, `"noisy expected improvement"` | plain EI |
