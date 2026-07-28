---
name: acquisition-functions
description: Use when choosing or designing the acquisition function for a gpCAM experiment — deciding which built-in actually fits the scientist's goal (and confirming it with them rather than defaulting to expected improvement), or writing a custom one for exploration/exploitation balance, multi-objective targets, constrained search regions, cost-aware moves, change-mapping via gradient or radical gradient, knowledge gradient and noisy EI under observation noise, UCB/LCB, target-value, or threshold-finding criteria.
gpcam_version: "8.4.x"
fvgp_version: "4.8.x"
last_verified: "2026-07-28 (gpCAM 3d576fa)"
---

# Skill: gpCAM Acquisition Functions

*Verified against gpCAM 8.4.x / fvgp 4.8.x — last checked 2026-07-28 (gpCAM `3d576fa`).*

Choose — and where needed, write — the acquisition function that controls where gpCAM measures next.

## When to Use

Any time an experiment script needs an acquisition function. That includes simply
picking a built-in string, not only writing a custom callable:
- Deciding what to pass to `ask(acquisition_function=...)`
- Balancing exploration and exploitation
- Multi-objective optimization
- Constrained search (avoid forbidden regions)
- Cost-aware acquisition (expensive moves)
- Upper/lower confidence bounds
- Probability of improvement

## Choose the Acquisition Function *With* the User

**Never pick an acquisition function silently.** It is the single choice that most
determines where the instrument actually goes, and the right answer depends on what
the scientist wants to walk away with — which is rarely what their first sentence says.

Before generating a script:

1. **Name your recommendation and the one tradeoff it makes.** "I'd use `variance`,
   which will give you an even-quality map but won't spend extra shots refining the
   peak once it finds it."
2. **Resolve the ambiguity in "find the best conditions."** That phrase covers two
   different experiments. Ask which one:
   - *the single best point* — they will set the instrument there and run production;
     the rest of the space is disposable → exploitation-leaning (`ucb`, EI late)
   - *a trustworthy map that contains the best point* — they need to see the landscape,
     report it, or find secondary optima → exploration-leaning (`variance`, gradient
     forms, `relative information entropy`)
3. **Ask what they'd do with a wrong answer.** If over-exploiting a noise spike would
   waste a shift of beam time, weight toward exploration.
4. **Confirm before writing the loop**, and put the reasoning in a comment in the
   generated script so the choice is revisitable.

Do **not** default to `"expected improvement"`. It is the most famous acquisition
function in the literature and the most over-applied in practice; in gpCAM it also
carries behavior that surprises people — see
[gpCAM-Specific Behavior](#gpcam-specific-behavior-you-must-know) below.

### Decision Table — start from what they want to *learn*

| They say… | Start from | Tradeoff to state out loud |
|---|---|---|
| "map the whole space" / "replace my raster scan" | `"variance"` | even posterior confidence everywhere; won't refine the peak |
| "find where the signal *changes*" / "find the edges, the transition" | `radical_gradient` callable (below), or `"gradient"` | targets steep regions; ignores flat regions even if unmeasured |
| "find the single best point, I have limited time" | `"ucb"` | exploits toward the max; `beta` is fixed at 3.0 for the built-in string |
| "find the best point, but **my measurements are noisy**" | `"noisy expected improvement"` | the correct EI under noise; costs more per evaluation than `"ucb"` |
| "it keeps re-measuring the same spot" / "EI has gone flat" | `"knowledge gradient"` | lookahead — values a point by how much it would improve your best *decision*, so it keeps exploring where EI has stalled; the most expensive built-in |
| "find where the signal crosses X" | `threshold_finder` callable (below) | concentrates on one level set; the rest of the space stays coarse |
| "find conditions where output ≈ X" | `"target probability"` + `args={'a':lo,'b':hi}` | same — a band, not a global map |
| "**minimize** this" | `"lcb"`, or a custom EI-for-minimization | **not** EI, PI, KG, NEI, or `"ucb"` — all are hardcoded for maximization |
| "give me N points per round" (batch) | `"total correlation"` or `"relative information entropy set"` | anything else is silently overridden — see caveat (e) |
| multi-task / spectra — *exploring* | `"relative information entropy set"` or `"variance"` | — |
| multi-task / spectra — *optimizing* | `"knowledge gradient"` or `"noisy expected improvement"` | both act on the task-summed objective; **not** plain EI — see caveat (d) |
| "I already know roughly where the peak is, refine it" | `"expected improvement"` if noise is low, `"noisy expected improvement"` if not | greedy; plain EI assumes the incumbent is known exactly |
| "I don't know yet" | `"variance"` for the first ~20 points, then revisit | you can change the acquisition between `ask()` calls — it is not fixed for the run |

That last row is worth saying to the scientist explicitly: **the acquisition function
is an argument to `ask()`, not a property of the GP.** A common good design is to
explore first and switch to an exploitative acquisition once the posterior is
informative.

For worked examples of each archetype — how the scientist phrases it, what to ask back,
and the resulting loop — see
[references/choosing-an-acquisition-function.md](references/choosing-an-acquisition-function.md).

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

| Name | String key | Best for | Avoid when |
|------|-----------|----------|------------|
| Variance | `"variance"` | Pure exploration / mapping | They need the peak refined, not the map |
| Expected Improvement | `"expected improvement"` | Late refinement of a known maximum, low noise, sequential, single-task | Minimizing; noisy data; batch (`n>1`); multi-task; early in a run (it degenerates — caveat a) |
| Probability of Improvement | `"probability of improvement"` | Risk-averse maximization | Minimizing; it is even greedier than EI and stalls easily |
| Upper Confidence Bound | `"ucb"` | Maximization; the most reliable general-purpose optimizer here | Minimizing; you need `beta ≠ 3.0` (write the callable — caveat f) |
| Lower Confidence Bound | `"lcb"` | Minimization | You need `beta ≠ 3.0` (write the callable) |
| Predicted Maximum | `"maximum"` | Pure exploitation — mean only, no uncertainty | Almost always: with no uncertainty term it re-measures the same point forever |
| Predicted Minimum | `"minimum"` | Pure exploitation for minimization | Same |
| Gradient | `"gradient"` | Seek steepest regions of the posterior mean | Signal is flat but interesting; you want uniform coverage |
| Target Probability | `"target probability"` | Find points with output near a target value | `args={'a':…, 'b':…}` not set — it raises |
| Relative Information Entropy | `"relative information entropy"` | Information-theoretic exploration | Speed matters — forces `vectorized=False` |
| RIE Set | `"relative information entropy set"` | Batch acquisition; multi-task exploration | — |
| Total Correlation | `"total correlation"` | Batch acquisition | Speed matters — forces `vectorized=False` |
| Knowledge Gradient | `"knowledge gradient"` | Maximization; lookahead, one-step-optimal, robust to noise; keeps exploring when EI stalls | Minimizing; large datasets or tight acquisition budgets — it loops per candidate over a joint posterior |
| Noisy Expected Improvement | `"noisy expected improvement"` | Maximization on noisy measurements — the right EI when the incumbent isn't known exactly | Minimizing; noiseless data (plain EI is cheaper); tight compute budgets |

To sanity-check any built-in or custom acquisition on a grid of candidates without calling `ask()`:
```python
scores = gpo.evaluate_acquisition_function(x_grid, acquisition_function="ucb")
```

Plotting these scores over a 1-D or 2-D grid next to the posterior mean is the fastest
way to show a scientist *why* one acquisition sends the instrument somewhere different
from another. Do this when they're unsure which to pick.

## gpCAM-Specific Behavior You Must Know

These are properties of gpCAM's implementations, not of the textbook definitions. Tell
the scientist about any that apply to their choice — several will otherwise silently do
something other than what they were told.

**(a) EI degenerates to pure exploration early in a run.**
`gpcam/surrogate_model.py:260` clips the improvement *before* forming the ratio:

```python
a = (m - last_best).reshape(len(x))
a[a < 0.] = 0.              # clipped BEFORE gamma, unlike textbook EI
gamma = a / (std + 1e-9)
return std * (gamma * cdf + pdf)
```

Textbook EI clips after forming `z = (m - y_best)/σ`. Here, at every candidate whose
posterior mean sits **below** the incumbent, `gamma == 0` and the score collapses to
`σ · φ(0) = 0.3989 · σ` — that is the `"variance"` acquisition up to a positive
constant. Until the posterior mean somewhere exceeds the best observed value, gpCAM's
EI *is* pure exploration. Never tell a scientist "EI will exploit toward the maximum"
without this caveat. Reported upstream as
[lbl-camera/gpCAM#55](https://github.com/lbl-camera/gpCAM/issues/55) — if that is fixed,
update this section. `"knowledge gradient"` and `"noisy expected improvement"` do not
have this problem.

Verify it on their own model in three lines — on a 15-point fit of `sin(x)` over
`[0, 10]`, 193 of 200 grid candidates sit below the incumbent and the ratio is exactly
`0.398942` at every one of them:

```python
ei  = gpo.evaluate_acquisition_function(x_grid, acquisition_function="expected improvement")
var = gpo.evaluate_acquisition_function(x_grid, acquisition_function="variance")
below = gpo.posterior_mean(x_grid)["m(x)"] < np.max(gpo.y_data)
print(np.unique(np.round(ei[below] / var[below], 6)))   # -> [0.398942] == norm.pdf(0)
```

(The public `gpo.evaluate_acquisition_function` returns the higher-is-better score —
it flips the internal sign back at `gp_optimizer_base.py:297`. Don't negate it again.)

**(b) EI, PI, KG, and NEI are all maximization-only.** `last_best = np.max(gpo.y_data)`
is hardcoded for EI/PI (`surrogate_model.py:251, 258`), and KG/NEI take
`np.max(mu_ref)` over the reference set. If the scientist is minimizing, every one of
them produces meaningless behavior with no error. Use `"lcb"` or the
`expected_improvement_minimize` callable below.

**(c) Plain EI/PI anchor to a noisy observation.** `np.max(y_data)` is the largest
*measured* value, not the posterior mean at that location, so on noisy data EI
over-exploits around a noise spike. **This is what `"noisy expected improvement"` is
for** — it integrates over the posterior of the incumbent instead of treating it as
known. Prefer NEI (or `"knowledge gradient"`) on noisy measurements; reach for `"ucb"`
only if you also need the cheaper evaluation.

**(d) Multi-task *plain* EI is dimensionally incoherent.** The `x_out` branch
(`surrogate_model.py:319`) sums the posterior mean across tasks and *sums* the standard
deviations across tasks, then compares that sum against `np.max(gpo.y_data)` of a
scalar observation. Summed std is not the std of the sum. Do not recommend plain EI for
`fvGPOptimizer` — use `"knowledge gradient"` or `"noisy expected improvement"`, which
scalarize the same task-summed objective but build the joint posterior properly via
`_scalarized_blocks`.

**(e) `ask(n>1)` silently rewrites string acquisitions.** With an `np.ndarray`
`input_set`, `n > 1`, `method != "hgdl"`, and a string acquisition,
`gpcam/gp_optimizer_base.py:528-536` replaces whatever was requested with
`"total correlation"` (a warning is emitted). Batch EI is not EI — and neither is batch
KG or batch NEI. Either use a genuinely batch-aware acquisition, or pass a callable and
`method="hgdl"` with a Dask client.

**(f) `"ucb"` and `"lcb"` hardcode `beta = 3.0`** (`surrogate_model.py:232, 236`). The
strings expose no tuning. To change the exploration/exploitation balance you must pass
the callable form below.

**(g) KG and NEI are the expensive ones.** Both build a joint posterior over the
reference set plus candidates on every evaluation, and KG loops in Python over
candidates. On a large dataset or a tight acquisition budget this dominates the loop.
Keep `kg_reference_set_size` / `nei_reference_set_size` modest (default 100, which
subsamples the data), and use `method="global"` — the `"local"` finite-difference
optimizer is weak on their flat score surfaces, exactly as with plain EI.

### Knowledge gradient and noisy expected improvement

Both are optimization acquisitions (find the maximum) that reason about the posterior
of the underlying **function** rather than the raw noisy observations. They are the
right choice over plain `"expected improvement"` when measurements are noisy, and
they work for both single-task and multi-task GPs.

```python
# Single-task
gpo.ask(bounds, acquisition_function="knowledge gradient")
gpo.ask(bounds, acquisition_function="noisy expected improvement")

# Multi-task — both act on the task-summed objective sum_t f(x, t),
# matching the built-in multi-task "expected improvement" / "ucb"
gpo.ask(bounds, x_out=np.array([0, 1, 2]),
        acquisition_function="knowledge gradient")
```

- **Knowledge gradient (KG)** — the expected increase in the maximum of the posterior
  *mean* after a hypothetical measurement at the candidate (KGCP scheme: the reference
  set is the data points plus the candidate). It is a one-step-optimal, lookahead
  criterion: it values a measurement by how much it is expected to improve your best
  decision, so it keeps exploring even when simple EI has gone to zero. KG is ~0 on
  top of existing data and largest where a measurement would most raise the achievable
  maximum.
- **Noisy expected improvement (NEI)** — expected improvement averaged over the
  posterior of the incumbent (the best value so far). Plain EI treats the incumbent as
  known exactly, which is wrong under observation noise; NEI integrates over it
  (Monte-Carlo over posterior samples of the objective at the observed points, with
  common random numbers so the score is smooth for the optimizer).

Both accept optional tuning through the constructor `args` dict:

```python
gpo = GPOptimizer(x_data, y_data, args={
    "kg_reference_set_size": 100,   # cap on reference points (subsampled if larger)
    "kg_seed": 0,                   # seed for the reference-set subsample
    "nei_samples": 128,             # Monte-Carlo incumbent samples for NEI
    "nei_reference_set_size": 100,
    "nei_seed": 0,                  # seed -> common random numbers, smooth objective
})
```

Cost note: both are substantially heavier than `"variance"` or `"ucb"` — see caveat (g)
above for what to do about it.

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
is hardcoded to maximize (caveat b). Note this version also clips after forming `z`, so
unlike the built-in it does not degenerate to pure exploration early on (caveat a).

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
