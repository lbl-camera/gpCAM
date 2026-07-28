---
name: acquisition-functions
description: Use when choosing or designing the acquisition function for a gpCAM experiment — deciding which built-in actually fits the scientist's goal (and confirming it with them rather than defaulting to expected improvement), or writing a custom one for exploration/exploitation balance, multi-objective targets, constrained search regions, cost-aware moves, change-mapping via gradient or radical gradient, UCB/LCB, target-value, or threshold-finding criteria.
---

# Skill: gpCAM Acquisition Functions

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
| "find where the signal crosses X" | `threshold_finder` callable (below) | concentrates on one level set; the rest of the space stays coarse |
| "find conditions where output ≈ X" | `"target probability"` + `args={'a':lo,'b':hi}` | same — a band, not a global map |
| "**minimize** this" | `"lcb"`, or a custom EI-for-minimization | **not** EI or `"ucb"` — both are hardcoded for maximization |
| "give me N points per round" (batch) | `"total correlation"` or `"relative information entropy set"` | anything else is silently overridden — see caveat (e) |
| multi-task / spectra / vector output | `"relative information entropy set"` or `"variance"` | **not** EI — see caveat (d) |
| "I already know roughly where the peak is, refine it" | `"expected improvement"` | greedy; assumes low noise, sequential, single-task |
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
`gpcam/surrogate_model.py:254` clips the improvement *before* forming the ratio:

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
without this caveat.

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

**(b) EI and PI are maximization-only.** `last_best = np.max(gpo.y_data)` is hardcoded
(`surrogate_model.py:250, 257`). If the scientist is minimizing, the built-in strings
produce meaningless behavior with no error. Use `"lcb"` or the
`expected_improvement_minimize` callable below.

**(c) The EI/PI incumbent is a noisy observation.** `np.max(y_data)` is the largest
*measured* value, not the posterior mean at that location. On noisy data EI anchors to
a noise spike and over-exploits a region that isn't actually best. Prefer `"ucb"` when
the measurement is noisy, or write a callable using
`np.max(gpo.posterior_mean(gpo.x_data)["m(x)"])` as the incumbent.

**(d) Multi-task EI is dimensionally incoherent.** The `x_out` branch
(`surrogate_model.py:306`) sums the posterior mean across tasks and *sums* the standard
deviations across tasks, then compares that sum against `np.max(gpo.y_data)` of a
scalar observation. Do not recommend EI for `fvGPOptimizer`.

**(e) `ask(n>1)` silently rewrites string acquisitions.** With an `np.ndarray`
`input_set`, `n > 1`, `method != "hgdl"`, and a string acquisition,
`gpcam/gp_optimizer_base.py:517-525` replaces whatever was requested with
`"total correlation"` (a warning is emitted). Batch EI is not EI. Either use a
genuinely batch-aware acquisition, or pass a callable and `method="hgdl"` with a Dask
client.

**(f) `"ucb"` and `"lcb"` hardcode `beta = 3.0`** (`surrogate_model.py:231, 235`). The
strings expose no tuning. To change the exploration/exploitation balance you must pass
the callable form below.

**(g) Transformed optimizers work in transformed space.** For `LogGPOptimizer` /
`LogitGPOptimizer`, `ask()` optimizes in log- or logit-space. Ranking acquisitions
(`variance`, `ucb`, `lcb`, `maximum`, `minimum`, EI, PI) still identify the same
locations because the transforms are monotone. But `"target probability"` bounds must
be given already transformed (`np.log(a)`, `np.log(b)`).

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
```python
def multi_objective(x, gpo):
    """Balance finding the max with reducing uncertainty."""
    mean = gpo.posterior_mean(x)["m(x)"]
    var = gpo.posterior_covariance(x, variance_only=True)["v(x)"]
    
    w_exploit = 0.7  # weight on exploitation
    w_explore = 0.3  # weight on exploration
    
    # Normalize each component to [0, 1]
    mean_norm = (mean - mean.min()) / (mean.max() - mean.min() + 1e-10)
    var_norm = (var - var.min()) / (var.max() - var.min() + 1e-10)
    
    return w_exploit * mean_norm + w_explore * var_norm
```

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
    """Find the boundary where f(x) = threshold."""
    threshold = 0.5  # EDIT THIS
    
    mean = gpo.posterior_mean(x)["m(x)"]
    var = gpo.posterior_covariance(x, variance_only=True)["v(x)"]
    std = np.sqrt(np.maximum(var, 1e-10))
    
    # Score is high near the threshold AND where uncertainty is high
    distance_to_threshold = np.abs(mean - threshold)
    return std / (distance_to_threshold + 0.01)
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

- If you access `gpo.y_data` directly (e.g., for `y_best`), make sure it's up to date after `tell()`
- The GP must be trained before acquisition makes sense — always call `train()` first
- For `variance_only=True`: faster, returns just diagonal variances (usually what you want)
- For full covariance: use `variance_only=False` but this is O(V²) memory

## Common Pitfalls

1. **Reaching for `"expected improvement"` by reflex.** It is the best-known acquisition
   function, not the best-suited one. Work the choice out with the scientist using the
   decision table above, and check it against the caveats before recommending it.
2. **Recommending a maximization acquisition to someone minimizing.** `"expected
   improvement"`, `"probability of improvement"`, `"ucb"`, and `"maximum"` all assume
   bigger-is-better. Confirm the direction explicitly — scientists often say "optimize"
   for both.
3. **Returning negative scores for points you want**: Remember, acquisition is MAXIMIZED.
4. **Division by zero in std**: Always use `np.maximum(var, 1e-10)` before taking sqrt.
5. **Not handling edge cases**: Early in the loop with few points, the GP posterior can be unreliable.
6. **Expensive acquisition functions**: They're evaluated many times during optimization. Keep them fast.
7. **Assuming the acquisition is fixed for the run.** It's an `ask()` argument. Switching
   from exploration to exploitation partway through is often the right design.
