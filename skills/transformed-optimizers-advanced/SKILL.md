---
name: transformed-optimizers-advanced
description: Use when observations are constrained — strictly positive (intensities, rates, concentrations) or bounded in [0, 1] (fractions, probabilities). LogGPOptimizer and LogitGPOptimizer fit a GP on transformed observations and push the posterior back through the inverse link, so predictions and credible intervals stay inside the constrained range. Includes how to get raw posterior samples for histograms.
---

# Skill: Transformed-Output GP Optimizers

Use this skill when measurements are guaranteed positive (`y > 0`) or bounded (`y ∈ [0, 1]`). A plain GP doesn't know about these constraints and will happily predict negative intensities or probabilities outside `[0, 1]`. The transformed optimizers fit the GP in an unconstrained "link" space (log or logit) and push the Gaussian posterior back through the inverse link via `evaluate_posterior(x)`, so the original-scale predictions and credible intervals are guaranteed to respect the constraint.

## When to use which

| Observation type | Optimizer | Example domains |
|---|---|---|
| Strictly positive, `y > 0` | `LogGPOptimizer` | Intensities, rates, concentrations, fluxes, lifetimes |
| Bounded, `y ∈ [a, b]` | `LogitGPOptimizer(..., range=(a, b))` | Yields (`[0, 100]%`), probabilities, contrasts, transmittance, normalized intensities |
| Unconstrained / can be negative | plain `GPOptimizer` | Phase shifts, demeaned signals, temperatures in °C |

Both transformed classes are drop-in replacements for `GPOptimizer` in the single-task scalar case — the constructor, `train`, `ask`, `tell`, `optimize`, kernel / mean / noise hooks, and pickling are inherited unchanged. The transform is invisible to the rest of the workflow.

## Quick start

```python
import numpy as np
from gpcam import LogGPOptimizer

# y_data must be > 0 — LogGPOptimizer raises ValueError otherwise
gpo = LogGPOptimizer(x_data, y_data)
gpo.train(hyperparameter_bounds=hp_bounds)

post = gpo.evaluate_posterior(x_grid)
# post["median"]                  -> exact (= exp of the latent GP mean)
# post["mean"]                    -> closed-form lognormal mean
# post["lower"], post["upper"]    -> exact 95% credible band; both > 0
```

```python
from gpcam import LogitGPOptimizer

# y_data must be in [0, 1]; boundary values are clipped to [eps, 1-eps] with a warning
gpo = LogitGPOptimizer(x_data, y_data, eps=1e-6, n_samples=10000)
gpo.train(hyperparameter_bounds=hp_bounds)

post = gpo.evaluate_posterior(x_grid)
# Same dict shape; mean/std are Monte-Carlo (logistic-normal has no closed form).
# All entries lie strictly inside (0, 1).
```

For observations bounded in an arbitrary closed interval `[a, b]` (yield in `[0, 100]%`,
transmittance in `[0, 1]`, an angle in `[0, 90]`, …), pass `range=(a, b)`. The data is
linearly rescaled to `[0, 1]` before the logit transform, posterior outputs are mapped
back to `[a, b]`, and predictions / credible bands stay strictly inside `(a, b)`:

```python
gpo = LogitGPOptimizer(x_data, y_data, range=(0.0, 100.0))   # yield in [0, 100]%
post = gpo.evaluate_posterior(x_grid)
# post["median"], post["lower"], post["upper"], post["samples"]  -- all in (0, 100).
```

## `evaluate_posterior` — the original-scale accessor

The inherited `posterior_mean(x)` / `posterior_covariance(x)` operate in the **transformed** (latent) space. Use `evaluate_posterior(x)` whenever you want the posterior on the **original** scale:

```python
post = gpo.evaluate_posterior(x, level=0.95)
# returns: {"median", "mean", "std", "lower", "upper", "level"}
```

Because both links are monotone increasing, the **median** and the credible **interval** transform exactly. The **mean/std** are exact closed forms for `LogGPOptimizer` (lognormal) and Monte-Carlo estimates for `LogitGPOptimizer`.

### Raw posterior samples (for histograms or custom quantities)

Pass `return_samples=True` to also get an array of original-scale posterior draws of shape `(n_points, n_samples)`:

```python
post = gpo.evaluate_posterior(x_query, return_samples=True, n_samples=8000)
samples = post["samples"]            # shape (len(x_query), 8000)

# Histogram at the first query point:
import matplotlib.pyplot as plt
plt.hist(samples[0], bins=50, density=True)
```

Distributions are: Gaussian for plain `GPOptimizer`, lognormal for `LogGPOptimizer`, logistic-normal for `LogitGPOptimizer`. Use samples to compute expectations of arbitrary functions: `np.mean(g(samples), axis=1)`.

## `LogitGPOptimizer` knobs

- `range` (default `(0.0, 1.0)`): `(lower, upper)` bounds of the observation domain. Data is linearly rescaled to `[0, 1]` before the logit transform; predictions are mapped back to `[lower, upper]`. Raises `ValueError` if `lower >= upper`.
- `eps` (default `1e-6`): clipping margin (in the rescaled `[0, 1]` space) so `logit(0)` / `logit(1)` don't blow up. Increase (e.g. `1e-4`) for noisy boundary data at the cost of small bias; decrease for cleaner data.
- `n_samples` (default `10000`): MC sample count for the closed-form-less mean / std (also the default for `return_samples=True` when no explicit `n_samples` is passed to `evaluate_posterior`). Higher is more accurate but slower.

## Acquisition functions on transformed data

`ask()` and the inherited acquisition functions all run on the **transformed** GP. Both `log` and `logit` are monotone increasing, so ranking acquisitions (`"variance"`, `"ucb"`, `"lcb"`, `"maximum"`, `"minimum"`) still pick the same locations as on the original scale.

Two cases that need care:

- **`"target probability"`**: pass already-transformed bounds. For `LogGPOptimizer`, use `args={"a": np.log(a), "b": np.log(b)}`. For `LogitGPOptimizer`, use `args={"a": scipy.special.logit(a), "b": scipy.special.logit(b)}`.
- **`"expected improvement"` / `"probability of improvement"`**: these compare against the best observed transformed value, which is also the best original value, so semantics carry over. The score *magnitude* differs from a plain GP fit but the *ranking* is consistent.

## What `get_data()` returns

`get_data()["y data"]` is the **transformed** y stored on the GP. The transformed optimizers add a `"original y data"` key containing the inverse-transformed values for convenience:

```python
data = gpo.get_data()
data["y data"]            # log(y) or logit(y)  -- in the GP's modeling space
data["original y data"]   # y  -- the observations as you provided them
```

## Noise variances

When you provide `noise_variances` (observation noise) at construction or via `tell()`, they are transformed by the delta method (`var_z ≈ (g'(y))² · var_y`) before reaching the latent GP. **Provide noise in the original scale**; the optimizer handles the transform.

## Pickling

`LogGPOptimizer` and `LogitGPOptimizer` pickle the same way as `GPOptimizer`. `LogitGPOptimizer`'s extra attributes (`eps`, `n_samples`) survive the round-trip.

## Example notebook

See [`examples/GPOptimizer_LogAndLogit.ipynb`](../../examples/GPOptimizer_LogAndLogit.ipynb) for a side-by-side visual tour — latent-space GP, back-transformed band, plain-GP comparison that violates the constraint, and posterior-sample histograms (lognormal vs. logistic-normal).
