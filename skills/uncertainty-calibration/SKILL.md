---
name: uncertainty-calibration
description: Use to validate that a gpCAM posterior's error bars are trustworthy before publishing them or letting an autonomous loop act on them — scoring rules (NLPD, CRPS, MSLL), calibration/coverage curves, PICP and interval width, and what to do when the model is over- or under-confident.
gpcam_version: "8.4.x"
fvgp_version: "4.8.x"
last_verified: "2026-07-23 (gpCAM dadeb65)"
---

# Skill: Validating gpCAM Uncertainty

*Verified against gpCAM 8.4.x / fvgp 4.8.x — last checked 2026-07-23 (gpCAM `dadeb65`).*

A GP's headline product is not the prediction — it is the *uncertainty on* the
prediction. An autonomous loop acts on that uncertainty at every step: `"variance"`
goes where the model claims ignorance, `"ucb"` and `"expected improvement"` weigh
mean against spread, and any convergence rule that stops when variance falls below a
threshold is trusting the error bar completely. **An overconfident GP will stop
early, under-sample the interesting region, and report a wrong error bar with
conviction.**

This skill is about checking that the error bars are honest. Run it after training
and before believing any result.

## When to Use

- Before publishing a posterior mean with error bars
- Before trusting a variance-based stopping criterion
- When the fit "looks fine" but the loop keeps re-sampling the same spot (or refuses to)
- When comparing kernels or noise models — RMSE alone cannot tell you which is better calibrated
- After changing the noise model, since that is what sets the error bars

## The One-Minute Check

```python
import numpy as np

# Hold out data the GP has never seen. Random split, not the tail of the loop —
# adaptively-acquired points are not i.i.d. and will flatter the model.
print(f"RMSE  {gpo.rmse(x_test, y_test):.4f}")     # accuracy only
print(f"NLPD  {gpo.nlpd(x_test, y_test):.4f}")     # accuracy + calibration (lower better)
print(f"PICP  {gpo.picp(x_test, y_test, interval=0.95):.3f}   (target 0.95)")
print(f"MPIW  {gpo.mpiw(x_test, interval=0.95):.4f}")  # average width of that interval
```

If `PICP ≈ 0.95` and `NLPD` is small, the posterior is roughly trustworthy. If
`PICP` is far from `0.95`, stop and read the rest of this skill — nothing downstream
is reliable.

**Shapes must match exactly.** `rmse`, `nlpd`, and `crps` assert
`posterior_mean(x_test).shape == y_test.shape`. For a single-task GP, `y_test` must
be shape `(V,)`, not `(V, 1)`.

## The Metrics

| Method | What it measures | Reading it |
|---|---|---|
| `rmse(x_test, y_test)` | Accuracy of the mean only | Lower better. **Blind to uncertainty** — a model with absurd error bars can have excellent RMSE. |
| `nrmse(x_test, y_test)` | RMSE / range(y) | Dimensionless; use to compare across datasets. |
| `mae` / `mape` | Mean absolute / percentage error | Robust to outliers; `mape` breaks near `y = 0`. |
| `r2(x_test, y_test)` | Fraction of variance explained | 1 is perfect, 0 is no better than predicting the mean. |
| **`nlpd(x_test, y_test)`** | **Negative log predictive density** | **The primary scoring rule.** Penalizes both wrong means and wrong spreads. Lower is better; can be negative. Use this to choose between models. |
| `crps(x_test, y_test)` | Continuous ranked probability score | **Returns a tuple `(mean, std)`** — unpack it. Lower better, in the units of `y`, and less brutal than NLPD toward badly-underestimated variances. |
| `msll(x_test, y_test)` | Mean standardized log loss | NLPD relative to a trivial Gaussian baseline fitted to the training data. **Negative means you beat the baseline**; ≈ 0 means the GP learned nothing useful. |
| `picp(x_test, y_test, interval=q)` | Prediction Interval Coverage Probability — fraction of test points inside the `q` credible interval | Compare against `q` itself. This is the calibration workhorse. |
| `mpiw(x_test, interval=q)` | Mean Prediction Interval Width | Only meaningful **alongside** `picp`: an interval can always reach its coverage by growing uselessly wide. |
| `interval_score(x_test, y_test, interval=q)` | Coverage and width in one number | Lower better. Penalizes both misses and excessive width — the right single metric when tuning for interval quality. |
| `coverage_curve(x_test, y_test, intervals=None)` | `picp` across many levels | The full calibration picture. See below. |

`crps` returning two values is a common trip-up:

```python
crps_mean, crps_std = gpo.crps(x_test, y_test)     # NOT a single float
```

## Calibration Curves — the informative check

A single `PICP@95` compresses the whole calibration story into one number. The
coverage curve evaluates `picp` across many levels at once and shows you *how* the
model is wrong, not just *that* it is:

```python
cc = gpo.coverage_curve(x_test, y_test)     # default: 19 levels from 0.05 to 0.95
target = np.array(cc["target_coverage"])
measured = np.array(cc["measured_coverage"])

for t, m in zip(target, measured):
    print(f"  target {t:.2f} -> measured {m:.3f}  {'over-confident' if m < t else 'under-confident'}")

# Single-number summary: mean absolute calibration error
print(f"Calibration error: {np.mean(np.abs(measured - target)):.4f}")
```

Plot `measured` against `target`:

```python
import matplotlib.pyplot as plt
plt.plot(target, measured, "o-", label="model")
plt.plot([0, 1], [0, 1], "k--", label="perfect calibration")
plt.xlabel("target coverage"); plt.ylabel("measured coverage"); plt.legend()
```

| Curve shape | Meaning | Consequence in an autonomous loop |
|---|---|---|
| On the diagonal | Well calibrated | Error bars mean what they say |
| **Below** the diagonal | **Over-confident** — intervals too narrow | Loop stops early, under-explores, publishes error bars that are too small. **The dangerous failure.** |
| **Above** the diagonal | **Under-confident** — intervals too wide | Loop over-explores and wastes measurements; conclusions are safe but inefficient |
| S-shaped / crossing | Mis-specified shape, not just scale | Usually a wrong noise model or a non-Gaussian likelihood — a transformed optimizer may be the fix |

A worked example of what over-confidence looks like: understating the noise variance
by 100× on an otherwise identical model moved `PICP@95` from 0.95 to 0.88 and `NLPD`
from 0.23 to 4.21. RMSE barely moved. **This is exactly why RMSE cannot be your only
check.**

## Diagnosing and Fixing

### Over-confident (measured coverage below target)

Most common, most dangerous. In rough order of likelihood:

1. **Understated observation noise.** You passed `noise_variances` that are too small,
   or your noise function's hyperparameter hit its lower bound. Check:
   ```python
   print(gpo.get_data()["measurement variances"][:5])
   print(gpo.hyperparameters)     # is a noise hp pinned at its bound?
   ```
   Fix: raise the noise, or switch from fixed `noise_variances` to a learnable
   `noise_function` so the data can speak. See the `noise-functions` skill.

2. **Length scale too short.** The GP fits every wiggle, treats the residual as
   signal, and leaves no uncertainty between points. Check whether a length scale is
   sitting at its lower bound; raise that bound.

3. **Wrong noise structure.** Heteroscedastic data fitted with constant noise is
   over-confident in the noisy region and under-confident in the quiet one — this
   produces the S-shaped curve. Fix with a position-dependent `noise_function`.

4. **Test set is not independent.** Points chosen by `ask()` cluster where the model
   was uncertain, so scoring on them is not a fair test. Hold out a random subset
   before the loop starts.

### Under-confident (measured coverage above target)

Less harmful but wasteful:

1. **Overstated noise** — lower `noise_variances`, or lower the noise hyperparameter's
   upper bound.
2. **Length scale too long** — the GP is underfitting and attributing real structure
   to noise. Check `r2`; if it is poor *and* the model is under-confident, it is
   underfitting.
3. **Signal variance bound too high** — an inflated `hps[0]` inflates every interval.

### Both, in different regions

Use `mpiw` on subsets to localize it, or plot residuals against predicted std:

```python
mean = gpo.posterior_mean(x_test)["m(x)"]
std = np.sqrt(gpo.posterior_covariance(x_test, variance_only=True)["v(x)"])
z = (y_test - mean) / std          # should be ~ N(0, 1) if well calibrated

print(f"z mean {z.mean():+.3f} (want 0)   z std {z.std():.3f} (want 1)")
plt.hist(z, bins=30, density=True)
```
`z.std() > 1` means over-confident, `< 1` means under-confident, and a non-zero
`z.mean()` means the prior mean is biased — see the `prior-mean-functions` skill.
This standardized-residual histogram is the analogue of a PIT histogram for a
Gaussian posterior, and it localizes problems that a single scalar hides.

## Constrained Observations

If `y` is strictly positive or bounded, a plain GP is often *structurally*
mis-calibrated near the boundary: it puts probability mass on impossible values, so
lower credible bounds go negative and coverage is wrong exactly where the data is
most interesting.

Check calibration on the **original** scale using `evaluate_posterior`, not the
latent-space metrics:

```python
post = gpo.evaluate_posterior(x_test, level=0.95)
covered = (y_test >= post["lower"]) & (y_test <= post["upper"])
print(f"original-scale PICP@95: {covered.mean():.3f}")
```

The inherited `picp` / `nlpd` / `coverage_curve` operate in the GP's latent space,
which is the correct check for the *latent* model but not the number you report to a
collaborator. If latent calibration is good but original-scale coverage is poor, the
transform is the problem. See the `transformed-optimizers-advanced` skill.

## Validation Inside an Autonomous Loop

Track calibration as data accumulates — it will drift, especially early:

```python
holdout_x, holdout_y = ...      # reserved BEFORE the loop, never told to the GP
history = []

for i in range(N_ITERATIONS):
    ...                          # ask / measure / tell
    if (i + 1) % RETRAIN_EVERY == 0:
        gpo.train(hyperparameter_bounds=hp_bounds, method="local")
        history.append({
            "n": len(gpo.y_data),
            "rmse": gpo.rmse(holdout_x, holdout_y),
            "nlpd": gpo.nlpd(holdout_x, holdout_y),
            "picp95": gpo.picp(holdout_x, holdout_y, interval=0.95),
        })
        print(history[-1])
```

Expect `rmse` and `nlpd` to fall and `picp95` to settle near 0.95. If `picp95` drifts
steadily *down* as points accumulate, the hyperparameters are overfitting the growing
dataset — retrain globally rather than locally, or tighten the length-scale lower
bound.

Never score against points the GP has been told; `rmse` on training data measures
interpolation, not prediction, and will look excellent regardless.

## Quick Visual Check

```python
gpo.plot_observed_vs_predicted(x_test, y_test)   # needs matplotlib
```
Points should fall on the diagonal with scatter consistent with the error bars.

## Common Pitfalls

1. **Reporting RMSE alone.** It says nothing about the error bars. Pair it with `nlpd` and `picp` at minimum.
2. **Unpacking `crps` as a scalar.** It returns `(mean, std)`.
3. **`picp` without `mpiw`.** Coverage is trivially achievable with absurdly wide intervals; always report both, or use `interval_score`, which combines them.
4. **Testing on adaptively-acquired points.** They are not i.i.d.; hold out a random subset before the loop starts.
5. **Shape mismatch.** `y_test` must be `(V,)` for a single-task GP — the assertion message tells you the expected shape.
6. **Judging a transformed optimizer by latent-space metrics.** Report original-scale coverage via `evaluate_posterior`.
7. **Calibrating once and never again.** Calibration drifts as the loop adds data and retrains.

## Reference

- `gpcam.GPOptimizer` inherits all of these from `fvgp.GP` — see the fvGP source for exact formulas
- `experiment-designer` skill — where to put these checks in a generated script
- `noise-functions` skill — the noise model is usually what calibration is really testing
- `transformed-optimizers-advanced` skill — for constrained observations
