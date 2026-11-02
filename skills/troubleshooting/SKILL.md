---
name: troubleshooting
description: Use when a gpCAM script crashes or misbehaves — Cholesky/non-positive-definite errors, singular covariance, hyperparameter bounds mismatches, NaNs, kernel signature errors, Dask hangs, pickling failures, or an ask/tell loop that keeps picking the same point. Maps the actual error text to the cause and the fix.
gpcam_version: "8.7.x"
fvgp_version: "4.11.x"
last_verified: "2026-10-28 (gpCAM 8.7.0, a21385a)"
---

# Skill: Troubleshooting gpCAM

*Verified against gpCAM 8.7.0 / fvgp 4.11.1 — last checked 2026-10-28 (gpCAM `a21385a`).*

A decision tree from the error you actually saw to the thing that is actually wrong.
The audience is a scientist who does not want to learn GP internals to get unstuck.

**First, read the exception text.** fvGP's most common errors carry diagnostics and a
list of likely causes inline. The Cholesky failure in particular prints
`min(diag(M))` and `max|M - M.T|`, which usually identifies the problem by itself.

## Fast Triage

| Error text (first line) | Jump to |
|---|---|
| `Cholesky factorization failed: ... not positive definite` | [Non-PD covariance](#non-positive-definite-covariance) |
| `index N is out of bounds for axis 0 with size M` (during `train`) | [Bounds mismatch](#hyperparameter-bounds-mismatch) |
| `NaNs encountered in dataset.` | [NaNs](#nans-in-the-data) |
| `No valid kernel function signature` (or mean / noise) | [Signature](#wrong-callable-signature) |
| `prior covariance K must be 2-d` | [Kernel shape](#kernel-returns-the-wrong-shape) |
| `'GPOptimizer' object has no attribute 'posterior'` | [Uninitialized GP](#gp-not-initialized-yet) |
| `Reading the arguments for acq func 'target probability' failed.` | [Acquisition args](#acquisition-function-arguments) |
| `LogGPOptimizer requires strictly positive observations` | [Constrained data](#constrained-observations) |
| Script hangs / spawns endless processes | [Dask & multiprocessing](#dask-hangs-and-process-storms) |
| `PicklingError` / `Can't pickle local object` | [Pickling](#pickling-and-checkpointing) |
| `returned the hyperparameters it started from` (warning) | [train() changed nothing](#train-returned-and-the-hyperparameters-did-not-change) |
| `they were rejected and the previous hyperparameters kept` (warning) | [train() changed nothing](#train-returned-and-the-hyperparameters-did-not-change) |
| No error, but the loop misbehaves | [Silent problems](#silent-problems-no-exception) |

---

## Non-Positive-Definite Covariance

```
NonPositiveDefiniteError: Cholesky factorization failed: the 12x12 prior covariance
matrix is not positive definite.
Diagnostics: min(diag(M)) = ..., max|M - M.T| = ... (should be ~0).
```

The single most common gpCAM crash. Use the printed diagnostics to pick the branch:

**`max|M - M.T|` is not ~0 → your kernel is not symmetric.**
A kernel must satisfy `k(x1, x2) == k(x2, x1).T`. Check for an accidental asymmetry
such as indexing `x1` where you meant `x2`. Test it directly:
```python
K12 = my_kernel(x1, x2, hps)
K21 = my_kernel(x2, x1, hps)
assert np.allclose(K12, K21.T), "kernel is not symmetric"
```

**`min(diag(M))` is very small or negative → the kernel is not PSD.**
Sums and products of valid kernels are valid; **differences are not**. If you wrote
`k1 - k2`, that is almost certainly the cause. Verify:
```python
K = my_kernel(x_data, x_data, hps)
print("min eigenvalue:", np.linalg.eigvalsh((K + K.T) / 2).min())   # must be >= 0
```
A non-stationary kernel with an input-dependent length scale needs its normalizing
prefactor — see the Gibbs recipe in the `kernel-designer` skill.

**Diagnostics look fine → conditioning.** In order of likelihood:

1. **Duplicate or near-duplicate `x` points** with tiny noise. Two identical rows make
   `K` rank-deficient. This is common when an instrument reports rounded positions, or
   when `ask()` returns a point you already measured.
   ```python
   from scipy.spatial.distance import pdist
   print("closest pair distance:", pdist(gpo.x_data).min())
   ```
   Fix: deduplicate, or ensure noise is not effectively zero (below).

2. **Noise driven to zero.** A learnable noise hyperparameter that hits a lower bound
   of `0` removes the diagonal jitter holding the matrix together.
   ```python
   print(gpo.hyperparameters)                              # any hp pinned at a bound?
   print(gpo.get_data()["measurement variances"][:5])      # are these ~0?
   ```
   Fix: set the noise lower bound to something small but positive (`1e-3`, or
   `1e-6 * var(y)`), never `0`.

3. **Length scale collapsed.** A length scale at its lower bound makes every
   off-diagonal entry ~0 except for near-coincident points. Raise the lower bound.

4. **Genuinely ill-conditioned large problem.** Try `linalg_mode="Chol"` explicitly,
   or for large sparse problems see the `gp2scale-advanced` skill.

---

## Hyperparameter Bounds Mismatch

```
IndexError: index 2 is out of bounds for axis 0 with size 2
```

Raised during `train()` when your kernel / mean / noise function reads a
hyperparameter index that the bounds array does not cover. The bounds array length
*defines* the hyperparameter vector length during training.

```python
# Add this right after construction — it catches the problem at the source:
assert len(gpo.hyperparameters) == len(hp_bounds), (
    f"{len(hp_bounds)} bounds rows but {len(gpo.hyperparameters)} hyperparameters")
```

**The usual root cause is a hardcoded `K = 3`**, which is correct only for `D = 2` with the default single-task ARD kernel. Derive `K` from the input dimensionality (`K = 1 + x.shape[1]` for single-task, or `K = 1 + (D + 1)` in `"cartesian"` multi-task mode) and allocate disjoint slices in order: kernel (`0..K-1`), prior mean (`K..K+M-1`), noise (`K+M..`). See the `experiment-designer` skill ("Hyperparameter Coordination") for the canonical index table.

Related and nastier: if the indices *overlap* rather than run past the end, there is
**no error at all** — two components silently share a hyperparameter. See
[Silent problems](#silent-problems-no-exception).

---

## NaNs in the Data

```
Exception: NaNs encountered in dataset.
```

A `NaN` reached `x_data` or `y_data` — usually a failed measurement, a detector
dropout, or a division by zero in a user callable.

```python
print("NaN in x:", np.isnan(gpo.x_data).any() if gpo.x_data is not None else "n/a")
print("NaN in y:", np.isnan(y_new).any())
```

Fix: filter failed measurements before `tell()`:
```python
ok = np.isfinite(y_new)
if ok.any():
    gpo.tell(x_new[ok], y_new[ok])
```

**Exception — multi-task.** For `GPOptimizer(..., multi_task_mode="cartesian")` (the former
`fvGPOptimizer`), `NaN` in `y_data` is legitimate and
means "this task was not measured at this point". The matching `noise_variances` entry
**must also** be `NaN`. A `NaN` in `y` with a finite noise entry is an error.

---

## Wrong Callable Signature

```
Exception: No valid kernel function signature
```
(or the prior-mean / noise equivalent)

gpCAM dispatches on **parameter count** via `inspect.signature`: kernels take 3 (`f(x1, x2, hps)`) or 4 (`f(x1, x2, hps, args)`) parameters, mean and noise functions take 2 (`f(x, hps)`) or 3 (`f(x, hps, args)`), and cost (`f(origin, x)`) and custom acquisition (`f(x, gp_obj)`) functions always take 2 parameters and never receive `args` (bind extra parameters with `functools.partial`). A default value still counts toward `len(inspect.signature(f).parameters)`, so `def my_kernel(x1, x2, hps, args=None)` is dispatched as a 4-parameter kernel. See the `experiment-designer` skill ("User-function arguments (`args`)") for the canonical dispatch table.

---

## Kernel Returns the Wrong Shape

```
AssertionError: prior covariance K must be 2-d
```

The kernel must return `(len(x1), len(x2))`. Common causes: returning a distance
matrix reduced along an axis, or a `np.sum` without `axis=`, or forgetting that
`x1`/`x2` are always 2-D even for 1-D inputs (`x[:, 0]`, not `x`).

```python
K = my_kernel(x1, x2, hps)
assert K.shape == (len(x1), len(x2)), f"got {K.shape}, expected {(len(x1), len(x2))}"
```

---

## GP Not Initialized Yet

```
AttributeError: 'GPOptimizer' object has no attribute 'posterior'
```

`GPOptimizer()` constructed without `x_data`/`y_data` defers building the underlying
`fvgp.GP` until the first `tell()`. Until then, `posterior_mean`, `train`, `ask`, and
most properties are unavailable.

```python
gpo = GPOptimizer()          # lazy — no GP yet
gpo.tell(x_init, y_init)     # GP is built here
gpo.train(hyperparameter_bounds=hp_bounds)
```

Either pass data to the constructor, or `tell()` before doing anything else. `gpo.gp`
is the boolean that tracks this.

---

## Acquisition Function Arguments

```
Exception: Reading the arguments for acq func `target probability` failed.
```

`"target probability"` needs an interval:
```python
gpo.ask(bounds, acquisition_function="target probability", args={"a": 1.0, "b": 2.0})
```
With a transformed optimizer, pass the bounds **already transformed**
(`np.log(a)` for `LogGPOptimizer`). See `transformed-optimizers-advanced`.

Other acquisition strings that need attention:
- Multi-task (`x_out is not None`) supports a **smaller set** than single-task.
  `"maximum"`, `"minimum"`, `"gradient"`, `"probability of improvement"`, and
  `"target probability"` are single-task only, and `"total correlation"` is not available when `multi_task_mode="kronecker"`.
- A custom callable must return a **1-D array of length `len(x)`**. Returning a scalar
  or a `(V, 1)` column raises further down in the optimizer.

---

## Constrained Observations

```
ValueError: LogGPOptimizer requires strictly positive observations (y > 0).
```

Exactly what it says — a zero or negative value reached a `LogGPOptimizer`. If zeros
are physically meaningful (a genuinely empty detector bin), either add a small floor
appropriate to your instrument, or use a plain `GPOptimizer` and accept that the
posterior can go negative.

`LogitGPOptimizer` clips boundary values to `[eps, 1-eps]` with a warning rather than
raising. If you see that warning a lot, raise `eps` (e.g. `1e-4`) — repeated clipping
at `1e-6` produces enormous latent values and an ill-conditioned GP.

---

## Dask Hangs and Process Storms

If a script spawns endless processes or hangs at startup, the cause is almost always a
missing entrypoint guard. Dask/`multiprocessing` re-imports your module in each worker;
without the guard, module-level code creates another cluster, recursively.

```python
if __name__ == "__main__":       # REQUIRED whenever a Dask client is involved
    main()
```

Other Dask issues:
- **Workers not ready.** Call `client.wait_for_workers(n)` before constructing a
  gp2Scale GP.
- **Two live GPs on one client.** Constructing a second gp2Scale GP on a client that
  still holds a live one triggers scatter race conditions. Use a separate client or
  let the first be garbage-collected.
- **`method="hgdl"` without a client.** HGDL needs `dask_client=`.
- **Async training left running.** `train(asynchronous=True)` returns immediately;
  call `gpo.stop_training(opt_obj)` and `kill_client()` when done, or the process will
  not exit.

---

## Pickling and Checkpointing

```
PicklingError: Can't pickle local object '...'
AttributeError: Can't pickle local object 'make_cost.<locals>.cost'
```

`GPOptimizer` instances are picklable, but **the callables you attached must pickle by
reference**, which means they must be defined at **module level**. A closure or a
lambda defined inside another function cannot pickle — replace it with a module-level function bound via `functools.partial(cost, speed=2.5)` (see the `cost-functions` skill for a complete example).

When reloading saved arrays that contain Python objects, remember
`np.load(path, allow_pickle=True)`.

---

## Silent Problems (No Exception)

The failures that cost the most time, because nothing crashes.

### The loop keeps picking the same point (or a corner)

1. **Hyperparameters are stale.** Retrain more often; a GP with a wildly wrong length
   scale produces a nearly flat acquisition and the optimizer returns an arbitrary
   point — often a bound.
2. **Length scale collapsed** → every unmeasured point looks equally uncertain.
   Check whether a length scale sits at its lower bound and raise that bound.
3. **Cost function dominating.** The acquisition is *divided* by the cost, so a large
   cost pins the optimizer next to `origin`. Compare magnitudes with
   `gpo.evaluate_acquisition_function(x_grid)`.
4. **`n > len(candidates)`** on a discrete set — you get the whole set back.

### The cost function seems to do nothing

`cost_function` is applied **only when `origin is not None`**. You must pass
`ask(..., position=current_position)`. Without it, gpCAM silently ignores the cost.

### Hyperparameters train to nonsense / the likelihood is flat

Two components are reading the same index. fvGP requires the index ranges used by
kernel, mean, and noise to be **disjoint**, and assumes a hyperparameter belonging to
the mean has zero kernel derivative — an overlap corrupts the gradients too. Standard
ordering is **kernel, then mean, then noise**; never read `hps[-1]` in a prior mean.

A related silent case: a deep kernel whose recipe never calls `set_weights` /
`set_biases`. The network keeps its random initialization, the NN hyperparameters have
no effect, and training wanders over an exactly flat likelihood. See the
`kernel-designer` skill.

Test that a hyperparameter actually matters:
```python
h1 = gpo.hyperparameters.copy(); h2 = h1.copy(); h2[i] *= 1.5
assert not np.isclose(gpo.log_likelihood(hyperparameters=h1),
                      gpo.log_likelihood(hyperparameters=h2)), f"hps[{i}] does nothing"
```

### `train()` returned and the hyperparameters did not change

Two different causes, and since fvgp 4.8.8 both announce themselves — read the warning
rather than guessing.

**"returned the hyperparameters it started from"** — `method='bo'` evaluated its whole
design and nothing beat the starting point. That is a real answer, not a crash, and it is
usually right: a marginal likelihood has a narrow ridge along which the signal variance
and the length scales trade off *together*, so every one-at-a-time move off it looks
useless. A space-filling design cannot land on a curve of that shape, and BO's
axis-aligned surrogate cannot follow it. It gets worse fast with the hyperparameter count.
Switch to `method='local'` or `'mcmc'`, which follow such a ridge directly — measured on a
14-hyperparameter problem, BO returned the start at log-likelihood −93.4 while `local`
from the *same* start reached −43.2. Use `bo` only where it earns its keep: an expensive,
noisy, gradient-free likelihood (the gp2Scale/mBCG regime).

Note also that `method='bo'` stops on `bo_args['patience']` (default 10) long before
`max_iter`; raising `max_iter` alone changes nothing.

**"they were rejected and the previous hyperparameters kept"** — `train()` found worse
hyperparameters and rolled back, which `accept_only_if_improved=True` (the default) does
for `method` in `'local'`, `'hgdl'`, `'adam'` or a callable. `'local'` accepts SciPy's
point regardless of its `success` flag, so this fires most often there. It means training
genuinely failed to improve, not that the guard misfired — give it a different
`init_hyperparameters`, wider bounds, or more iterations. Pass
`accept_only_if_improved=False` to take the worse point anyway.

A change of *zero* with no warning at all is a third thing: `method='mcmc'` returns the
posterior median, so repeating it on converged hyperparameters is expected to move little.

### Error bars look wrong

Do not diagnose this by eye — run the checks in the `uncertainty-calibration` skill.
An overconfident model can have excellent RMSE.

### Results change between runs

The default trainer draws its four extra starting points at random (fvgp 4.9:
`method='local'` with `n_restarts=5`), and `method="global"` uses `differential_evolution`;
both use NumPy's global generator, so seed `np.random.seed(...)` for repeatable fits.
`method="mcmc"` takes `mcmc_seed=...` instead, and gives a posterior over hyperparameters
rather than a point estimate (in `gpo.mcmc_info`). If fits differ in *quality*, not just in
the last digits, a single local run is landing in different basins: keep the restarts
(`n_restarts=1` disables them) or raise `n_restarts`. Large run-to-run swings usually mean the
likelihood surface is flat or multi-modal — often too little data for the number of
hyperparameters.

---

## General Debugging Recipe

```python
# 1. Does the kernel behave on its own, before any GP is involved?
K = my_kernel(x_data, x_data, init_hps)
assert K.shape == (len(x_data), len(x_data))
assert np.allclose(K, K.T), "not symmetric"
print("min eigenvalue:", np.linalg.eigvalsh((K + K.T) / 2).min(), "(want >= 0)")

# 2. Do the counts line up?
print(f"bounds rows {len(hp_bounds)}, hyperparameters {len(gpo.hyperparameters)}")

# 3. Is the data clean?
print("x NaN:", np.isnan(x_data).any(), " y NaN:", np.isnan(y_data).any())
from scipy.spatial.distance import pdist
print("closest pair:", pdist(x_data).min(), "(near 0 -> duplicates)")

# 4. Turn on gpCAM's logging for the training loop
from loguru import logger; logger.enable("gpcam")
gpo.train(hyperparameter_bounds=hp_bounds, info=True)

# 5. Are any hyperparameters pinned at their bounds after training?
for i, (h, (lo, hi)) in enumerate(zip(gpo.hyperparameters, hp_bounds)):
    if np.isclose(h, lo) or np.isclose(h, hi):
        print(f"  hps[{i}] = {h:.4g} is pinned at a bound [{lo}, {hi}] -- widen it")
```

Step 5 is worth running routinely: a pinned hyperparameter is the fingerprint of a
bound that is too tight, and it precedes most of the failures above.

## Reference

- `uncertainty-calibration` skill — for "the error bars look wrong"
- `kernel-designer` skill — PSD rules and the deep-kernel pitfall
- `prior-mean-functions` / `noise-functions` skills — the index convention
- [gpCAM documentation](https://gpcam.readthedocs.io)
