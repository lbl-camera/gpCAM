---
name: noise-functions
description: Use when modeling position-dependent, heteroscedastic, or otherwise structured noise in gpCAM — e.g., detector characteristics, count-rate-dependent variance, or non-uniform measurement uncertainty.
gpcam_version: "8.7.x"
fvgp_version: "4.11.x"
last_verified: "2026-10-28 (gpCAM 8.7.0, a21385a)"
---

# Skill: gpCAM Noise Functions

*Verified against gpCAM 8.7.0 / fvgp 4.11.1 — last checked 2026-10-28 (gpCAM a21385a).*

Design custom noise models for experiments with non-uniform or structured noise.

## When to Use

- Noise varies across the parameter space (e.g., edges of detector have more noise)
- Noise depends on signal intensity (Poisson/shot noise)
- You want the noise level to be learned as a hyperparameter
- Correlated noise between measurements

## Noise Function Contract

```python
def my_noise(x, hyperparameters):
    """
    Parameters
    ----------
    x : np.ndarray, shape (N, D)
        Input positions.
    hyperparameters : np.ndarray, 1D
        The FULL hyperparameter vector (shared with kernel and mean).
        Read ONLY the indices reserved for the noise function.
    
    Returns
    -------
    noise : np.ndarray, shape (N,)
        1-D array of strictly positive noise variances (one per point).
        (Returning a 2-D (N, N) noise covariance matrix is deprecated via FutureWarning.)
    """
```

- **Optional `args` parameter:** Declare a 3rd positional parameter `my_noise(x, hps, args)` to receive `GPOptimizer(..., args=...)` (`gp_optimizer.py:179-180`; see `experiment-designer` for the full `args` dispatch table).
- **Analytic gradient (`noise_function_grad`):** Optional callable `f(x, hyperparameters)` (or `f(x, hyperparameters, args)`) returning a 2-D array of shape `(len(hyperparameters), N)` of partial derivatives of the noise variances (`0.0` for kernel/mean hyperparameter rows; or with `gradient_per_direction=True`, `f(x, hyperparameters, direction)` returning shape `(N,)`, `gp_optimizer.py:188-202`).
- **Multi-task shapes (`gp_optimizer.py:100-102, 181-182`):**
  - `noise_variances` array: shape `(V,)` for single-task, `"independent"`, and `"kronecker"` modes; shape `(V, No)` for `multi_task_mode="cartesian"` (with `np.nan` wherever `y_data` is `np.nan`).
  - `noise_function(x, hps)`: in `"independent"` and `"kronecker"` modes `x` has shape `(N, D)`; in `"cartesian"` mode `x` has shape `(N, D + 1)` with the task index in the last column (`x[:, -1]`), and the function returns a 1-D vector of length `N`.

## When to Use What

| Scenario | Approach |
|----------|----------|
| Known, uniform noise | Use `noise_variances=np.full(N, sigma**2)` — no noise function needed |
| Known per-point noise | Use `noise_variances=my_array` (`(N,)` or `(N, No)` in `"cartesian"` mode) |
| Unknown uniform noise | Use a noise function with a learnable hyperparameter |
| Position-dependent noise | Use a noise function that depends on `x` |
| No noise info at all | Don't provide either — gpCAM defaults to `(0.01 * mean|y|)²` |

## Hyperparameter Offset (`K`)

See `experiment-designer` ("Hyperparameter Coordination") for the canonical layout (`hps[:K]` kernel, `hps[K : K + M]` prior mean, `hps[K + M :]` noise). With the single-task default ARD kernel and no custom prior mean, `K = 1 + D = 1 + x.shape[1]`; with a custom kernel or prior mean, start noise indices at `K + M` (and under `multi_task_mode="cartesian"`, remember `x.shape[1]` is `D + 1`).

## Recipes

### Learnable Constant Noise
The noise level is a hyperparameter that gets optimized:
```python
def learnable_noise(x, hps):
    """hps[K] = noise standard deviation (learned). Uses 1 hyperparameter."""
    K = 1 + x.shape[1]
    return np.full(len(x), hps[K]**2)  # return VARIANCE, not std
```

### Position-Dependent Noise
More noise at edges of the measurement range. The bounds must be passed in
explicitly — a noise function that reads a module-level `parameter_bounds` global
raises `NameError` anywhere that global does not happen to exist. Two clean options:

**(a) Via the `args` dict** (declare a third argument; gpCAM passes
`GPOptimizer(..., args=...)` positionally):
```python
def edge_noise(x, hps, args):
    """
    Higher noise near boundaries.
    args["bounds"] = np.ndarray of shape (D, 2), the parameter bounds.
    Uses 1 hyperparameter.
    """
    K = 1 + x.shape[1]
    bounds = args["bounds"]
    base_noise = hps[K]**2
    center = np.mean(bounds, axis=1)
    half_range = (bounds[:, 1] - bounds[:, 0]) / 2
    dist_from_center = np.abs(x - center) / half_range   # 0 at center, 1 at edge
    edge_factor = 1.0 + 5.0 * np.max(dist_from_center, axis=1)**2
    return base_noise * edge_factor

gpo = GPOptimizer(x_data, y_data,
                  noise_function=edge_noise,
                  args={"bounds": parameter_bounds})
```

**(b) Via a factory that closes over the bounds** (no `args` plumbing needed):
```python
def make_edge_noise(bounds):
    """Returns a two-argument noise function with `bounds` captured."""
    center = np.mean(bounds, axis=1)
    half_range = (bounds[:, 1] - bounds[:, 0]) / 2

    def edge_noise(x, hps):
        K = 1 + x.shape[1]
        dist_from_center = np.abs(x - center) / half_range
        return hps[K]**2 * (1.0 + 5.0 * np.max(dist_from_center, axis=1)**2)
    return edge_noise

gpo = GPOptimizer(x_data, y_data, noise_function=make_edge_noise(parameter_bounds))
```
Note that a closure is not picklable by reference — use option (a) if you checkpoint
the optimizer with `pickle`.

### Poisson-Like (Signal-Dependent) Noise
Common in photon-counting detectors, where variance scales with the count rate.

**The obstacle:** true Poisson noise needs the expected signal at `x`, but the noise
function is called *while building the covariance matrix*, so calling
`posterior_mean()` inside it is circular. You need an externally supplied signal
estimate. Two workable approaches:

**(a) Affine-in-signal noise using a supplied estimate** — the honest version. Pass
a callable or lookup that estimates the signal without consulting the current GP
(a pilot scan, a detector calibration, or the previous iteration's posterior mean):
```python
def poisson_noise(x, hps, args):
    """
    Variance affine in the expected signal:  var = a + b * |signal(x)|
    which reproduces counting statistics (var ~ counts) plus a read-noise floor.

    args["signal_estimate"] : callable f(x) -> shape (N,), the expected signal.
    hps[K]   = read-noise floor (std)      -> a = hps[K]**2
    hps[K+1] = gain / scale factor         -> b = hps[K+1]
    Uses 2 hyperparameters.
    """
    K = 1 + x.shape[1]
    signal = np.abs(args["signal_estimate"](x))
    return hps[K]**2 + hps[K+1] * signal
```

To refresh the estimate from the previous iteration's fit inside an ask/tell loop,
update the dict between iterations rather than reading the live GP:
```python
gpo.set_args({"signal_estimate": my_updated_estimator})
gpo.set_hyperparameters(gpo.hyperparameters)   # flush: forces the callables to re-run
```
`set_args` does **not** invalidate the cached covariance — the new dict is only picked
up the next time the callables are invoked (a `train()`, a `set_hyperparameters()`, or
an `update_gp_data(append=False)`). The explicit `set_hyperparameters` call above is
the documented flush.

**Caveat:** this couples the noise model to the current fit. Refresh it at a fixed
cadence (e.g. only when you retrain), never every likelihood evaluation, or training
chases its own tail and the hyperparameters will not settle.

**(b) Skip the noise function entirely.** If you know the counts, Poisson variance is
just the counts — pass it directly and let gpCAM treat it as known:
```python
gpo = GPOptimizer(x_data, y_data, noise_variances=np.maximum(counts, 1.0))
```
This is simpler, exact, and has no feedback loop. **Prefer it whenever you can
measure or estimate counts per point** — which on a photon-counting detector you
usually can.

### Two-Level Noise (Different Detectors/Modes)
```python
def two_detector_noise(x, hps):
    """
    Different noise for two measurement modes.
    Assumes the last dimension of x encodes the mode (0 or 1).
    Uses 2 hyperparameters.
    """
    K = 1 + x.shape[1]
    noise = np.empty(len(x))
    mode_0 = x[:, -1] < 0.5
    noise[mode_0] = hps[K]**2       # detector 1
    noise[~mode_0] = hps[K+1]**2    # detector 2
    return noise
```

## Choosing Noise Bounds

Noise hyperparameters occupy the trailing slice `hps[K + M :]` of the shared hyperparameter vector (see `experiment-designer` for the full layout and disjoint-slice rules).

- **Lower bound**: Never 0 — use `0.001` minimum (prevents singular matrices)
- **Upper bound**: `10 * std(y_data)` — noise shouldn't be larger than the signal
- **Initial value**: `0.01 * mean(|y_data|)` (gpCAM's own default)

## Important: noise_variances vs noise_function

**Do not provide both.** Use one or the other:

```python
# Option A: fixed known noise
gpo = GPOptimizer(x_data, y_data, noise_variances=np.full(N, 0.01))

# Option B: learnable noise function
gpo = GPOptimizer(x_data, y_data, noise_function=learnable_noise)
```

## Common Pitfalls

1. **Returning std instead of variance**: The noise function must return **variance** (σ²), not standard deviation (σ).
2. **Hardcoding `K = 3`**: Correct only for 2-D input with the default kernel. Derive `K = 1 + D` from `x.shape[1]`.
3. **Referencing an out-of-scope global**: A noise function that reads `parameter_bounds` from module scope raises `NameError` wherever that global doesn't exist. Pass bounds via `args` or a factory closure.
4. **Zero noise**: Causes singular matrix errors. Always ensure noise > 0.
5. **Providing both**: Don't pass `noise_variances` AND `noise_function` — pick one.
6. **Forgetting bounds**: The noise hyperparameter needs bounds in the `hyperparameter_bounds` array.
7. **Wrong shape**: Return a 1-D vector of shape `(N,)` from `noise_function` (matrix-valued `(N, N)` noise is deprecated via `FutureWarning`).
