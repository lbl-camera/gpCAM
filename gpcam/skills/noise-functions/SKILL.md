---
name: noise-functions
description: Use when modeling position-dependent, heteroscedastic, or otherwise structured noise in gpCAM — e.g., detector characteristics, count-rate-dependent variance, or non-uniform measurement uncertainty.
gpcam_version: "8.4.x"
fvgp_version: "4.8.x"
last_verified: "2026-07-23 (gpCAM dadeb65)"
---

# Skill: gpCAM Noise Functions

*Verified against gpCAM 8.4.x / fvgp 4.8.x — last checked 2026-07-23 (gpCAM `dadeb65`).*

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
    noise : np.ndarray
        Either shape (N,) for diagonal noise (independent per point),
        or shape (N, N) for full noise covariance matrix.
    """
```

An optional third argument receives the constructor's `args` dict — gpCAM dispatches
on parameter count, so declaring it is what turns it on:

```python
def my_noise(x, hps, args):      # GPOptimizer(..., args={"bounds": ...})
    ...
```

Pass it with `noise_function=my_noise`. For analytic gradients, also pass
`noise_function_grad=`.

## When to Use What

| Scenario | Approach |
|----------|----------|
| Known, uniform noise | Use `noise_variances=np.full(N, sigma**2)` — no noise function needed |
| Known per-point noise | Use `noise_variances=my_array` — no noise function needed |
| Unknown uniform noise | Use a noise function with a learnable hyperparameter |
| Position-dependent noise | Use a noise function that depends on `x` |
| No noise info at all | Don't provide either — gpCAM defaults to `(0.01 * mean|y|)²` |

## The Index Convention

Every recipe below starts its hyperparameters at index `K`. **Derive `K`; never
hardcode it.** With the default ARD Matérn kernel the kernel owns `hps[0]` (signal
variance) plus one length scale per input dimension:

```python
K = 1 + D    # D = number of input dimensions = x.shape[1]
```

`K = 3` is only correct for a 2-D input space. In 1-D it points at a length scale;
in 3-D the noise function returns a squared length scale as a variance while training
optimizes one parameter that two components are reading. All of that is silent.

The standard ordering across gpCAM skills is **kernel, then mean, then noise** — so
if you also use a custom prior mean, the noise indices start after the mean's, and
`K` grows accordingly. See the `prior-mean-functions` skill.

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

## Hyperparameter Coordination

Noise function hyperparameters come from the **same vector** as kernel and mean
hyperparameters, and the index ranges must be **disjoint** — fvGP assumes a
hyperparameter belonging to the noise function has zero kernel derivative, so an
overlap corrupts the training gradients as well as the model.

Standard ordering: **kernel, then mean, then noise.** Noise is last, which is exactly
why a prior mean function must never read `hps[-1]`.

```python
# Example layout (2-D input, default kernel, no custom mean):
# hps[0]     = signal variance     (kernel)
# hps[1:3]   = length scales       (kernel)
# hps[3]     = noise std dev       (noise function) ← K = 1 + D = 3
#
# Total: 4 hyperparameters

hp_bounds = np.array([
    [0.001, 100.0],  # signal variance
    [0.01, 50.0],    # length scale dim 0
    [0.01, 50.0],    # length scale dim 1
    [0.001, 10.0],   # noise std dev
])
```

### Choosing Noise Bounds
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
7. **Wrong shape**: Return shape `(N,)` for independent noise, `(N, N)` for correlated.
