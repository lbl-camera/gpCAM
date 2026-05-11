---
name: noise-functions
description: Use when modeling position-dependent, heteroscedastic, or otherwise structured noise in gpCAM — e.g., detector characteristics, count-rate-dependent variance, or non-uniform measurement uncertainty.
---

# Skill: gpCAM Noise Functions

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
    
    Returns
    -------
    noise : np.ndarray
        Either shape (N,) for diagonal noise (independent per point),
        or shape (N, N) for full noise covariance matrix.
    """
```

## When to Use What

| Scenario | Approach |
|----------|----------|
| Known, uniform noise | Use `noise_variances=np.full(N, sigma**2)` — no noise function needed |
| Known per-point noise | Use `noise_variances=my_array` — no noise function needed |
| Unknown uniform noise | Use a noise function with a learnable hyperparameter |
| Position-dependent noise | Use a noise function that depends on `x` |
| No noise info at all | Don't provide either — gpCAM defaults to `(0.01 * mean|y|)²` |

## Recipes

### Learnable Constant Noise
The noise level is a hyperparameter that gets optimized:
```python
def learnable_noise(x, hps):
    """hps[K] = noise standard deviation (learned)."""
    K = 3  # INDEX WHERE NOISE HP STARTS — adjust for your kernel
    return np.full(len(x), hps[K]**2)  # return VARIANCE, not std
```

### Position-Dependent Noise
More noise at edges of the measurement range:
```python
def edge_noise(x, hps):
    """Higher noise near boundaries."""
    K = 3
    base_noise = hps[K]**2
    # Increase noise near edges (within 10% of range)
    center = np.mean(parameter_bounds, axis=1)
    half_range = (parameter_bounds[:, 1] - parameter_bounds[:, 0]) / 2
    dist_from_center = np.abs(x - center) / half_range  # 0 at center, 1 at edge
    edge_factor = 1.0 + 5.0 * np.max(dist_from_center, axis=1)**2
    return base_noise * edge_factor
```

### Poisson-Like (Signal-Dependent) Noise
Common in photon-counting detectors:
```python
def poisson_noise(x, hps):
    """
    Noise proportional to sqrt of expected signal.
    Uses the GP posterior mean as estimate of signal.
    
    NOTE: This creates a feedback loop — use carefully.
    hps[K] = noise scale factor
    """
    K = 3
    # Can't call posterior_mean here directly (circular dependency)
    # Instead, use a fixed estimate or pass through args
    # Simple version: just scale with position
    return np.full(len(x), hps[K]**2)
```

### Two-Level Noise (Different Detectors/Modes)
```python
def two_detector_noise(x, hps):
    """
    Different noise for two measurement modes.
    Assumes last dimension of x encodes the mode (0 or 1).
    """
    K = 3
    noise = np.empty(len(x))
    mode_0 = x[:, -1] < 0.5
    mode_1 = ~mode_0
    noise[mode_0] = hps[K]**2      # detector 1
    noise[mode_1] = hps[K+1]**2    # detector 2
    return noise
```

## Hyperparameter Coordination

Noise function hyperparameters come from the **same vector** as kernel and mean hyperparameters.

```python
# Example layout:
# hps[0]     = signal variance     (kernel)
# hps[1:3]   = length scales       (kernel)
# hps[3]     = noise std dev       (noise function) ← YOUR NOISE HP
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
2. **Zero noise**: Causes singular matrix errors. Always ensure noise > 0.
3. **Providing both**: Don't pass `noise_variances` AND `noise_function` — pick one.
4. **Forgetting bounds**: The noise hyperparameter needs bounds in the `hyperparameter_bounds` array.
5. **Wrong shape**: Return shape `(N,)` for independent noise, `(N, N)` for correlated.
