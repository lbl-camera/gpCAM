---
name: prior-mean-functions
description: Use when encoding known physics, theoretical models, or expected trends as prior mean functions for gpCAM — useful when there's a baseline expectation the GP should regress against rather than a flat zero prior.
---

# Skill: gpCAM Prior Mean Functions

Design prior mean functions that encode known physics or expected trends.

## When to Use

When the user has prior knowledge about the expected behavior:
- A known baseline or background (linear trend, polynomial)
- Physical model (Gaussian peak, Lorentzian, Bragg's law)
- Expected shape from previous experiments
- The signal is known to be non-zero on average

## Prior Mean Function Contract

```python
def my_mean(x, hyperparameters):
    """
    Parameters
    ----------
    x : np.ndarray, shape (N, D)
        Input positions.
    hyperparameters : np.ndarray, 1D
        The FULL hyperparameter vector (shared with kernel and noise).
    
    Returns
    -------
    m : np.ndarray, shape (N,)
        Prior mean value at each input point.
    """
```

**If no prior mean is provided**, gpCAM uses the average of `y_data` as a constant mean — this is fine for most cases.

## When NOT to Use a Prior Mean

- You don't have a strong physical expectation → use default (mean of data)
- Your model might be wrong → a bad prior mean biases the GP and can hurt more than help
- You're purely exploring → the GP will learn the mean from data

## Recipes

### Constant Mean (explicit)
```python
def constant_mean(x, hps):
    """Explicit constant mean. hps[-1] = mean value."""
    return np.full(len(x), hps[-1])
```

### Linear Trend
```python
def linear_mean(x, hps):
    """
    Linear prior mean: m(x) = a + b0*x0 + b1*x1 + ...
    Uses hps[K], hps[K+1], ..., hps[K+D] where K is where mean hps start.
    """
    K = 3  # INDEX WHERE MEAN HYPERPARAMETERS START — adjust for your kernel
    D = x.shape[1]
    intercept = hps[K]
    slopes = hps[K+1:K+1+D]
    return intercept + x @ slopes
```

### Gaussian Peak (known approximate location)
```python
def gaussian_peak_mean(x, hps):
    """
    Prior: expect a Gaussian peak near a known location.
    hps[K]: amplitude
    hps[K+1]: center_x
    hps[K+2]: center_y
    hps[K+3]: width
    """
    K = 3  # adjust
    amp = hps[K]
    cx, cy = hps[K+1], hps[K+2]
    w = hps[K+3]
    r2 = (x[:, 0] - cx)**2 + (x[:, 1] - cy)**2
    return amp * np.exp(-r2 / (2 * w**2))
```

### Polynomial Background
```python
def quadratic_mean(x, hps):
    """Quadratic background: a + b*x + c*x^2 (1D)."""
    K = 2  # adjust
    return hps[K] + hps[K+1] * x[:, 0] + hps[K+2] * x[:, 0]**2
```

### Physics-Informed (Bragg's Law Example)
```python
def bragg_mean(x, hps):
    """
    Prior mean based on Bragg's law: peak at 2*d*sin(theta) = n*lambda.
    x[:, 0] = 2theta angle in degrees
    hps[K] = amplitude
    hps[K+1] = d-spacing estimate
    """
    K = 3
    wavelength = 1.54  # Cu K-alpha, fixed
    two_theta = np.radians(x[:, 0])
    # Expected peak positions
    d = hps[K+1]
    expected = hps[K] * np.exp(-(np.sin(two_theta/2) - wavelength/(2*d))**2 / 0.001)
    return expected
```

## Hyperparameter Coordination

**This is where most bugs happen.** The prior mean function receives the same hyperparameter vector as the kernel and noise functions. You must:

1. Decide which indices the mean function uses
2. Add bounds for those hyperparameters
3. Document the full layout

```python
# Example layout:
# hps[0]     = signal variance     (kernel)
# hps[1:3]   = length scales       (kernel, 2D input)
# hps[3]     = intercept           (mean function)
# hps[4:6]   = slopes              (mean function)
# hps[6]     = noise amplitude     (noise function)
#
# Total: 7 hyperparameters

hp_bounds = np.array([
    [0.001, 100.0],   # signal variance
    [0.01, 50.0],     # length scale dim 0
    [0.01, 50.0],     # length scale dim 1
    [-10.0, 10.0],    # intercept
    [-5.0, 5.0],      # slope dim 0
    [-5.0, 5.0],      # slope dim 1
    [0.001, 10.0],    # noise amplitude
])
```

## Setting Mean Function Hyperparameter Bounds

- **Intercept/amplitude**: `[min(y_data) * 2, max(y_data) * 2]`
- **Slopes**: `[-range(y)/range(x), +range(y)/range(x)]` 
- **Peak position**: `[known_position - tolerance, known_position + tolerance]`
- **Width**: `[min_expected_width, max_expected_width]`

## Common Pitfalls

1. **Wrong hyperparameter indices**: Double-check which indices the mean function reads — they must not overlap with kernel/noise indices.
2. **Overconfident prior**: If the mean function is too specific and wrong, the GP will fight between data and prior. Keep it loose.
3. **Forgetting to add bounds**: Every hyperparameter used by the mean function needs a row in `hyperparameter_bounds`.
4. **Return shape**: Must return 1D array of length `len(x)`, not a scalar.
