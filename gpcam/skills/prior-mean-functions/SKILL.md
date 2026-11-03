---
name: prior-mean-functions
description: Use when encoding known physics, theoretical models, or expected trends as prior mean functions for gpCAM — useful when there's a baseline expectation or physical model the GP should regress against rather than the default constant data-mean prior (mean(y_data)).
gpcam_version: "8.7.x"
fvgp_version: "4.11.x"
last_verified: "2026-10-28 (gpCAM 8.7.0, a21385a)"
---

# Skill: gpCAM Prior Mean Functions

*Verified against gpCAM 8.7.0 / fvgp 4.11.1 — last checked 2026-10-28 (gpCAM a21385a).*

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
        Read ONLY the indices reserved for the mean function.
    
    Returns
    -------
    m : np.ndarray, shape (N,)
        Prior mean value at each input point.
    """
```

Attach it with the `prior_mean_function` keyword:

```python
from gpcam import GPOptimizer

gpo = GPOptimizer(
    x_data=x_data,
    y_data=y_data,
    init_hyperparameters=init_hps,      # must cover kernel + mean (+ noise)
    prior_mean_function=my_mean,
)
gpo.train(hyperparameter_bounds=hp_bounds)
```

- **Optional `args` parameter:** Declare a 3rd positional parameter `my_mean(x, hps, args)` to receive `GPOptimizer(..., args=...)` (`gp_optimizer.py:153-154`; see `experiment-designer` for the full `args` dispatch table).
- **Analytic gradient (`prior_mean_function_grad`):** Optional callable returning shape `(len(hps), N)` (`gp_optimizer.py:165-171`).
- **Default prior mean:** When `prior_mean_function` is omitted, gpCAM uses `fvgp.GP._default_mean_function`, which is a constant equal to each output's average of `y_data` (`gp_optimizer.py:157-160`) — not zero.
- **Multi-task shapes (`gp_optimizer.py:153-160`):**
  - In `"independent"` and `"kronecker"` modes, `x` has shape `(N, D)` and `prior_mean_function` may return `(N,)` or `(N, No)`.
  - In `multi_task_mode="cartesian"`, `x` has shape `(N, D + 1)` with the task index in the last column (`x[:, -1]`), and `prior_mean_function` returns a 1-D vector of shape `(N,)` (remember `x.shape[1]` is `D + 1`).

## When NOT to Use a Prior Mean

- You don't have a strong physical expectation → use default (mean of data)
- Your model might be wrong → a bad prior mean biases the GP and can hurt more than help
- You're purely exploring → the GP will learn the mean from data

## Hyperparameter Offset (`K`)

See `experiment-designer` ("Hyperparameter Coordination") for the canonical layout (`hps[:K]` kernel, `hps[K : K + M]` prior mean, `hps[K + M :]` noise). With the single-task default ARD kernel, `K = 1 + D = 1 + x.shape[1]`; with a custom kernel, `K` is the number of parameters that kernel consumes. Never index with `hps[-1]` in a prior mean function — the trailing slice belongs to the noise function.

## Recipes

### Constant Mean (explicit)
```python
def constant_mean(x, hps):
    """
    Explicit constant mean.
    hps[K] = mean value, where K = 1 + D for the default kernel.
    """
    K = 1 + x.shape[1]
    return np.full(len(x), hps[K])
```

### Linear Trend
```python
def linear_mean(x, hps):
    """
    Linear prior mean: m(x) = a + b0*x0 + b1*x1 + ...
    hps[K]         = intercept
    hps[K+1:K+1+D] = slopes
    Uses D + 1 hyperparameters.
    """
    D = x.shape[1]
    K = 1 + D            # default kernel: hps[0] + D length scales
    intercept = hps[K]
    slopes = hps[K+1:K+1+D]
    return intercept + x @ slopes
```

### Gaussian Peak (known approximate location)
```python
def gaussian_peak_mean(x, hps):
    """
    Prior: expect a Gaussian peak near a known location. 2-D input.
    hps[K]:   amplitude
    hps[K+1]: center_x
    hps[K+2]: center_y
    hps[K+3]: width
    Uses 4 hyperparameters.
    """
    K = 1 + x.shape[1]
    amp = hps[K]
    cx, cy = hps[K+1], hps[K+2]
    w = hps[K+3]
    r2 = (x[:, 0] - cx)**2 + (x[:, 1] - cy)**2
    return amp * np.exp(-r2 / (2 * w**2))
```

### Polynomial Background
```python
def quadratic_mean(x, hps):
    """
    Quadratic background: a + b*x + c*x^2 (1-D input, so K = 2).
    Uses 3 hyperparameters.
    """
    K = 1 + x.shape[1]
    return hps[K] + hps[K+1] * x[:, 0] + hps[K+2] * x[:, 0]**2
```

### Physics-Informed (Bragg's Law Example)
```python
def bragg_mean(x, hps):
    """
    Prior mean based on Bragg's law: peak at 2*d*sin(theta) = n*lambda.
    x[:, 0] = 2theta angle in degrees
    hps[K]   = amplitude
    hps[K+1] = d-spacing estimate
    Uses 2 hyperparameters.
    """
    K = 1 + x.shape[1]
    wavelength = 1.54  # Cu K-alpha, fixed
    two_theta = np.radians(x[:, 0])
    d = hps[K+1]
    expected = hps[K] * np.exp(-(np.sin(two_theta/2) - wavelength/(2*d))**2 / 0.001)
    return expected
```

## Complete Worked Example (Kernel + Mean + Noise)

Prior mean hyperparameters sit at `hps[K : K + M]` between the kernel (`hps[:K]`) and noise (`hps[K + M :]`) slices — see `experiment-designer` for the full coordination rules.

```python
import numpy as np
from gpcam import GPOptimizer

D = 2
K = 1 + D                      # mean hyperparameters start here

def linear_mean(x, hps):
    return hps[K] + x @ hps[K+1:K+1+D]

def learnable_noise(x, hps):
    return np.full(len(x), hps[K + 1 + D]**2)     # noise sits after the mean

hp_bounds = np.array(
    [[0.001, 100.0]] +          # signal variance      (kernel)
    [[0.01, 50.0]] * D +        # length scales        (kernel)
    [[-10.0, 10.0]] +           # intercept            (mean)
    [[-5.0, 5.0]] * D +         # slopes               (mean)
    [[0.001, 10.0]]             # noise amplitude      (noise)
)

gpo = GPOptimizer(
    x_data, y_data,
    init_hyperparameters=np.ones(len(hp_bounds)),
    prior_mean_function=linear_mean,
    noise_function=learnable_noise,
)
gpo.train(hyperparameter_bounds=hp_bounds)

assert len(gpo.hyperparameters) == len(hp_bounds)   # cheap layout check
```

## Setting Mean Function Hyperparameter Bounds

- **Intercept/amplitude**: `[min(y_data) * 2, max(y_data) * 2]`
- **Slopes**: `[-range(y)/range(x), +range(y)/range(x)]` 
- **Peak position**: `[known_position - tolerance, known_position + tolerance]`
- **Width**: `[min_expected_width, max_expected_width]`

## Common Pitfalls

1. **Hardcoding the start index**: `K = 3` is right only for 2-D input with the default kernel. Derive `K = 1 + D`.
2. **Negative indices**: `hps[-1]` collides with the noise function under the standard kernel → mean → noise ordering. Always index forward from `K`.
3. **Wrong hyperparameter indices**: Double-check which indices the mean function reads — they must not overlap with kernel/noise indices, or the gradients are wrong too.
4. **Forgetting `prior_mean_function=`**: Defining the function is not enough; it must be passed to the constructor.
5. **Overconfident prior**: If the mean function is too specific and wrong, the GP will fight between data and prior. Keep it loose.
6. **Forgetting to add bounds**: Every hyperparameter used by the mean function needs a row in `hyperparameter_bounds`.
7. **Return shape**: Must return 1D array of length `len(x)`, not a scalar.
