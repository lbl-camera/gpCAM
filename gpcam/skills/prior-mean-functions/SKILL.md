---
name: prior-mean-functions
description: Use when encoding known physics, theoretical models, or expected trends as prior mean functions for gpCAM — useful when there's a baseline expectation the GP should regress against rather than a flat zero prior.
gpcam_version: "8.4.x"
fvgp_version: "4.8.x"
last_verified: "2026-07-23 (gpCAM dadeb65)"
---

# Skill: gpCAM Prior Mean Functions

*Verified against gpCAM 8.4.x / fvgp 4.8.x — last checked 2026-07-23 (gpCAM `dadeb65`).*

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

An optional third argument receives the constructor's `args` dict — declare it and
gpCAM passes it positionally (dispatch is by parameter count, so the name is yours
to choose):

```python
def my_mean(x, hps, args):          # GPOptimizer(..., args={"d_spacing": 3.14})
    return np.full(len(x), args["d_spacing"] * hps[K])
```

If you also want analytic gradients, pass `prior_mean_function_grad=` returning shape
`(len(hps), N)`. Without it gpCAM uses finite differences, which is fine but slower.

**If no prior mean is provided**, gpCAM uses **the average of `y_data`** as a constant
mean (`fvgp.GP._default_mean_function`) — not a flat zero. This matters: away from
data the posterior reverts to `mean(y_data)`, not to zero. It is a sensible default
for most cases.

## When NOT to Use a Prior Mean

- You don't have a strong physical expectation → use default (mean of data)
- Your model might be wrong → a bad prior mean biases the GP and can hurt more than help
- You're purely exploring → the GP will learn the mean from data

## The Index Convention

Every recipe below starts its hyperparameters at index `K`. **Derive `K`; never
hardcode it.** With the default ARD Matérn kernel the kernel owns `hps[0]` (signal
variance) plus one length scale per input dimension, so:

```python
K = 1 + D    # D = number of input dimensions = x.shape[1]
```

`K = 3` is only correct for a 2-D input space. In 1-D it points at a length scale;
in 3-D it points at the wrong slot and two components silently share a
hyperparameter. Compute `K` inside the function from `x.shape[1]`, or define it once
as a module-level constant next to your bounds array so the two cannot drift apart.

**Never use negative indices** (`hps[-1]`). The end of the vector belongs to the
noise function under the standard layout — see the ordering rule below — so `hps[-1]`
silently ties your prior mean to the noise amplitude.

With a custom kernel, `K` is however many hyperparameters your kernel reads.

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

## Hyperparameter Coordination

**This is where most bugs happen.** The prior mean function receives the same
hyperparameter vector as the kernel and noise functions, and fvGP requires the index
ranges each callable reads to be **disjoint**. The gradient computation depends on
it: when a hyperparameter index belongs to the mean function, its kernel derivative
is assumed zero, and vice versa. Overlapping indices therefore produce not just a
wrong model but wrong gradients.

**The standard ordering used across all gpCAM skills is: kernel, then mean, then
noise.**

```
[ kernel hps ][ mean hps ][ noise hps ]
  0 .. K-1     K .. M-1     M .. end
```

You must:

1. Decide which indices the mean function uses — derive `K`, don't hardcode it
2. Add bounds for those hyperparameters
3. Document the full layout

```python
# Example layout (2-D input, linear mean, learnable noise):
# hps[0]     = signal variance     (kernel)
# hps[1:3]   = length scales       (kernel, 2D input)
# hps[3]     = intercept           (mean function)   <- K = 1 + D = 3
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

### Complete worked example

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
