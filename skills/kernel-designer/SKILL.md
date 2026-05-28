---
name: kernel-designer
description: Use when designing or composing custom kernel (covariance) functions for gpCAM that encode domain knowledge — smoothness, periodicity, symmetry, anisotropy, or non-Euclidean input spaces.
---

# Skill: gpCAM Kernel Designer

Design custom kernel (covariance) functions for gpCAM that encode domain knowledge about the experiment.

## When to Use

When a user needs a kernel that goes beyond the default ARD Matérn-3/2:
- Known periodicity in the data
- Symmetry constraints (mirror, rotational)
- Different smoothness in different dimensions
- Combining multiple correlation structures (sum/product kernels)
- Non-Euclidean input spaces (strings, graphs, categorical)
- Non-stationary behavior (varying length scales)

## Kernel Function Contract

Every gpCAM kernel must satisfy:

```python
def my_kernel(x1, x2, hyperparameters):
    """
    Parameters
    ----------
    x1 : np.ndarray, shape (N1, D)
    x2 : np.ndarray, shape (N2, D)
    hyperparameters : np.ndarray, 1D
    
    Returns
    -------
    K : np.ndarray, shape (N1, N2)
        Must be symmetric positive semi-definite.
    """
```

- `x1` and `x2` are 2D arrays even for 1D inputs
- The output must be an `(N1, N2)` matrix
- The kernel must be symmetric: `k(x1, x2) = k(x2, x1).T`
- The kernel must be positive semi-definite
- Use vectorized numpy operations — avoid Python loops over data points

## Building Blocks

Import from `gpcam.kernels` (or define locally):

```python
from gpcam.kernels import (
    matern_kernel_diff1,      # Matérn ν=3/2: once differentiable
    matern_kernel_diff2,      # Matérn ν=5/2: twice differentiable
    squared_exponential_kernel,  # RBF/SE: infinitely smooth
    wendland_kernel,           # Compact support (for gp2Scale)
    get_distance_matrix,       # Euclidean distance
    get_anisotropic_distance_matrix,  # ARD distance
)
```

These base kernels operate on **distance matrices**, not raw points. The pattern is:
1. Compute a distance matrix from points
2. Apply a base kernel to the distance matrix

## Kernel Recipes

### Standard Anisotropic (ARD) Kernel
```python
def anisotropic_matern(x1, x2, hps):
    """
    hps[0]: signal variance
    hps[1:D+1]: per-dimension length scales
    """
    d = get_anisotropic_distance_matrix(x1, x2, hps[1:])
    return hps[0] * matern_kernel_diff1(d, 1.0)
```

### Periodic Kernel
For data with known periodicity (e.g., angular measurements, crystal lattice):
```python
def periodic_kernel(x1, x2, hps):
    """
    hps[0]: signal variance
    hps[1]: length scale
    hps[2]: period
    """
    d = np.abs(np.subtract.outer(x1[:, 0], x2[:, 0]))
    return hps[0] * np.exp(-2.0 * np.sin(np.pi * d / hps[2])**2 / hps[1]**2)
```

### Periodic + Smooth (Product Kernel)
Periodic in one dimension, smooth Matérn in others:
```python
def periodic_plus_smooth(x1, x2, hps):
    """
    hps[0]: signal variance
    hps[1]: periodic length scale
    hps[2]: period
    hps[3:]: Matérn length scales for remaining dims
    """
    # Periodic in dim 0
    d_periodic = np.abs(np.subtract.outer(x1[:, 0], x2[:, 0]))
    k_periodic = np.exp(-2.0 * np.sin(np.pi * d_periodic / hps[2])**2 / hps[1]**2)
    
    # Matérn in remaining dims
    d_other = get_anisotropic_distance_matrix(x1[:, 1:], x2[:, 1:], hps[3:])
    k_other = matern_kernel_diff1(d_other, 1.0)
    
    return hps[0] * k_periodic * k_other
```

### Sum Kernel (Multiple Scales)
Captures both coarse and fine structure:
```python
def multi_scale_kernel(x1, x2, hps):
    """
    hps[0]: variance of coarse component
    hps[1]: length scale of coarse component
    hps[2]: variance of fine component
    hps[3]: length scale of fine component
    """
    d = get_distance_matrix(x1, x2)
    k_coarse = hps[0] * matern_kernel_diff2(d, hps[1])  # smooth, long-range
    k_fine = hps[2] * matern_kernel_diff1(d, hps[3])     # rougher, short-range
    return k_coarse + k_fine
```

### Symmetry-Enforcing Kernel
For data known to be symmetric about an axis:
```python
def symmetric_kernel_x(x1, x2, hps):
    """Mirror symmetry about x=0 in the first dimension."""
    x1_flip = x1.copy()
    x1_flip[:, 0] = -x1_flip[:, 0]
    x2_flip = x2.copy()
    x2_flip[:, 0] = -x2_flip[:, 0]
    
    d = get_anisotropic_distance_matrix
    k = lambda a, b: hps[0] * matern_kernel_diff1(d(a, b, hps[1:]), 1.0)
    
    return 0.25 * (k(x1, x2) + k(x1_flip, x2) + k(x1, x2_flip) + k(x1_flip, x2_flip))
```

### L1 (Manhattan) Distance Kernel
Separable per dimension — each dimension contributes independently:
```python
def l1_kernel(x1, x2, hps):
    """Product of per-dimension exponential kernels (L1 distance)."""
    k = hps[0] * np.ones((len(x1), len(x2)))
    for i in range(x1.shape[1]):
        d_i = np.abs(np.subtract.outer(x1[:, i], x2[:, i]))
        k *= np.exp(-d_i / hps[1 + i])
    return k
```

### Non-Euclidean Input Spaces (strings, graphs, categorical)

gpCAM accepts arbitrary Python objects as inputs — `x_data` can be a list of strings, graphs, molecules, etc. The only requirement is that your kernel computes a valid covariance between any two objects.

```python
from gpcam import GPOptimizer
from gpcam.kernels import matern_kernel_diff1

def string_distance(s1, s2):
    diff = abs(len(s1) - len(s2))
    common = min(len(s1), len(s2))
    return diff + sum(a != b for a, b in zip(s1[:common], s2[:common]))

def string_kernel(x1, x2, hps):
    """x1, x2 are lists/sequences of strings; hps = [signal_var, length_scale]."""
    d = np.array([[string_distance(a, b) for b in x2] for a in x1])
    return hps[0] * matern_kernel_diff1(d, hps[1])

x_data = ["hello", "world", "this", "is", "gpcam"]
y_data = np.array([2.0, 1.9, 1.8, 3.0, 5.0])

gp = GPOptimizer(x_data, y_data,
                 init_hyperparameters=np.ones(2),
                 kernel_function=string_kernel)
gp.train(hyperparameter_bounds=np.array([[1e-3, 100.], [1e-3, 100.]]))

# Predict on new objects:
gp.posterior_mean(["full"])["m(x)"]

# Ask which of a candidate set to measure next:
gp.ask(["who", "could", "it", "be"], n=4)
```

Notes:
- A Python `for` loop over objects is fine here (you're bottlenecked by the distance function, not numpy).
- If the distance function is symmetric, the resulting kernel is automatically symmetric; you still need the base kernel (Matérn/SE) to make it PSD.
- For multi-task on non-Euclidean inputs: `fvGPOptimizer(x_strings, y_multi, kernel_function=...)` works the same way.

### Deep Kernel (NN-warped input space)

For hard multi-task structure or learned metrics, warp the input through a small neural net and then apply a stationary kernel in the warped space. `gpcam.deep_kernel_network.Network` gives you an MLP whose weights you read out of `hps`:

```python
from gpcam.deep_kernel_network import Network
from gpcam.kernels import get_distance_matrix, matern_kernel_diff1

iset_dim = 3
layer_width = 5
n = Network(iset_dim, layer_width)
# n.number_of_hps tells you how many NN hyperparameters to reserve.

def deep_kernel(x1, x2, hps):
    signal_var, length_scale = hps[0], hps[1]
    # unpack the remaining hps into n.set_weights / n.set_biases (layout: see manual)
    x1_nn = n.forward(x1)
    x2_nn = n.forward(x2)
    d = get_distance_matrix(x1_nn, x2_nn)
    return signal_var * matern_kernel_diff1(d, length_scale)
```

Hyperparameter layout: `hps[0]=signal_var`, `hps[1]=length_scale`, `hps[2:]` = flattened NN weights+biases in the order `Network` expects. Use `method="mcmc"` or `method="adam"` for training; `global`/`local` don't scale well to NN-sized hyperparameter vectors.

## Smoothness Guide

| Kernel | Smoothness | When to Use |
|--------|-----------|-------------|
| Matérn-1/2 (exponential) | Rough, continuous but not differentiable | Sharp peaks, discontinuities |
| Matérn-3/2 | Once differentiable | **Default choice** — most physical data |
| Matérn-5/2 | Twice differentiable | Smoother physical signals |
| Squared Exponential (RBF) | Infinitely smooth | Very smooth data; tends to oversmooth |

## Hyperparameter Coordination

**Critical:** When designing a custom kernel, you must:

1. **Document the hyperparameter layout** — which index maps to what
2. **Set matching bounds** — the `hyperparameter_bounds` array must have one row per hyperparameter
3. **Coordinate with noise/mean functions** — all three share the same hyperparameter vector

```python
# ALWAYS include a comment block like this:
#
# Hyperparameter layout for my_kernel:
# hps[0]     = signal variance          bounds: [0.001, 100]
# hps[1]     = length scale dim 0       bounds: [0.01, range_dim0 * 10]
# hps[2]     = length scale dim 1       bounds: [0.01, range_dim1 * 10]
# hps[3]     = period (if periodic)     bounds: [expected_period * 0.5, expected_period * 2]
```

### Setting Initial Hyperparameters
- **Signal variance**: start near `np.var(y_data)` or `1.0`
- **Length scales**: start near `0.1 * range(x_dim)` — not too small (overfitting) or too large (underfitting)
- **Period**: start near the expected period if known

### Setting Bounds
- **Signal variance**: `[0.001, 10 * np.std(y_data)]`
- **Length scales**: `[0.01, 10 * range(x_dim)]` — lower bound prevents overfitting
- **Period**: `[0.5 * expected, 2 * expected]` — tight if well-known

## Common Pitfalls

1. **Non-PSD kernel**: Sums and products of valid kernels are valid. Differences are NOT guaranteed PSD.
2. **Python loops over data**: Use `np.subtract.outer` and vectorized ops. A double for-loop over N points is O(N²) in Python — unusable for >100 points.
3. **Forgetting signal variance**: Always include a leading amplitude hyperparameter (`hps[0] * ...`).
4. **Length scale = 0**: Causes division by zero. Set lower bounds > 0 (e.g., 0.001).
5. **Mismatched hyperparameter count**: The bounds array rows must equal the total hyperparameter vector length.

## Reference

See `gpcam/kernels.py` (re-exports the fvGP kernel library) for the full library of kernel building blocks and the [gpCAM documentation](https://gpcam.readthedocs.io) for mathematical details.
