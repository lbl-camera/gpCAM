---
name: kernel-designer
description: Use when designing or composing custom kernel (covariance) functions for gpCAM that encode domain knowledge — smoothness, periodicity, symmetry, anisotropy, or non-Euclidean input spaces.
gpcam_version: "8.4.x"
fvgp_version: "4.8.x"
last_verified: "2026-07-23 (gpCAM dadeb65)"
---

# Skill: gpCAM Kernel Designer

*Verified against gpCAM 8.4.x / fvgp 4.8.x — last checked 2026-07-23 (gpCAM `dadeb65`).*

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
def periodic_kernel_1d(x1, x2, hps):
    """
    1-D INPUTS ONLY. Reads x[:, 0] and ignores every other column, so on
    D > 1 inputs the covariance is constant along the remaining axes — the
    matrix becomes rank-deficient and Cholesky will fail or the posterior
    will be degenerate. For D > 1, use `periodic_plus_smooth` below, which
    multiplies the periodic factor by a Matérn factor over the other dims.

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
    ISOTROPIC — uses a single Euclidean distance, so all input dimensions share
    the same length scale. For per-dimension length scales, build each component
    with `get_anisotropic_distance_matrix` as in `anisotropic_matern` above.

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
    """
    SEPARABLE and ANISOTROPIC — a product of independent per-dimension
    exponential kernels, so dimension i has its own length scale hps[1 + i].
    Requires D + 1 hyperparameters. Rougher than Matérn-3/2 in every direction.
    """
    k = hps[0] * np.ones((len(x1), len(x2)))
    for i in range(x1.shape[1]):
        d_i = np.abs(np.subtract.outer(x1[:, i], x2[:, i]))
        k *= np.exp(-d_i / hps[1 + i])
    return k
```

### Non-Stationary Kernels (varying amplitude or length scale)

A stationary kernel assumes the covariance depends only on `x1 - x2`, so the signal
amplitude and smoothness are the same everywhere. Real experiments often violate
this — a sample is featureless in one corner and structured in another. Two
practical constructions:

**(a) Varying amplitude.** `non_stat_kernel(x1, x2, x0, w, l)` builds `g(x1) g(x2)ᵀ`
from radial basis functions at locations `x0` with weights `w` (one per basis
location) and scalar width `l`. On its own it is low-rank and degenerate — and when
the learned amplitude `g(x)` passes through zero at some points, those rows of the
covariance vanish and Cholesky fails with `array must not contain infs or NaNs`. Two
things are therefore mandatory: **multiply by a stationary kernel**, and **add a
small constant floor** so the amplitude never collapses the matrix:

```python
from gpcam.kernels import non_stat_kernel, get_anisotropic_distance_matrix, matern_kernel_diff1

# Basis-function locations: a coarse grid over the input space. Fixed, not learned.
X0 = np.array([[2.0], [5.0], [8.0]])
NB = len(X0)

def varying_amplitude_kernel(x1, x2, hps):
    """
    Amplitude varies over the input space; smoothness does not.

    hps[0]          = basis width l (scalar, > 0)
    hps[1:1+NB]     = basis weights w, one per basis location (may be negative)
    hps[1+NB:]      = D stationary length scales
    Total: 1 + NB + D hyperparameters
    """
    floor = 1e-2      # keeps the amplitude term full-rank when g(x) crosses zero
    # NOTE the argument order: non_stat_kernel(x1, x2, x0, w, l) — weights, then width.
    k_amp = non_stat_kernel(x1, x2, X0, hps[1:1 + NB], hps[0]) + floor
    d = get_anisotropic_distance_matrix(x1, x2, hps[1 + NB:])
    return k_amp * matern_kernel_diff1(d, 1.0)
```

Watch the argument order — `non_stat_kernel(x1, x2, x0, w, l)` takes the **weight
vector before the scalar width**, which is easy to reverse. The `floor` term is not
cosmetic: without it, training wanders into weight vectors that zero out `g(x)` at a
data point and the run dies mid-optimization. More basis functions means more
flexibility and more hyperparameters — start with 3-5 and only add more if the fit
demands it.

**(b) Varying length scale (Gibbs kernel).** Smoothness itself changes across the
space — sharp features in one region, smooth background in another. The prefactor
below is what keeps it PSD; do not drop it:

```python
def gibbs_kernel_1d(x1, x2, hps):
    """
    1-D Gibbs kernel with a linearly varying length scale l(x) = a + b*|x|.

    hps[0] = a, base length scale     (> 0)
    hps[1] = b, growth rate           (>= 0; b = 0 recovers a stationary SE kernel)
    hps[2] = signal variance
    Total: 3 hyperparameters
    """
    a, b, sv = hps[0], hps[1], hps[2]
    l1 = a + b * np.abs(x1[:, 0])
    l2 = a + b * np.abs(x2[:, 0])
    L1, L2 = np.meshgrid(l1, l2, indexing="ij")
    prefactor = np.sqrt(2.0 * L1 * L2 / (L1**2 + L2**2))   # required for PSD
    d2 = np.subtract.outer(x1[:, 0], x2[:, 0])**2
    return sv * prefactor * np.exp(-d2 / (L1**2 + L2**2))
```

Keep `a` bounded away from zero (e.g. `[0.05, 5.0]`) — a length scale that collapses
toward zero is the most common cause of singular covariance matrices in
non-stationary models. Any `l(x)` that stays strictly positive works in place of the
linear form; for D > 1 use a per-dimension `l_i(x)` and take the product.

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

# Ask which of a candidate set to measure next.
# Keep n well below len(candidates) — asking for as many points as the set
# contains just returns the whole set and selects nothing.
candidates = ["who", "could", "it", "be", "hello", "there", "gpcam", "rules"]
gp.ask(candidates, n=2)
```

Notes:
- A Python `for` loop over objects is fine here (you're bottlenecked by the distance function, not numpy).
- If the distance function is symmetric, the resulting kernel is automatically symmetric; you still need the base kernel (Matérn/SE) to make it PSD.
- For multi-task on non-Euclidean inputs: `fvGPOptimizer(x_strings, y_multi, kernel_function=...)` works the same way.

### Deep Kernel (NN-warped input space)

For hard multi-task structure or learned metrics, warp the input through a small neural net and then apply a stationary kernel in the warped space. `gpcam.deep_kernel_network.Network` gives you an MLP whose weights you read out of `hps`.

**You must call `set_weights` / `set_biases` inside the kernel.** If you skip that step the network keeps its randomly initialized weights, `hps[2:]` have no effect on the covariance, and training silently optimizes over a flat likelihood — you get a fixed random feature map that looks trained but is not.

`Network(dim, layer_width)` has three `nn.Linear` layers, so the parameter block is:

| Slice | Shape | Count |
|---|---|---|
| `w1` (layer1 weight) | `(layer_width, dim)` | `layer_width * dim` |
| `w2` (layer2 weight) | `(layer_width, layer_width)` | `layer_width ** 2` |
| `w3` (layer3 weight) | `(dim, layer_width)` | `dim * layer_width` |
| `b1`, `b2` (biases) | `(layer_width,)` each | `2 * layer_width` |
| `b3` (layer3 bias) | `(dim,)` | `dim` |

which sums to `n.number_of_hps == 2*dim*layer_width + layer_width**2 + 2*layer_width + dim`.

```python
import numpy as np
from gpcam import GPOptimizer
from gpcam.deep_kernel_network import Network
from gpcam.kernels import get_distance_matrix, matern_kernel_diff1

ISET_DIM = 3
LAYER_WIDTH = 5
n = Network(ISET_DIM, LAYER_WIDTH)
N_NN = n.number_of_hps        # number of NN hyperparameters to reserve

def _load_network(nn_hps, dim=ISET_DIM, w=LAYER_WIDTH):
    """Unpack a flat hyperparameter slice into the network's weights and biases."""
    i = 0
    w1 = nn_hps[i:i + w * dim].reshape(w, dim);   i += w * dim
    w2 = nn_hps[i:i + w * w].reshape(w, w);       i += w * w
    w3 = nn_hps[i:i + dim * w].reshape(dim, w);   i += dim * w
    b1 = nn_hps[i:i + w];                         i += w
    b2 = nn_hps[i:i + w];                         i += w
    b3 = nn_hps[i:i + dim];                       i += dim
    assert i == len(nn_hps), f"expected {len(nn_hps)} NN hps, consumed {i}"
    n.set_weights(w1, w2, w3)
    n.set_biases(b1, b2, b3)

def deep_kernel(x1, x2, hps):
    """
    hps[0]  = signal variance
    hps[1]  = length scale (in the warped space)
    hps[2:] = NN weights and biases (length n.number_of_hps)
    """
    _load_network(hps[2:])            # <-- REQUIRED; without it hps[2:] do nothing
    d = get_distance_matrix(n.forward(x1), n.forward(x2))
    return hps[0] * matern_kernel_diff1(d, hps[1])

# Total hyperparameters: 2 + N_NN
init_hps = np.concatenate([[1.0, 1.0], np.random.randn(N_NN) * 0.5])
hp_bounds = np.array([[0.01, 10.0], [0.01, 10.0]] + [[-2.0, 2.0]] * N_NN)

gp = GPOptimizer(x_data, y_data, init_hyperparameters=init_hps,
                 kernel_function=deep_kernel)
gp.train(hyperparameter_bounds=hp_bounds, method="global", max_iter=100)
```

Sanity check that the unpacking is live — evaluate the kernel at two different NN hyperparameter draws and confirm the matrices differ:

```python
a = np.concatenate([[1., 1.], np.random.randn(N_NN)])
b = np.concatenate([[1., 1.], np.random.randn(N_NN)])
assert not np.allclose(deep_kernel(x1, x2, a), deep_kernel(x1, x2, b)), \
    "NN hyperparameters are not reaching the kernel"
```

Training notes: `method="mcmc"` or `method="adam"` scale better than `global`/`local` to NN-sized hyperparameter vectors. Bounds on NN weights should be symmetric around zero (e.g. `[-2, 2]`) — weights are not scale parameters and must be allowed to go negative.

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
