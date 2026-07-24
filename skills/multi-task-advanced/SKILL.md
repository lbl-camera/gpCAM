---
name: multi-task-advanced
description: Use for multi-output, vector-valued, or function-valued gpCAM experiments using fvGPOptimizer — useful when a single measurement returns multiple correlated quantities (e.g., spectra, multi-channel detectors).
gpcam_version: "8.4.x"
fvgp_version: "4.8.x"
last_verified: "2026-07-23 (gpCAM dadeb65)"
---

# Skill: Multi-Task GPs with fvGPOptimizer

*Verified against gpCAM 8.4.x / fvgp 4.8.x — last checked 2026-07-23 (gpCAM `dadeb65`).*

Design experiments with vector-valued or function-valued outputs using gpCAM's multi-task GP.

## When to Use

- Measuring a spectrum (many output channels per input point)
- Multiple correlated outputs (e.g., intensity at different energies)
- Exploiting correlations between tasks to improve predictions

## Key Concept

In fvGP, a multi-task GP is a single-task GP over the **Cartesian product** of input space × output space. The output dimension is appended as an extra column to the input.

For example, with 2D input and 3 output channels, a point looks like:
```
[x0, x1, task_id]  where task_id ∈ {0, 1, 2}
```

This means your kernel must handle D+1 dimensional inputs, where the last dimension is the task index.

## Basic Setup

```python
import numpy as np
from gpcam import fvGPOptimizer

# 100 input points, 5 output channels
x_data = np.random.uniform(0, 1, (100, 2))  # 2D input
y_data = np.random.randn(100, 5)             # 5 outputs per point

# Default path — uses a built-in deep kernel, no hyperparameter bounds required:
gpo = fvGPOptimizer(x_data, y_data)
gpo.train(max_iter=20)

# Custom kernel path — supply init_hyperparameters and hp bounds as with GPOptimizer.
# The layout below matches `multi_task_kernel` in "Multi-Task Kernel Design" below:
# D + 2 hyperparameters for D input dimensions.
gpo = fvGPOptimizer(
    x_data=x_data,
    y_data=y_data,
    init_hyperparameters=np.ones(4) / 10.0,
    kernel_function=multi_task_kernel,
)
gpo.train(hyperparameter_bounds=np.array([
    [0.01, 100.0],  # hps[0] signal variance
    [0.01, 10.0],   # hps[1] length scale dim 0
    [0.01, 10.0],   # hps[2] length scale dim 1
    [0.0, 1.0],     # hps[3] task correlation  (index D+1, D=2)
]))
```

### Predictions and ask — the `x_out` argument

Multi-task prediction methods take `x_out`, an array of task indices you want predictions for:

```python
# Predict all 5 task outputs at a grid of input points:
mean = gpo.posterior_mean(x_grid, x_out=np.array([0, 1, 2, 3, 4]))["m(x)"]  # shape (N, 5)
std  = np.sqrt(gpo.posterior_covariance(x_grid, x_out=np.array([0, 1, 2, 3, 4]))["v(x)"])

# Ask for the next best input point across all tasks:
gpo.ask(parameter_bounds, x_out=np.array([0, 1, 2, 3, 4]), n=1)

# Ask for a batch of 4 points using a batch-aware acquisition:
gpo.ask(parameter_bounds, x_out=np.array([0, 1]), n=4,
        acquisition_function="relative information entropy set", vectorized=True)
```

### One-shot optimize

For simple black-box vector-valued optimization, `optimize()` replaces the manual loop.

**The shape contract is asymmetric and is the easiest thing to get wrong.**
`optimize()` calls `func` with a **single point of shape `(D,)`** during initial
sampling (via `map`), then with **shape `(1, D)`** inside the loop. For multi-task the
return shape must follow suit: a **1-D vector of length T** for a single point, and
**`(n, T)`** for a batch. Returning `(1, T)` for the single-point case fails with
`AssertionError: updated x and y do not have the same lengths`.

```python
def f(x):
    """
    Single point (D,)  -> returns (T,)  values and (T,)  noise
    Batch      (n, D)  -> returns (n,T) values and (n,T) noise
    """
    single = np.ndim(x) == 1
    xa = np.atleast_2d(x)
    y = np.column_stack([np.sin(xa[:, 0]), np.cos(xa[:, 0])])   # (n, T)
    v = np.full(y.shape, 0.01)
    return (y[0], v[0]) if single else (y, v)

result = fvGPOptimizer().optimize(
    func=f,
    x_out=np.array([0, 1]),     # which task indices to treat as outputs
    search_space=np.array([[0., 1.]]),
    max_iter=50,
)
```

If you would rather not reason about this, use the explicit ask/tell loop instead —
there you control every call and `tell()` takes a plain `(n, D)` / `(n, T)` pair.

## Multi-Task Kernel Design

The kernel receives inputs with the task index as the last column. You need to model both within-task and between-task correlations:

```python
from gpcam.kernels import matern_kernel_diff1, get_anisotropic_distance_matrix

def multi_task_kernel(x1, x2, hps):
    """
    ARD over the input dimensions, plus a task-correlation term.

    x1, x2: shape (N, D+1), where the last column is the task index.

    hps[0]     : signal variance
    hps[1:D+1] : input-space length scales (one per input dimension)
    hps[D+1]   : task correlation strength, in [0, 1]

    Total: D + 2 hyperparameters.
    """
    D = x1.shape[1] - 1                 # strip the task column

    # Spatial kernel — ARD over the input dimensions only
    d_spatial = get_anisotropic_distance_matrix(x1[:, :D], x2[:, :D], hps[1:D+1])
    k_spatial = matern_kernel_diff1(d_spatial, 1.0)

    # Task kernel — same task = 1, different task = hps[D+1]
    same_task = np.equal.outer(x1[:, -1], x2[:, -1]).astype(float)
    k_task = same_task + hps[D+1] * (1.0 - same_task)

    return hps[0] * k_spatial * k_task
```

The task-correlation hyperparameter must be bounded to `[0, 1]`. Values above 1 make
the task block non-PSD; negative values are representable but rarely what you want
and make the matrix indefinite for more than two tasks.

Matching bounds for the D = 2, 5-task example above:

```python
D = 2
hp_bounds = np.array(
    [[0.01, 100.0]] +           # hps[0]   signal variance
    [[0.01, 10.0]] * D +        # hps[1:3] length scales, one per input dim
    [[0.0, 1.0]]                # hps[3]   task correlation  <- index D+1
)
gpo = fvGPOptimizer(x_data, y_data,
                    init_hyperparameters=np.ones(D + 2),
                    kernel_function=multi_task_kernel)
gpo.train(hyperparameter_bounds=hp_bounds)

assert len(gpo.hyperparameters) == len(hp_bounds)
```

Note that `hps[D+1]` is the task index **only because** the kernel reads `D` length
scales. If you simplify to a single isotropic length scale, the task correlation moves
to `hps[2]` and the bounds array shrinks accordingly — keep the docstring, the code,
and the bounds in agreement, or you will train a hyperparameter that does nothing
while reading a length scale as a task correlation.

## Important Notes

1. **Multi-task acquisition**: Use `"relative information entropy set"` / `"relative information entropy"` / `"variance"` / `"total correlation"` for batch acquisition across tasks. Pass `x_out=...` to the `ask()` call. Custom callables are supported and often advisable when you care about a specific task or combination.
2. **Missing task observations**: `y_data` can have `np.nan` entries (e.g., task 1 wasn't measured at some x); the corresponding `noise_variances` entry **must also** be `np.nan`. The GP just ignores those entries — no imputation needed.
3. **Default kernel**: `fvGPOptimizer(x, y)` with no kernel uses a built-in deep kernel that learns its own hyperparameters and doesn't require bounds. If you supply a custom `kernel_function`, you become responsible for the full `init_hyperparameters` + `hyperparameter_bounds` layout.
4. **Deep kernel via NN warping**: For harder multi-task structure, `from gpcam.deep_kernel_network import Network` gives you an MLP you parametrize from `hps` — see the `kernel-designer` skill for the pattern.

## Common Pitfalls

1. **Forgetting the task dimension**: The kernel sees D+1 columns, not D.
2. **Noise shape**: `noise_variances` is shape `(N, No)` for multi-task, not `(N,)`.
3. **Scaling**: N data points × No outputs = N×No rows in the internal GP. Gets large fast.
