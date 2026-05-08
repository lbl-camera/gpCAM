---
name: multi-task-advanced
description: Use for multi-output, vector-valued, or function-valued gpCAM experiments using fvGPOptimizer — useful when a single measurement returns multiple correlated quantities (e.g., spectra, multi-channel detectors).
---

# Skill: Multi-Task GPs with fvGPOptimizer

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

# Custom kernel path — supply init_hyperparameters and hp bounds as with GPOptimizer:
gpo = fvGPOptimizer(
    x_data=x_data,
    y_data=y_data,
    init_hyperparameters=np.ones(4) / 10.0,
    kernel_function=my_multi_task_kernel,
)
gpo.train(hyperparameter_bounds=np.array([
    [0.01, 10.0],  # signal variance
    [0.01, 10.0],  # length scale dim 0
    [0.01, 10.0],  # length scale dim 1
    [0.01, 10.0],  # length scale for task dimension
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

For simple black-box vector-valued optimization, `optimize()` replaces the manual loop:
```python
def f(x):                      # x shape (N, D)
    y = np.column_stack([...])  # shape (N, T)
    noise = np.full(y.shape, 0.01)
    return y, noise

result = fvGPOptimizer().optimize(
    func=f,
    x_out=np.array([0, 1]),     # which task indices to treat as outputs
    search_space=np.array([[0., 1.]]),
    max_iter=50,
)
```

## Multi-Task Kernel Design

The kernel receives inputs with the task index as the last column. You need to model both within-task and between-task correlations:

```python
from gpcam.kernels import matern_kernel_diff1, get_distance_matrix

def multi_task_kernel(x1, x2, hps):
    """
    x1, x2: shape (N, D+1) where last column is task index
    hps[0]: signal variance
    hps[1:D+1]: input space length scales
    hps[D+1]: task correlation strength
    """
    # Spatial kernel (input dimensions only)
    d_spatial = get_distance_matrix(x1[:, :-1], x2[:, :-1])
    k_spatial = matern_kernel_diff1(d_spatial, hps[1])
    
    # Task kernel (last dimension)
    task1 = x1[:, -1]
    task2 = x2[:, -1]
    # Simple: same task = 1, different task = correlation
    same_task = np.equal.outer(task1, task2).astype(float)
    k_task = same_task + hps[2] * (1 - same_task)
    
    return hps[0] * k_spatial * k_task
```

## Important Notes

1. **Multi-task acquisition**: Use `"relative information entropy set"` / `"relative information entropy"` / `"variance"` / `"total correlation"` for batch acquisition across tasks. Pass `x_out=...` to the `ask()` call. Custom callables are supported and often advisable when you care about a specific task or combination.
2. **Missing task observations**: `y_data` can have `np.nan` entries (e.g., task 1 wasn't measured at some x); the corresponding `noise_variances` entry **must also** be `np.nan`. The GP just ignores those entries — no imputation needed.
3. **Default kernel**: `fvGPOptimizer(x, y)` with no kernel uses a built-in deep kernel that learns its own hyperparameters and doesn't require bounds. If you supply a custom `kernel_function`, you become responsible for the full `init_hyperparameters` + `hyperparameter_bounds` layout.
4. **Deep kernel via NN warping**: For harder multi-task structure, `from gpcam.deep_kernel_network import Network` gives you an MLP you parametrize from `hps` — see the `kernel-designer` skill for the pattern.

## Common Pitfalls

1. **Forgetting the task dimension**: The kernel sees D+1 columns, not D.
2. **Noise shape**: `noise_variances` is shape `(N, No)` for multi-task, not `(N,)`.
3. **Scaling**: N data points × No outputs = N×No rows in the internal GP. Gets large fast.
