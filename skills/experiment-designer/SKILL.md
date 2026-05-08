---
name: experiment-designer
description: Use for end-to-end autonomous experiment design with gpCAM. Translates a scientist's description of their measurement into a complete, runnable gpCAM script — useful for replacing raster scans with adaptive sampling, peak-finding, or parameter optimization.
---

# Skill: gpCAM Experiment Designer

Design complete autonomous experiment scripts using gpCAM. You translate a scientist's description of their measurement into a runnable Python script.

## When to Use

When a user wants to:
- Set up an autonomous/smart scan or optimization
- Replace a raster scan with adaptive sampling
- Find optimal experimental conditions (peak finding, parameter optimization)
- Explore a parameter space efficiently

## Your Role

You are helping **beamline scientists** who may not know GP math or the gpCAM API. Your job is to:
1. Understand their experiment (what they measure, what they control, what they want to find)
2. Generate a complete, well-commented Python script they can adapt
3. Explain the key choices you made in plain language

## Conversation Flow

### Step 1: Understand the Experiment
Ask about:
- **Input dimensions**: What parameters do they control? (motor positions, temperature, voltage, etc.)
- **Input bounds**: What range for each parameter?
- **Output**: What do they measure? (intensity, spectrum, image, scalar?)
- **Goal**: Exploration (map everything)? Optimization (find the peak)? Both?
- **Constraints**: Any forbidden regions? Cost of moving between points?
- **Prior knowledge**: Do they know roughly what to expect? (smooth? periodic? sharp features?)
- **Data size**: How many measurements can they afford? (determines if gp2Scale is needed)
- **Noise**: Is the measurement noisy? Does noise vary across the parameter space?

### Step 2: Design Choices
Based on their answers, decide:

| Choice | Guidance |
|--------|----------|
| **Kernel** | Default Matérn-3/2 ARD is good for most cases. Use periodic kernel if periodicity is known. Use Matérn-1/2 for rough/discontinuous data, Matérn-5/2 or SE for very smooth. See `kernel-designer` skill for custom kernels. |
| **Acquisition function** | `'variance'` for exploration/mapping. `'expected improvement'` or `'ucb'` for optimization (UCB exposes a tunable exploration/exploitation tradeoff via `beta`). Custom callable for multi-objective or constraints. See `acquisition-functions` skill. |
| **Prior mean** | Zero (default) unless they have a physical model. See `prior-mean-functions` skill. |
| **Noise model** | Use `noise_variances` if noise is known and uniform. Use `noise_function` if noise varies. See `noise-functions` skill. |
| **Training strategy** | `method='global'` for first training, `method='local'` for re-training during the loop. Other options: `"mcmc"` (Bayesian — returns posterior samples over hyperparameters), `"adam"` (stochastic-gradient, fast, works well for high-dimensional hyperparameter vectors like deep kernels), `"hgdl"` (distributed local+global hybrid — needs a `dask_client`). |
| **calc_inv** | `True` if <2000 points, `False` otherwise. |
| **Number of initial points** | Rule of thumb: 5-10× the input dimensionality for initial random sampling. |
| **Validation** | `gpo.rmse(x_test, y_true)` and `gpo.crps(x_test, y_true)` give RMSE and continuous ranked probability score on a held-out grid — call these after training to sanity-check the fit. |

### Step 3: Generate the Script

**Two paths:** If the scientist only needs a scalar black-box optimized and has no need to inspect or customize the ask/tell loop, use the one-shot `GPOptimizer.optimize()` shortcut (A). Otherwise use the full template (B). The full template is required when they want custom acquisition, mid-loop re-training with specific schedules, async training, checkpointing, validation plots during the run, or integration with a live instrument.

#### A. One-shot optimize (simplest)

```python
import numpy as np
from gpcam import GPOptimizer

def f(x):
    # x is shape (N, D); return (y, noise_variance) with matching shapes
    y = np.sin(x[:, 0]) * np.cos(x[:, 1])
    return y, np.full(len(x), 0.01)

gpo = GPOptimizer()
result = gpo.optimize(
    func=f,
    search_space=np.array([[0., 1.], [0., 1.]]),
    max_iter=50,
)
```

`optimize()` handles initial sampling, training schedule, the ask/tell loop, and termination. Use it when the scientist has a simulator or instrument wrapper they can hand gpCAM as a Python function. For fvGP/multi-task, pass `x_out=np.array([...])`.

#### B. Full template (adaptive loop)

Output a complete Python script with this structure:

```python
"""
Autonomous Experiment: [description]
Generated for gpCAM v8.3.x

Input space: [dimensions and ranges]
Output: [what is measured]
Strategy: [exploration/optimization/hybrid]
"""

import numpy as np
from gpcam import GPOptimizer

# ============================================================
# 1. EXPERIMENT PARAMETERS — EDIT THESE
# ============================================================
# Define the parameter space bounds
# Each row: [min, max] for one dimension
parameter_bounds = np.array([
    [0.0, 10.0],   # motor_x (mm)
    [0.0, 5.0],    # motor_y (mm)
])
parameter_names = ["motor_x", "motor_y"]

N_INITIAL = 10      # Initial random measurements
N_ITERATIONS = 50   # Adaptive measurements
RETRAIN_EVERY = 10  # Re-train hyperparameters every N iterations

# ============================================================
# 2. YOUR MEASUREMENT FUNCTION — REPLACE THIS
# ============================================================
def measure(x):
    """
    Replace this with your actual measurement.
    
    Parameters
    ----------
    x : np.ndarray, shape (1, D)
        The point to measure. x[0, 0] is motor_x, x[0, 1] is motor_y.
    
    Returns
    -------
    y : float
        The measured value (scalar).
    noise_variance : float or None
        The estimated variance of this measurement, or None if unknown.
    """
    # EXAMPLE: replace with your instrument call
    y = np.sin(x[0, 0]) * np.cos(x[0, 1])
    noise_variance = 0.01  # or None
    return y, noise_variance

# ============================================================
# 3. KERNEL (optional customization)
# ============================================================
# The default ARD Matérn-3/2 kernel is used if kernel_function=None.
# Uncomment and modify to use a custom kernel.
# from gpcam.kernels import matern_kernel_diff1, get_anisotropic_distance_matrix
# 
# def my_kernel(x1, x2, hyperparameters):
#     d = get_anisotropic_distance_matrix(x1, x2, hyperparameters[1:])
#     return hyperparameters[0] * matern_kernel_diff1(d, 1.0)

kernel_function = None  # None = default ARD Matérn-3/2

# ============================================================
# 4. HYPERPARAMETER BOUNDS
# ============================================================
# For the default kernel: [signal_variance, length_scale_dim1, ..., length_scale_dimD]
# Rule of thumb:
#   signal_variance bounds: [0.01, 10 * std(y)]  (estimated after initial data)
#   length_scale bounds:    [0.01, 10 * range(x_dim)]
D = parameter_bounds.shape[0]
hp_bounds = np.array(
    [[0.001, 100.0]] +                                          # signal variance
    [[0.01, 10.0 * (b[1] - b[0])] for b in parameter_bounds]   # length scales
)

# ============================================================
# 5. ACQUISITION FUNCTION
# ============================================================
acquisition_function = "variance"  # Options: "variance", "expected improvement",
                                   #          "ucb", "relative information entropy",
                                   #          or a callable

# ============================================================
# 6. RUN THE EXPERIMENT
# ============================================================
def run():
    # --- Initial random sampling ---
    x_init = np.random.uniform(
        parameter_bounds[:, 0], parameter_bounds[:, 1],
        size=(N_INITIAL, D)
    )
    y_init = np.zeros(N_INITIAL)
    noise_init = np.zeros(N_INITIAL)
    
    for i in range(N_INITIAL):
        y_init[i], nv = measure(x_init[i:i+1])
        noise_init[i] = nv if nv is not None else 0.0
    
    # --- Initialize GP ---
    gpo = GPOptimizer(
        x_data=x_init,
        y_data=y_init,
        noise_variances=noise_init if noise_init.any() else None,
        kernel_function=kernel_function,
        calc_inv=(N_INITIAL + N_ITERATIONS < 2000),
    )
    
    # --- Initial training ---
    gpo.train(hyperparameter_bounds=hp_bounds, method="global", max_iter=200)
    
    # --- Adaptive loop ---
    for i in range(N_ITERATIONS):
        # Ask: where should we measure next?
        result = gpo.ask(
            input_set=parameter_bounds,
            acquisition_function=acquisition_function,
        )
        next_x = result["x"]
        
        # Measure
        new_y, new_nv = measure(next_x)
        
        # Tell: update the GP
        gpo.tell(
            next_x,
            np.array([new_y]),
            noise_variances=np.array([new_nv]) if new_nv is not None else None,
        )
        
        # Re-train periodically
        if (i + 1) % RETRAIN_EVERY == 0:
            gpo.train(hyperparameter_bounds=hp_bounds, method="local", max_iter=100)
        
        print(f"Iteration {i+1}/{N_ITERATIONS}: "
              f"measured at {next_x[0]} -> {new_y:.4f}")
    
    # --- Results ---
    data = gpo.get_data()
    print(f"\nDone! Collected {len(data['x data'])} points total.")
    print(f"Final hyperparameters: {data['hyperparameters']}")
    
    return gpo

if __name__ == "__main__":
    gpo = run()
```

## Key Rules

1. **Always generate a complete, runnable script** — not fragments. Scientists should be able to copy-paste and run it.
2. **The `measure()` function is a placeholder** — clearly mark it and explain what to replace.
3. **Comment heavily** — explain every choice for the non-expert.
4. **Hyperparameter bounds matter** — set sensible defaults based on the parameter ranges and expected signal scale.
5. **Default kernel is usually fine** — only suggest custom kernels when there's a clear reason (known periodicity, symmetry, etc.).
6. **Training schedule** — train globally once at the start, then locally every N iterations. Don't train every iteration (too slow).
7. **Initial points** — always start with random initial sampling before the adaptive loop.

## Hyperparameter Coordination

This is critical and often the source of bugs:

- The hyperparameter vector is shared across kernel, mean, and noise functions
- For the default kernel with D input dimensions: `hps[0]` = signal variance, `hps[1:D+1]` = length scales
- If you add a custom noise function that uses hyperparameters, those come AFTER the kernel hyperparameters
- If you add a prior mean function that uses hyperparameters, document which indices it uses
- **Always document the hyperparameter layout** in a comment at the top of the script
- **Always set bounds for ALL hyperparameters** — the bounds array must match the total hyperparameter count

Example with custom noise:
```python
# Hyperparameter layout:
# hps[0]     = signal variance (kernel)
# hps[1:D+1] = length scales (kernel)  
# hps[D+1]   = noise amplitude (noise function)
#
# Total: D + 2 hyperparameters

def my_noise(x, hps):
    return np.full(len(x), hps[D+1]**2)

hp_bounds = np.array(
    [[0.001, 100.0]] +                                        # signal variance
    [[0.01, 10.0 * (b[1]-b[0])] for b in parameter_bounds] +  # length scales
    [[0.001, 10.0]]                                            # noise amplitude
)
```

## Advanced Options (mention only if needed)

- **Incremental data updates**: `gpo.tell(x_new, y_new, append=True)` adds points without overwriting; `append=False` replaces. Default is to replace — use `append=True` in a streaming instrument loop.
- **Async training**: `opt_obj = gpo.train(..., asynchronous=True, method="hgdl", dask_client=client)` returns immediately; later call `gpo.update_hyperparameters(opt_obj)` to pull current best hyperparameters, and `gpo.stop_training(opt_obj)` to finish. Useful when training is expensive and the loop shouldn't block.
- **Async ask**: `method="hgdlAsync"` starts a background search for the next point; the result dict contains an `opt_obj` you can `kill_client()` once you've used the suggestion.
- **Checkpointing**: `GPOptimizer` instances are picklable — `pickle.dumps(gpo)` before a long run lets you reload state later.
- **Info measures**: `gpo.gp_mutual_information(x_test)` and `gpo.gp_total_correlation(x_test)` report information content at a candidate set.
- **User-function arguments**: `GPOptimizer(..., args={"a": 1.5, "b": 2.0})` plumbs a dict through to custom kernel/mean/noise/cost functions (they receive it as an extra argument when they declare it).

## Reference

For detailed API docs, kernel math, and advanced options, see the [gpCAM documentation](https://gpcam.readthedocs.io).
