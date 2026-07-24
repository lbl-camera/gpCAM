---
name: experiment-designer
description: Use for end-to-end autonomous experiment design with gpCAM. Translates a scientist's description of their measurement into a complete, runnable gpCAM script — useful for replacing raster scans with adaptive sampling, peak-finding, or parameter optimization.
gpcam_version: "8.4.x"
fvgp_version: "4.8.x"
last_verified: "2026-07-23 (gpCAM dadeb65)"
---

# Skill: gpCAM Experiment Designer

*Verified against gpCAM 8.4.x / fvgp 4.8.x — last checked 2026-07-23 (gpCAM `dadeb65`).*

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
| **Optimizer class** | Pick by the support of the observations: `GPOptimizer` for unconstrained or negative-allowed `y`; `LogGPOptimizer` if `y > 0` (intensities, rates, concentrations); `LogitGPOptimizer` if `y ∈ [0, 1]` (yields, fractions, probabilities). A plain GP on positive-only or bounded data can predict invalid values — see `transformed-optimizers-advanced` skill. |
| **Kernel** | Default Matérn-3/2 ARD is good for most cases. Use periodic kernel if periodicity is known. Use Matérn-1/2 for rough/discontinuous data, Matérn-5/2 or SE for very smooth. See `kernel-designer` skill for custom kernels. |
| **Acquisition function** | `'variance'` for exploration/mapping. `'expected improvement'` or `'ucb'` for optimization (UCB exposes a tunable exploration/exploitation tradeoff via `beta`). Custom callable for multi-objective or constraints. See `acquisition-functions` skill. |
| **Prior mean** | Default is a **constant equal to `mean(y_data)`** — not zero. Away from data the posterior reverts to that constant. Override with `prior_mean_function=` only if they have a physical model. See `prior-mean-functions` skill. |
| **Noise model** | Use `noise_variances` if noise is known and uniform. Use `noise_function` if noise varies. See `noise-functions` skill. |
| **Training strategy** | `method='global'` for first training, `method='local'` for re-training during the loop. Other options: `"mcmc"` (Bayesian — returns posterior samples over hyperparameters), `"adam"` (stochastic-gradient, fast, works well for high-dimensional hyperparameter vectors like deep kernels), `"hgdl"` (distributed local+global hybrid — needs a `dask_client`). |
| **linalg_mode** | Leave at the default (`None`) — gpCAM picks `"Chol"` automatically. For frequent posterior-covariance calls on small datasets (<5 000 points), set `linalg_mode="CholInv"` to store the inverse for a 3-10× speedup. For sparse / gp2Scale problems, see the `gp2scale-advanced` skill. |
| **Number of initial points** | Rule of thumb: 5-10× the input dimensionality for initial random sampling. |
| **Input scaling** | gpCAM does **not** normalize inputs internally. With mixed units (mm, °C, volts) each dimension keeps its own scale, so per-dimension length-scale bounds must be derived from that dimension's own range — which the template below does. See "Input Scaling" below. |
| **Validation** | `gpo.rmse`, `gpo.crps`, `gpo.nlpd`, and `gpo.coverage_curve` on a held-out set. Do not ship an error bar without checking calibration — see the `uncertainty-calibration` skill. |

### Step 3: Generate the Script

**Two paths:** If the scientist only needs a scalar black-box optimized and has no need to inspect or customize the ask/tell loop, use the one-shot `GPOptimizer.optimize()` shortcut (A). Otherwise use the full template (B). The full template is required when they want custom acquisition, mid-loop re-training with specific schedules, async training, checkpointing, validation plots during the run, or integration with a live instrument.

#### A. One-shot optimize (simplest)

```python
import numpy as np
from gpcam import GPOptimizer

def f(x):
    """
    optimize() calls this with a SINGLE point of shape (D,) during its initial
    sampling, and with shape (1, D) inside the loop. `np.atleast_2d` handles both.

    Returns (y, noise_variances), each of length len(x).
    """
    x = np.atleast_2d(x)
    y = np.sin(x[:, 0]) * np.cos(x[:, 1])
    return y, np.full(len(x), 0.01)

gpo = GPOptimizer()
result = gpo.optimize(
    func=f,
    search_space=np.array([[0., 1.], [0., 1.]]),
    max_iter=50,
)
# result: {'trace f(x)', 'trace x', 'f(x)', 'x'} — the traces plus the last point
```

**The shape contract is the thing to get right.** `optimize()` evaluates `func` with
`map(func, x0)` during initial sampling, so each call receives one point as a **1-D
array of shape `(D,)`** — not a batch. Inside the loop it passes `ask()`'s output,
shape `(1, D)`. A function written only for `(N, D)` input crashes on the first call
with `IndexError: too many indices for array`. Starting the body with
`x = np.atleast_2d(x)` makes it work in both cases.

`optimize()` handles initial sampling (10 random points), the training schedule
(`train_at=(10, 20, 50, 100, 200)` by default), the ask/tell loop, and termination.
Note `acq_func='lcb'` is the default, which **minimizes** `func` — pass
`acq_func='ucb'` to maximize. Other useful arguments: `hyperparameter_bounds=`
(required for any custom kernel/mean/noise), `callback=f(x_data, y_data)` called every
iteration, and `break_condition=f(x_data, y_data)` returning `True` to stop early.

Use it when the scientist has a simulator or instrument wrapper they can hand gpCAM as
a Python function. For fvGP/multi-task, pass `x_out=np.array([...])` and note the
return shape differs — see the `multi-task-advanced` skill.

#### B. Full template (adaptive loop)

Output a complete Python script with this structure:

```python
"""
Autonomous Experiment: [description]
Generated for gpCAM v8.4.x

Input space: [dimensions and ranges]
Output: [what is measured]
Strategy: [exploration/optimization/hybrid]
"""

import numpy as np
from gpcam import GPOptimizer
# For strictly positive observations use LogGPOptimizer; for y in [0, 1] use
# LogitGPOptimizer (drop-in replacements for GPOptimizer — see the
# transformed-optimizers-advanced skill).

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
# 3b. PRIOR MEAN (optional customization)
# ============================================================
# Default: a constant equal to mean(y_data) — NOT zero. Away from data the
# posterior reverts to that constant. Override only with a real physical model.
# Mean hyperparameters start at index K = 1 + D (after the default kernel's).
#
# D_ = parameter_bounds.shape[0]
# def my_mean(x, hps):
#     K = 1 + D_                      # derive it; never hardcode 3
#     return hps[K] + x @ hps[K+1:K+1+D_]     # linear trend
# ...and add one bounds row per mean hyperparameter in section 4.

prior_mean_function = None  # None = constant mean(y_data)

# ============================================================
# 4. HYPERPARAMETER BOUNDS
# ============================================================
# Layout convention: KERNEL first, then MEAN, then NOISE. The ranges each
# callable reads must be disjoint.
#
# hps[0]     = signal variance   (kernel)
# hps[1:D+1] = length scales     (kernel)
# hps[D+1:]  = mean, then noise hyperparameters, if you add any
#
# Rule of thumb:
#   signal_variance bounds: [0.01, 10 * std(y)]  (estimated after initial data)
#   length_scale bounds:    [0.01, 10 * range(x_dim)]   <- per dimension, so
#                           mixed units are handled without rescaling inputs
D = parameter_bounds.shape[0]
hp_bounds = np.array(
    [[0.001, 100.0]] +                                          # signal variance
    [[0.01, 10.0 * (b[1] - b[0])] for b in parameter_bounds]   # length scales
    # + [[-10.0, 10.0]] ...                                    # mean hps, if any
    # + [[0.001, 10.0]]                                        # noise hps, if any
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
        prior_mean_function=prior_mean_function,
        # linalg_mode=None lets gpCAM pick Cholesky; for many posterior_covariance
        # calls on a small dataset (<5 000 points) use linalg_mode="CholInv".
    )
    
    # Cheap guard against the most common bug: bounds not matching the number of
    # hyperparameters the kernel/mean/noise functions actually read.
    assert len(gpo.hyperparameters) == len(hp_bounds), (
        f"{len(hp_bounds)} bounds rows but {len(gpo.hyperparameters)} hyperparameters"
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

- The hyperparameter vector is shared across kernel, mean, and noise functions, and the index ranges each one reads **must be disjoint**. fvGP assumes a hyperparameter belonging to the mean has zero kernel derivative and vice versa, so an overlap corrupts the training gradients as well as the model.
- **Standard ordering: kernel, then mean, then noise.** Every gpCAM skill uses this layout.
- For the default kernel with D input dimensions: `hps[0]` = signal variance, `hps[1:D+1]` = length scales. So custom mean/noise hyperparameters start at `K = 1 + D`.
- **Derive `K = 1 + D`; never hardcode it.** `K = 3` is correct only for 2-D input. In 1-D it aliases a length scale; in 3-D two components silently share a hyperparameter.
- **Never read `hps[-1]` in a prior mean function** — the end of the vector belongs to the noise function.
- Assert `len(gpo.hyperparameters) == len(hp_bounds)` after construction; it catches most layout mistakes for one line.
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
    K = 1 + x.shape[1]              # = D + 1; derived, not hardcoded
    return np.full(len(x), hps[K]**2)

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

### User-function arguments (`args`)

`GPOptimizer(..., args={"a": 1.5})` plumbs a dict to **kernel, prior mean, and noise
functions only**. Dispatch is by **parameter count** (`inspect.signature`), not by
name — declare one extra positional parameter and gpCAM passes the dict; omit it and
it doesn't. The parameter name is yours to choose.

| Callable | Without `args` | With `args` |
|---|---|---|
| Kernel | `f(x1, x2, hps)` | `f(x1, x2, hps, args)` |
| Prior mean | `f(x, hps)` | `f(x, hps, args)` |
| Noise | `f(x, hps)` | `f(x, hps, args)` |
| **Cost** | `f(origin, x)` | **not supported** — see below |
| **Acquisition** | `f(x, gp_obj)` | **not supported** — see below |

```python
def my_kernel(x1, x2, hps, args):        # 4 params -> gets args
    return hps[0] * matern_kernel_diff1(
        get_anisotropic_distance_matrix(x1, x2, hps[1:]), args["nu_scale"])

def my_mean(x, hps, args):               # 3 params -> gets args
    return np.full(len(x), args["baseline"])

def my_noise(x, hps, args):              # 3 params -> gets args
    return np.full(len(x), args["floor"] + hps[K]**2)

gpo = GPOptimizer(x_data, y_data, args={"nu_scale": 1.0, "baseline": 3.0, "floor": 1e-4},
                  kernel_function=my_kernel, prior_mean_function=my_mean,
                  noise_function=my_noise)
```

**Cost and acquisition functions never receive `args`.** gpCAM calls them as exactly
`cost_function(origin, x)` and `acquisition_function(x, gp_obj)`. A cost function
written as `f(origin, x, arguments=None)` will run, but `arguments` stays `None`
forever. Bind their parameters with a closure or `functools.partial` instead.

Wrong signature → `Exception("No valid kernel function signature")` (or the mean /
noise equivalent) at construction, so mistakes here fail loudly rather than silently.

`gpo.set_args(new_dict)` updates the dict later, but does **not** invalidate the
cached covariance — follow it with `gpo.set_hyperparameters(gpo.hyperparameters)` to
force the callables to re-run.

### Input Scaling

gpCAM does not rescale `x_data`. The default ARD kernel learns one length scale per
dimension, so unequal units are handled *if the bounds allow it* — a dimension
spanning 0-1000 V needs a length-scale upper bound near 1000, not near 1. The
template's `[[0.01, 10.0 * (b[1] - b[0])] for b in parameter_bounds]` does this
automatically; keep that derivation if you change the bounds.

If you prefer to normalize, scale inputs to the unit cube yourself before constructing
the GP and remember to map `ask()` results back:

```python
lo, hi = parameter_bounds[:, 0], parameter_bounds[:, 1]
to_unit = lambda X: (X - lo) / (hi - lo)
from_unit = lambda U: U * (hi - lo) + lo

gpo = GPOptimizer(to_unit(x_raw), y_data)
next_x = from_unit(gpo.ask(np.array([[0., 1.]] * D))["x"])   # back to physical units
```
Normalizing makes a single shared length-scale bound (`[0.01, 10.0]`) valid for every
dimension, which is convenient with custom isotropic kernels. It is optional — the
default ARD kernel does not require it.

### Convergence and Termination

`optimize()` exposes only `max_iter`; the full template gives you the loop, so put the
stopping rule there. gpCAM has no built-in convergence criterion — a fixed budget is
the honest default when measurements are the scarce resource. Useful signals:

```python
history = []
for i in range(N_ITERATIONS):
    result = gpo.ask(input_set=parameter_bounds, acquisition_function=acquisition_function)
    next_x, acq_value = result["x"], result["f_a(x)"]
    history.append(float(np.max(acq_value)))
    ...
    # (a) Acquisition floor — for "variance", this is a real uncertainty target
    #     in the units of y**2, so it can be set from the measurement precision.
    if acquisition_function == "variance" and history[-1] < TARGET_VARIANCE:
        print(f"Converged: posterior variance below {TARGET_VARIANCE}"); break

    # (b) Stalled improvement — no better observation in the last P iterations
    if len(gpo.y_data) > PATIENCE and np.argmax(gpo.y_data) < len(gpo.y_data) - PATIENCE:
        print("Stalled: no improvement in the last PATIENCE measurements"); break
```

Distinguishing converged from stalled is the point: a **falling** acquisition trace
means the model is genuinely running out of things to learn; a **flat but high** trace
usually means the hyperparameters are stale (retrain more often) or the length scales
have collapsed so every point looks equally uninformative. Plot `history` before
trusting either stopping rule, and check `coverage_curve` — an overconfident model
reports low variance everywhere and will stop early for the wrong reason.

## Reference

For detailed API docs, kernel math, and advanced options, see the [gpCAM documentation](https://gpcam.readthedocs.io).
