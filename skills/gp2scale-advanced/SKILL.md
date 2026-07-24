---
name: gp2scale-advanced
description: Use for large-scale gpCAM experiments (>10k points up to millions) using sparse compactly-supported kernels and Dask distributed computing for exact GP computation at scale.
gpcam_version: "8.4.x"
fvgp_version: "4.8.x"
last_verified: "2026-07-23 (gpCAM dadeb65)"
---

# Skill: gp2Scale — Large-Scale GPs

*Verified against gpCAM 8.4.x / fvgp 4.8.x — last checked 2026-07-23 (gpCAM `dadeb65`).*

Design experiments with tens of thousands to millions of data points using gpCAM's gp2Scale mode for exact GP computation at scale.

## When to Use

- Dataset will exceed ~10,000 points (and can go into the millions with enough cluster)
- Need exact GP inference — gp2Scale is **not** an approximation like Vecchia or inducing-point methods; it only exploits naturally-occurring sparsity induced by compact-support kernels, so the result is exact
- Have access to multiple CPU cores, GPUs, or a compute cluster
- Willing to use compactly-supported kernels (Wendland)

## Key Concepts

gp2Scale uses:
1. **Wendland kernels** with compact support → sparse covariance matrix (zero covariance beyond the support radius)
2. **Dask distributed** for parallel covariance computation — one covariance block per worker
3. **Sparse linear algebra** (LU, Cholesky) via SciPy / imate instead of dense
4. **Random linear algebra** for log-determinants at scale (install `imate`)

## Basic Setup

```python
from distributed import Client
from gpcam import GPOptimizer

# Start a local Dask cluster
client = Client()                  # uses all available cores
client.wait_for_workers(4)         # good practice: wait for workers before constructing

gpo = GPOptimizer(
    x_data=x_data,
    y_data=y_data,
    gp2Scale=True,
    dask_client=client,
    # gp2Scale_batch_size defaults to 10000 — leave it alone unless profiling
    # says otherwise. See "Choosing the Batch Size" below.
    init_hyperparameters=np.array([0.73, 0.0014]),  # signal var, length scale
)

gpo.train(hyperparameter_bounds=hps_bounds, max_iter=25, info=True)
```

## Choosing the Batch Size

`gp2Scale_batch_size` is the side length of the covariance sub-blocks handed to Dask
workers. **The default is 10000 and is the right starting point.** Blocks are
`batch_size × batch_size`, so the number of tasks scales as `(N / batch_size)²` —
dropping the batch size by 20× multiplies the task count by 400 and the run becomes
scheduler-bound rather than compute-bound.

Tune only if profiling shows a problem, and in this direction:

| Symptom | Change |
|---|---|
| Dask dashboard shows tiny tasks, scheduler pegged, workers idle | **Increase** batch size |
| Workers running out of memory on a single block | **Decrease** batch size |
| Many workers idle because there are fewer blocks than workers | **Decrease**, so `(N / batch_size)² ≳ n_workers` |

The constraint is that one block must fit comfortably in one worker's memory
(`batch_size² × 8` bytes for float64 — 10000 gives an 800 MB dense block, which is why
the compact-support kernel producing a *sparse* block matters), while still producing
enough blocks to keep every worker busy.

Estimate the covariance-assembly time before committing to a long run — time one
block on one worker, then:

```python
# time_per_worker_execution: seconds for one worker to compute one block
gpo.get_gp2Scale_exec_time(time_per_worker_execution=0.05, number_of_workers=32)
# returns D**2 * tb / (2 * n * b**2), with D = len(x_data), b = batch size
```

## Kernel Requirement

When `gp2Scale=True`, the kernel MUST produce a sparse matrix. The default switches to an anisotropic Wendland kernel automatically if no custom kernel is provided.

If providing a custom kernel, it must have compact support:
```python
from gpcam.kernels import wendland_anisotropic

def my_gp2scale_kernel(x1, x2, hps):
    """Custom kernel with compact support for gp2Scale."""
    return wendland_anisotropic(x1, x2, hps)
```

## Hyperparameters for Wendland Kernel

```
hps[0]     = signal variance
hps[1:D+1] = per-dimension length scales (also control support radius)
```

The length scales in the Wendland kernel also determine the support radius — points further apart than the length scale have zero covariance.

## Linear Algebra Modes

Leave `linalg_mode=None` (the default) and gpCAM picks the best sparse mode
automatically. Override only when profiling justifies it:

| Mode | Description |
|------|-------------|
| `"sparseLU"` | Sparse LU factorization — good default for sparse systems up to ~50 000 points |
| `"sparseSolve"` | Direct sparse solve via SciPy |
| `"sparseCG"` | Conjugate gradient (iterative) |
| `"sparseMINRES"` | MINRES (iterative) |
| `"sparseCGpre"` | Preconditioned CG — for large, poorly conditioned systems |
| `"sparseMINRESpre"` | Preconditioned MINRES |

Preconditioned modes take their preconditioner from
`args["sparse_preconditioner_type"]` (default `"ilu"`), or use the
`"sparseCGpre_<type>"` shortcut, e.g. `linalg_mode="sparseCGpre_amg"` (needs `pyamg`).
The incomplete-Cholesky options `"ichol"` / `"ichol0"` need the optional `ilupp`
package and raise a clear `ImportError` with install instructions if it is missing;
`"native_ic"` is a slower pure-Python fallback that always works.

**`"Chol"`, `"CholInv"`, and `"Inv"` are the dense (non-gp2Scale) modes** — do not use
them here.

## Custom Block-MCMC Training

For expensive gp2Scale likelihoods, standard `method="mcmc"` training may be too slow. gpCAM exposes a block Metropolis-Hastings sampler you can drive directly against the GP's log-likelihood:

```python
import numpy as np
from gpcam import gpMCMC, ProposalDistribution

def in_bounds(v, bounds):
    return not (any(v < bounds[:, 0]) or any(v > bounds[:, 1]))

def prior_function(theta, args):
    return 0.0 if in_bounds(theta, args["bounds"]) else -np.inf

def log_likelihood(hps, args):
    return gpo.log_likelihood(hyperparameters=hps)   # exposed on GPOptimizer

pd = ProposalDistribution([0, 1], init_prop_Sigma=np.identity(2) * 0.01)

mcmc = gpMCMC(log_likelihood, prior_function, [pd],
              args={"bounds": hps_bounds})
result = mcmc.run_mcmc(x0=np.array([1.0, 0.01]), n_updates=200, info=True)

gpo.set_hyperparameters(result["mean(x)"])
```

`ProposalDistribution` takes the list of hyperparameter indices in that block and an initial proposal covariance. Stack multiple `ProposalDistribution`s for block-wise updates of high-dimensional hyperparameter vectors (deep kernels, etc.).

## HPC Setup (SLURM example)

```python
from dask_jobqueue import SLURMCluster
from distributed import Client

cluster = SLURMCluster(
    cores=32,
    memory="64GB",
    walltime="01:00:00",
)
cluster.scale(jobs=4)  # 4 nodes × 32 cores
client = Client(cluster)
```

## Common Pitfalls

1. **Using a non-sparse kernel**: gp2Scale won't speed up dense kernels — the kernel must have compact support.
2. **Batch size too small**: Task count grows as `(N / batch_size)²`, so a small batch size buries the scheduler in tiny tasks. Keep the default of 10000 unless profiling says otherwise.
3. **Forgetting Dask client**: A local client is created by default but explicit is better; call `client.wait_for_workers(n)` before constructing the GP.
4. **Length scales too large**: Reduces sparsity, defeating the purpose. Keep support radius reasonable.
5. **Two live GPs on one client**: Constructing a second gp2Scale GP on a Dask client that still has a live one causes scatter race conditions. Let the first go out of scope, or use a separate client.
