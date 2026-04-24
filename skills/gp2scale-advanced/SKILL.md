# Skill: gp2Scale — Large-Scale GPs

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
    gp2Scale_dask_client=client,
    gp2Scale_batch_size=500,       # typical; tune up for large clusters
    init_hyperparameters=np.array([0.73, 0.0014]),  # signal var, length scale
)

gpo.train(hyperparameter_bounds=hps_bounds, max_iter=25, info=True)
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

| Mode | Description |
|------|-------------|
| `"Chol"` | Sparse Cholesky — fastest for moderate sparsity |
| `"sparseLU"` | Sparse LU decomposition |
| `"sparseCG"` | Conjugate gradient (iterative) |
| `"sparseMINRES"` | MINRES (iterative) |

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

## Estimating Compute Time

```python
from gpcam.gp_optimizer import gp2Scale_time_estimate
gp2Scale_time_estimate(n_workers=8, worker_speed=500, n_data=100000)
```

## Common Pitfalls

1. **Using a non-sparse kernel**: gp2Scale won't speed up dense kernels.
2. **Batch size too small**: More overhead. Start with 10000.
3. **Forgetting Dask client**: A local client is created by default but explicit is better.
4. **Length scales too large**: Reduces sparsity, defeating the purpose. Keep support radius reasonable.
