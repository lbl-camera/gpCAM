---
name: gp2scale-advanced
description: Use for large-scale gpCAM experiments (>10k points up to millions) using sparse compactly-supported kernels and Dask distributed computing for exact GP computation at scale.
gpcam_version: "8.7.x"
fvgp_version: "4.11.x"
last_verified: "2026-10-28 (gpCAM 8.7.0, a21385a)"
---

# Skill: gp2Scale — Large-Scale GPs

*Verified against gpCAM 8.7.0 / fvgp 4.11.1 — last checked 2026-10-28 (gpCAM a21385a).*

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
import numpy as np
from distributed import Client
from gpcam import GPOptimizer

if __name__ == "__main__":
    # Start a local Dask cluster (must be inside __main__ guard for worker spawning)
    client = Client()                  # uses all available cores
    client.wait_for_workers(4)         # good practice: wait for workers before constructing

    D = x_data.shape[1]
    init_hps = np.array([0.73] + [0.01] * D)
    hps_bounds = np.array([[0.01, 10.0]] + [[1e-4, 0.1]] * D)

    gpo = GPOptimizer(
        x_data=x_data,
        y_data=y_data,
        gp2Scale=True,
        dask_client=client,
        # gp2Scale_batch_size defaults to 10000 and gp2Scale_distribution to "blockwise".
        # Leave both alone unless profiling says otherwise — the batch size means
        # different things under the two distributions. See "Choosing the Batch Size".
        init_hyperparameters=init_hps,  # signal var + D length scales
    )

    # Under gp2Scale, train() runs MCMC: its stochastic log-determinant rules out the
    # gradient-based default. max_iter=25 keeps this example quick; for a real fit leave it at
    # the default and let the convergence check end the chain (from 400 steps on).
    gpo.train(hyperparameter_bounds=hps_bounds, max_iter=25, info=True)
```

## Choosing the Batch Size

`gp2Scale_batch_size` is a **memory budget per Dask task**, and what it sizes depends on
`gp2Scale_distribution`. **The default is 10000 with `"blockwise"`, and that is the right
starting point.**

| `gp2Scale_distribution` | one task computes | task count |
|---|---|---|
| `"blockwise"` (default) | a `batch_size × batch_size` block | `(N / batch_size)² / 2` — upper triangle only |
| `"rowwise"` | a `batch_size × N` **row strip**, in one kernel call | `N / batch_size` |

So under block-wise, dropping the batch size by 20× multiplies the task count by 400 and
the run becomes scheduler-bound rather than compute-bound. Under row-wise the count grows
only linearly, so row-wise usually wants a *smaller* batch size than block-wise — it can
otherwise leave a large cluster with fewer tasks than workers.

The two numbers are **not comparable**. At `batch_size=10000` and N=1e6, block-wise
sanctions 0.8 GB per task and row-wise 80 GB, so the same setting can leave a computation
on the client under one mode and distribute it under the other.

That budget also decides **whether to distribute at all**: anything that fits inside one
task is computed in a single kernel call on the client instead, because scheduling costs
more than the work. This is what stops a tall thin cross-covariance from being shredded —
a 10000 × 2 posterior cross-covariance takes 0.83 ms directly against 1.8 s spread over
1000 tasks. Cross-covariances (`k(x_data, x_pred)` for the posterior, `k(x_old, x_new)`
when appending) are always cut into **strips** whatever the distribution says, since
blocking a tall thin matrix multiplies tasks without reducing per-task memory.

Chunks are a **maximum with a remainder**, not an exact division: N=100000 at 15000 gives
six chunks of 15000 and one of 10000.

Whether row-wise wins depends on the kernel, not the assembly. It cannot exploit symmetry
so it doubles the kernel evaluations, but each worker returns a finished sparse strip and
the host merely concatenates. Measured at N=6000 on 2 workers: with a **dense** kernel
block-wise is ahead (0.85 s vs 1.39 s — row-wise pays 2× evaluations for nothing), but
with the **support-aware sparse** kernel row-wise wins (0.171 s vs 0.193 s), because a
whole row lets it prune in a single neighbor search. Reach for row-wise when host assembly
is the bottleneck, or to cap host peak memory at the finished matrix plus one strip.

Tune only if profiling shows a problem, and in this direction:

| Symptom | Change |
|---|---|
| Dask dashboard shows tiny tasks, scheduler pegged, workers idle | **Increase** batch size |
| Workers running out of memory on a single block | **Decrease** batch size |
| Many workers idle because there are fewer blocks than workers | **Decrease**, so `(N / batch_size)² ≳ n_workers` |

The constraint is that one task must fit comfortably in one worker's memory, while still
producing enough tasks to keep every worker busy. Block-wise that is `batch_size² × 8`
bytes for float64 — 10000 gives an 800 MB dense block. Row-wise it is `batch_size × N × 8`,
which is 8 GB at N=100000 and a 10000 strip: that allocation, not the task count, is what
bounds worker memory under row-wise. Either way this is why the compact-support kernel
producing a *sparse* block matters — with
`wendland_anisotropic_gp2Scale_cpu_sparse` the dense array is never materialized.

Estimate the covariance-assembly time before committing to a long run — time one
block on one worker, then:

```python
# time_per_worker_execution: seconds for one worker to compute one block
gpo.get_gp2Scale_exec_time(time_per_worker_execution=0.05, number_of_workers=32)
# returns D**2 * tb / (2 * n * b**2), with D = len(x_data), b = batch size
```

## Kernel Requirement

When `gp2Scale=True`, the kernel MUST produce a sparse matrix. If `kernel_function=None`, gpCAM automatically uses the default compactly supported Wendland kernel (`gp_optimizer.py:126-131, 224-225`).

`gpcam.kernels` re-exports all `fvgp.kernels` Wendland helpers (`gpcam/kernels.py:1`, `gp_optimizer.py:128-130`):
- `wendland_anisotropic_gp2Scale_cpu(x1, x2, hps)` — CPU Wendland kernel for `gp2Scale`
- `wendland_anisotropic_gp2Scale_cpu_sparse(x1, x2, hps)` — builds a `scipy.sparse` block directly without materializing the dense array
- `wendland_anisotropic_gp2Scale_gpu(x1, x2, hps)` — GPU Wendland kernel for `gp2Scale`
- `wendland_anisotropic(x1, x2, hps)` and the distance-matrix primitive `wendland_kernel(d, l)`

```python
from gpcam.kernels import wendland_anisotropic_gp2Scale_cpu_sparse

def my_gp2scale_kernel(x1, x2, hps):
    """Custom kernel with compact support returning a sparse block for gp2Scale."""
    return wendland_anisotropic_gp2Scale_cpu_sparse(x1, x2, hps)
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

During training, a preconditioner is only worth building if it is reused: rebuilt for every
likelihood evaluation, even a good one leaves CG slower than plain MINRES. So since fvgp 4.9
the preconditioned modes build one lazily. While the hyperparameters still jump (an MCMC
burn-in), training solves with unpreconditioned MINRES and builds nothing; once
`args["sparse_preconditioner_settle_steps"]` (default 3) evaluations in a row stay close, one
is built and reused for as long as K+V stays within `sparse_preconditioner_max_matrix_drift`
of it. `args["sparse_preconditioner_lazy"]=False` restores building on every evaluation.

The stochastic log-determinant stops sampling at `args["random_logdet_error_rtol"]`
(default 0.01, i.e. 1% of |log|K+V||) or `random_logdet_max_num_samples`. Before fvgp 4.9 a
negative log-determinant -- the common case -- never met the relative target and always ran
all 5000 probes; it now stops at 1%, often 10x faster. `random_logdet_error_atol` sets an
absolute target instead (with both given, the looser applies). Note that 1% of a large
log-determinant is a likelihood error of several nats -- tighten it if MCMC acceptance
decisions look noisy.

**`"Chol"`, `"CholInv"`, and `"Inv"` are the dense (non-gp2Scale) modes** — do not use
them here.

## Custom Block-MCMC Training

`train(method="mcmc")` -- the gp2Scale default -- is an adaptive Metropolis sampler: it
tunes its step size toward 23% acceptance and learns the posterior's shape from the whole
chain, samples positive hyperparameters spanning a decade in log space
(`mcmc_log_scale="auto"`), and stops once a convergence check passes
(`mcmc_break_condition="default"`; `None` runs exactly `max_iter` steps). The trained
hyperparameters are the posterior median after `mcmc_burn_in` (0.5), and `mcmc_seed` makes
the run reproducible. For full control, drive the sampler directly against the GP's
log-likelihood:

```python
import numpy as np
from gpcam import gpMCMC, ProposalDistribution

def log_likelihood(hps, args):
    return gpo.log_likelihood(hyperparameters=hps)   # exposed on GPOptimizer

pd = ProposalDistribution([0, 1], init_prop_Sigma=np.identity(2) * 0.01)

# the default prior is flat within `bounds`; pass prior_function=f(theta, bounds, args) for another
mcmc = gpMCMC(log_likelihood, bounds=hps_bounds, proposal_distributions=[pd])
result = mcmc.run_mcmc(x0=np.array([1.0, 0.01]), n_updates=2000, info=True,
                       break_condition="default", rng=0)

gpo.set_hyperparameters(result["median(x)"])
```

To check convergence properly, run several chains and compare them with
`gpcam.gp_mcmc.split_rhat` (want < 1.01) and `gpcam.gp_mcmc.ess`.

`ProposalDistribution` takes the list of hyperparameter indices in that block and an initial proposal covariance. Stack multiple `ProposalDistribution`s for block-wise updates of high-dimensional hyperparameter vectors (deep kernels, etc.).

## HPC Setup (SLURM example)

```python
from dask_jobqueue import SLURMCluster
from distributed import Client

if __name__ == "__main__":
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
