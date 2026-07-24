# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file serves two audiences:
1. **Scientists designing experiments** with gpCAM — routed through the skills below.
2. **Developers working on the gpCAM package itself** — see Development & Architecture.

---

## For scientists: designing autonomous experiments

You are helping scientists design autonomous experiments using gpCAM, a Gaussian Process-based Bayesian optimization toolkit. Read the appropriate skill file before generating code:

- **Designing an experiment**: `skills/experiment-designer/SKILL.md`
- **Custom kernels**: `skills/kernel-designer/SKILL.md`
- **Acquisition functions**: `skills/acquisition-functions/SKILL.md`
- **Prior mean functions**: `skills/prior-mean-functions/SKILL.md`
- **Noise models**: `skills/noise-functions/SKILL.md`
- **Cost functions (travel time)**: `skills/cost-functions/SKILL.md`
- **Validating uncertainty / calibration**: `skills/uncertainty-calibration/SKILL.md`
- **Debugging errors and silent misbehavior**: `skills/troubleshooting/SKILL.md`
- **Large-scale (>10k points)**: `skills/gp2scale-advanced/SKILL.md`
- **Multi-task/multi-output**: `skills/multi-task-advanced/SKILL.md`
- **Constrained observations (y>0 or y∈[0,1])**: `skills/transformed-optimizers-advanced/SKILL.md`

Each `SKILL.md` carries `gpcam_version` / `fvgp_version` / `last_verified` frontmatter.
When you change an API these skills document, update the stamp along with the text.

These skills also ship as a Claude Code plugin marketplace (`.claude-plugin/marketplace.json`, `.claude-plugin/plugin.json`), so they are available outside this repo once installed — see README.md.

### Key principles for generated experiment scripts

1. **Generate complete, runnable scripts** — not fragments
2. **Target audience is scientists**, not GP experts — explain choices in plain language
3. **Always document the hyperparameter layout** — which index maps to what
4. **Hyperparameter bounds must match** the total hyperparameter count across kernel + mean + noise
5. **Default kernel is usually fine** — only suggest custom when there's a clear reason
6. **Use vectorized numpy** — no Python loops over data points
7. **Return dict keys**: `posterior_mean()` returns `"m(x)"`, `posterior_covariance()` returns `"v(x)"` and `"S"`. NOT `"f(x)"`.
8. **`get_data()` keys use spaces**: `"x data"`, `"y data"`, `"hyperparameters"`, `"measurement variances"`. NOT underscores.
9. **Hyperparameter index layout is kernel → mean → noise**, and the ranges must be disjoint. Derive the start index as `K = 1 + D` for the default kernel; never hardcode `K = 3` (correct only for 2-D input) and never read `hps[-1]` in a prior mean (that is the noise function's slot).
10. **The prior mean keyword is `prior_mean_function=`**, and the default prior mean is a constant equal to `mean(y_data)` — not zero.
11. **`args` reaches kernel/mean/noise only** (dispatched by parameter count). Cost functions are called as `cost_function(origin, x)` and acquisition functions as `f(x, gp_obj)` — neither receives `args`; use `functools.partial`.
12. **`cost_function` is applied only when `ask(..., position=...)` is given.** Without `position`, gpCAM silently ignores it.

---

## Development & Architecture

### Commands

Running from source requires an install (the package reads a VCS-generated `gpcam/_version.py`):

```bash
pip install -e .[tests]          # editable install with test deps (pytest, torch, imate, ...)
pytest tests                      # run the full test suite
pytest tests --cov=./ --cov-report=xml   # exactly what CI runs
pytest tests/test_gpCAM.py::test_basic_1task   # run a single test
hatch build                       # build sdist + wheel (version comes from git tags via hatch-vcs)
```

- **Python >= 3.11**; CI matrix is 3.11 and 3.12 (`.github/workflows/gpCAM-CI.yml`). A second workflow refreshes Context7 docs on version tags.
- The whole test suite is `tests/test_gpCAM.py` (~6 tests). Most take the `client` fixture from `distributed.utils_test` and spin up a **local Dask cluster**, so they are slow and start real worker processes. Multiprocessing entrypoint guards (`if __name__ == "__main__":`) matter in user-facing example scripts.
- `tox.ini` and the `[bumpversion]` block in `setup.cfg` are **stale** (reference py35–py38 and a removed `setup.py`). Don't trust them; GitHub Actions + hatch-vcs are authoritative. Do not hand-edit `gpcam/_version.py` (auto-generated).
- Stray `.*.un~` files at the repo root and in `gpcam/` are editor undo files, not source. Ignore them.

### Version state

The repo is on the **8.4.x beta line** (`fvgp ~= 4.8.1`); **8.3.9 is the stable line** users may still be pinned to. HISTORY.rst and the README warning are the migration reference. Constructor/method kwargs renamed in 8.4:

| Old (8.3.x) | New (8.4.x) |
|---|---|
| `gp2Scale_dask_client` | `dask_client` |
| `gp2Scale_linalg_mode` | `linalg_mode` |
| `calc_inv=True` | `linalg_mode="CholInv"` |
| `tell(..., gp_rank_n_update=...)` | `tell(..., rank_n_update=...)` |

When answering user questions or writing examples, use the new names — but recognize the old ones as 8.3-era code, not typos.

### The big picture

**gpCAM is a thin Bayesian-optimization / autonomous-data-acquisition layer on top of the [`fvgp`](https://github.com/lbl-camera/fvgp) package** (`fvgp ~= 4.8.1` in `pyproject.toml`). The actual GP math — kernels, hyperparameter training, posterior evaluation, MCMC, deep kernels — lives in `fvgp`. gpCAM adds the `ask`/`tell`/`train`/`optimize` loop and acquisition-function optimization on top.

This means **the most important architectural fact is what is NOT in this repo**:

- `gpcam/kernels.py`, `gpcam/gp_mcmc.py`, and `gpcam/deep_kernel_network.py` are **one-line `from fvgp.X import *` re-exports**.
- Methods like `train()`, `posterior_mean()`, `posterior_covariance()`, `update_gp_data()`, `gp_relative_information_entropy()`, `gp_total_correlation()`, and the default kernel/mean/noise are all **inherited from `fvgp.GP`** — to understand or change them, read the installed `fvgp` source, not this repo.

Effectively the entire package is four files: `gp_optimizer_base.py` (loop logic + transform hooks), `gp_optimizer.py` (the four public classes), `surrogate_model.py` (acquisition evaluation + optimization), and `autonomous_experimenter.py` (deprecated).

### Class hierarchy

- `GPOptimizerBase(fvgp.GP)` — `gpcam/gp_optimizer_base.py`. Adds `ask()`, `tell()`, `optimize()`, `evaluate_acquisition_function()`, `evaluate_posterior()`, `get_data()`, the output-transform hooks, and pickling. This is where the optimization-loop logic lives.
- `GPOptimizer(GPOptimizerBase)` — single-task (scalar) GP. `multi_task=False`.
- `fvGPOptimizer(GPOptimizerBase, fvGP)` — multi-task (vector-valued) GP. `multi_task=True`. A multi-task GP is modeled as a single-task GP over the Cartesian product of input × output space, so the task index becomes an extra input dimension that kernel/mean/noise functions must handle.
- `LogGPOptimizer(GPOptimizer)` / `LogitGPOptimizer(GPOptimizer)` — transformed optimizers (below).

So a `GPOptimizer` instance *is* an `fvgp.GP` with the optimization methods mixed in.

### ask / tell / train loop

The canonical usage (no `AutonomousExperimenter` — see below):

```python
gp = GPOptimizer(x_data, y_data)
gp.train()                                   # inherited from fvgp.GP
for i in range(N):
    new = gp.ask(bounds, acquisition_function="variance")["x"]
    gp.tell(new, measure(new))               # appends + rank-n updates the GP
    if i in train_at: gp.train()
```

- **Lazy initialization**: if `x_data`/`y_data` are not passed to the constructor, the underlying `fvgp.GP` is not built until the first `tell()`. The `self.gp` boolean tracks this; many properties (`x_data`, `y_data`, `args`, ...) return `None` and most methods `assert self.gp` before doing anything.
- `optimize(...)` in `gp_optimizer_base.py` is a convenience wrapper that runs the whole tell/ask/train loop for a known `func`; it returns a trace dict (`'trace f(x)'`, ...).
- `get_data()` returns **different keys per mode**: single-task has `"original y data"` (the inverse-transformed observations); multi-task has `"transformed x data"` / `"transformed y data"` (the Cartesian-product representation the GP actually sees). Both share `"input dim"`, `"x data"`, `"y data"`, `"measurement variances"`, `"hyperparameters"`, `"cost function"`.

### Output transforms and the transformed optimizers

`GPOptimizerBase` defines a small hook protocol so a subclass can model observations in a transformed space while presenting results on the original scale. The hooks are **identity by default**, so plain `GPOptimizer` pays nothing:

`_prepare(y)` (validate/condition) → `_forward(y)` (into GP space) → `_inverse(z)` (back out) → `_forward_deriv(y)` (delta-method noise propagation, used by `_transform_noise_variances`) → `_moments(mu, var)` (original-space mean/std) → `_samples(mu, var, n)` (default: sample the GP-space Gaussian, push through `_inverse`).

- `LogGPOptimizer` — strictly positive `y`; GP sees `log(y)`; lognormal closed-form moments.
- `LogitGPOptimizer` — bounded `y`; takes `range=(lower, upper)` and `eps`; logistic-normal, moments by sampling (`n_samples`).

The critical consequence: **inherited `posterior_mean()` / `posterior_covariance()` operate in GP (transformed) space.** Original-scale answers come from `evaluate_posterior(x, x_out=None, level=0.95, return_samples=False, n_samples=10000)`, which returns `{"median", "mean", "std", "lower", "upper", "level"}` plus optional `"samples"` of shape `(n_points, n_samples)`. Ranking acquisitions (`variance`, `ucb`, `lcb`, `maximum`, `minimum`) are unaffected because the transforms are monotone, but `"target probability"` bounds must be given in **transformed space**.

### Acquisition functions (`gpcam/surrogate_model.py`)

`ask()` delegates to `surrogate_model.find_acquisition_function_maxima()`, which **maximizes** the acquisition function over `input_set` using one of three methods:

- `"global"` → scipy `differential_evolution` (default; supports vectorized eval)
- `"local"` → scipy `minimize` (L-BFGS-B with finite-difference gradient)
- `"hgdl"` → the in-house `hgdl` hybrid optimizer (needs a Dask client; used automatically when `n>1` with a callable acq func)

Sign convention to keep straight: acquisition functions are written to be **maximized**, but internally `evaluate_acquisition_function` returns the **negated** value (so the scipy minimizers maximize), and the result is divided by `cost_function`. `ask()` flips the sign back in its return dict. A user-supplied acquisition function `f(x, gp_obj)` must return a 1d array of length `len(x)` and is maximized.

Built-in acquisition strings are dispatched in `evaluate_gp_acquisition_function`, with **separate single-task vs multi-task branches** (`x_out is not None`) that do *not* support the same set — e.g. `"maximum"`, `"minimum"`, `"gradient"`, `"probability of improvement"`, and `"target probability"` exist only on the single-task branch. `"target probability"` requires `args={'a': lower, 'b': upper}`.

`ask()` returns `{'x': ..., 'f_a(x)': ..., 'opt_obj': ...}`. Non-Euclidean / mixed spaces are supported by passing `input_set` as a **list of candidates** instead of a bounds array.

### Pickling

`GPOptimizerBase.__getstate__`/`__setstate__` are hand-written (Dask clients and some fvgp internals don't pickle). `LogitGPOptimizer` overrides `__getstate__` again for its transform state. Callables handed to the optimizer (cost/acquisition functions) must be **module-level** to pickle by reference — `tests/test_gpCAM.py::test_pickle` depends on this.

### Deprecated code

`gpcam/autonomous_experimenter.py` — `AutonomousExperimenterGP` and `AutonomousExperimenterFvGP` are still exported from `gpcam/__init__.py` but both **raise on construction**. They are deprecated in favor of using `GPOptimizer`/`fvGPOptimizer` directly. Don't extend or revive them; point users at the optimizer classes (or the Tsuchinoko package).

## Reference materials

- [gpCAM documentation](https://gpcam.readthedocs.io) — full API reference and mathematical background (much of it the inherited `fvgp` API); sources in `docs/source/`, built with sphinx + myst-nb (`pip install -e .[docs]`)
- `examples/` — runnable notebooks covering the main paths: minimal loop, single/multi-task, gp2Scale, non-Euclidean input spaces, Log/Logit optimizers, plus `.py` templates for custom acquisition/cost/kernel functions
- The installed `fvgp` package source — authoritative for kernels, training, and posterior math
