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
- **Large-scale (>10k points)**: `skills/gp2scale-advanced/SKILL.md`
- **Multi-task/multi-output**: `skills/multi-task-advanced/SKILL.md`
- **Constrained observations (y>0 or y∈[0,1])**: `skills/transformed-optimizers-advanced/SKILL.md`

These skills also ship as a Claude Code plugin marketplace (see README.md), so they are available outside this repo once installed.

### Key principles for generated experiment scripts

1. **Generate complete, runnable scripts** — not fragments
2. **Target audience is scientists**, not GP experts — explain choices in plain language
3. **Always document the hyperparameter layout** — which index maps to what
4. **Hyperparameter bounds must match** the total hyperparameter count across kernel + mean + noise
5. **Default kernel is usually fine** — only suggest custom when there's a clear reason
6. **Use vectorized numpy** — no Python loops over data points
7. **Return dict keys**: `posterior_mean()` returns `"m(x)"`, `posterior_covariance()` returns `"v(x)"` and `"S"`. NOT `"f(x)"`.
8. **`get_data()` keys use spaces**: `"x data"`, `"y data"`, `"hyperparameters"`, `"measurement variances"`. NOT underscores.

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

- **Python >= 3.11**; CI matrix is 3.11 and 3.12 (`.github/workflows/gpCAM-CI.yml`).
- Tests use `distributed.utils_test` fixtures (`client`, `loop`, `cluster_fixture`) and spin up a **local Dask cluster**, so they are slow and start real worker processes. Multiprocessing entrypoint guards (`if __name__ == "__main__":`) matter — see `docs/source/common-bugs.md`.
- `tox.ini` and the `[bumpversion]` block in `setup.cfg` are **stale** (reference py35–py38 and a removed `setup.py`). Don't trust them; GitHub Actions + hatch-vcs are authoritative. Do not hand-edit `gpcam/_version.py` (auto-generated).

### The big picture

**gpCAM is a thin Bayesian-optimization / autonomous-data-acquisition layer on top of the [`fvgp`](https://github.com/lbl-camera/fvgp) package** (pinned `fvgp==4.7.9` in `pyproject.toml`). The actual GP math — kernels, hyperparameter training, posterior evaluation, MCMC, deep kernels — lives in `fvgp`. gpCAM adds the `ask`/`tell`/`train`/`optimize` loop and acquisition-function optimization on top.

This means **the most important architectural fact is what is NOT in this repo**:

- `gpcam/kernels.py`, `gpcam/gp_mcmc.py`, and `gpcam/deep_kernel_network.py` are **one-line re-exports** of the corresponding `fvgp` modules.
- Methods like `train()`, `posterior_mean()`, `posterior_covariance()`, `update_gp_data()`, `gp_relative_information_entropy()`, `gp_total_correlation()`, and the default kernel/mean/noise are all **inherited from `fvgp.GP`** — to understand or change them, read the installed `fvgp` source, not this repo.

### Class hierarchy

- `GPOptimizerBase(fvgp.GP)` — `gpcam/gp_optimizer_base.py`. Adds `ask()`, `tell()`, `optimize()`, `evaluate_acquisition_function()`, `get_data()`, and pickling. This is where the optimization-loop logic lives.
- `GPOptimizer(GPOptimizerBase)` — single-task (scalar) GP. `multi_task=False`.
- `fvGPOptimizer(GPOptimizerBase, fvGP)` — multi-task (vector-valued) GP. `multi_task=True`. A multi-task GP is modeled as a single-task GP over the Cartesian product of input × output space, so the task index becomes an extra input dimension that kernel/mean/noise functions must handle.

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

- **Lazy initialization**: if `x_data`/`y_data` are not passed to the constructor, the underlying `fvgp.GP` is not built until the first `tell()`. The `self.gp` boolean tracks this; many properties (`x_data`, `y_data`, `args`, ...) return `None` and most methods assert before init.
- `optimize(...)` in `gp_optimizer_base.py` is a convenience wrapper that runs the whole tell/ask/train loop for a known `func`.

### Acquisition functions (`gpcam/surrogate_model.py`)

`ask()` delegates to `surrogate_model.find_acquisition_function_maxima()`, which **maximizes** the acquisition function over `input_set` using one of three methods:

- `"global"` → scipy `differential_evolution` (default; supports vectorized eval)
- `"local"` → scipy `minimize` (L-BFGS-B with finite-difference gradient)
- `"hgdl"` → the in-house `hgdl` hybrid optimizer (needs a Dask client; used automatically when `n>1` with a callable acq func)

Sign convention to keep straight: acquisition functions are written to be **maximized**, but internally `evaluate_acquisition_function` returns the **negated** value (so the scipy minimizers maximize), and the result is divided by `cost_function`. `ask()` flips the sign back in its return dict. A user-supplied acquisition function `f(x, gp_obj)` must return a 1d array of length `len(x)` and is maximized.

Built-in acquisition strings (`"variance"`, `"ucb"`, `"lcb"`, `"expected improvement"`, `"relative information entropy"`, `"total correlation"`, `"target probability"`, etc.) are dispatched in `evaluate_gp_acquisition_function`, with separate single-task vs multi-task (`x_out is not None`) branches. `"target probability"` requires `args={'a': lower, 'b': upper}`.

`ask()` returns `{'x': ..., 'f_a(x)': ..., 'opt_obj': ...}`. Non-Euclidean / mixed spaces are supported by passing `input_set` as a **list of candidates** instead of a bounds array.

### Deprecated code

`gpcam/autonomous_experimenter.py` — `AutonomousExperimenterGP` and `AutonomousExperimenterFvGP` both **raise on construction**. They are deprecated in favor of using `GPOptimizer`/`fvGPOptimizer` directly. Don't extend or revive them; point users at the optimizer classes (or the Tsuchinoko package).

## Reference materials

- [gpCAM documentation](https://gpcam.readthedocs.io) — full API reference and mathematical background (much of it the inherited `fvgp` API)
- `docs/source/common-bugs.md` — recurring errors and fixes (singular matrix, Dask freeze_support, pickle/allow_pickle)
- The installed `fvgp` package source — authoritative for kernels, training, and posterior math
