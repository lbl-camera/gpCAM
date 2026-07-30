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

These skills also ship as a Claude Code plugin marketplace (`.claude-plugin/marketplace.json`, `.claude-plugin/plugin.json`), so they are available outside this repo once installed — see README.md. The manifests do **not** enumerate skills (they are auto-discovered from `skills/`), so adding or renaming a skill means touching four other places instead: the routing list above, the skills table in README.md, the table in `docs/source/claude-skills.md`, and a HISTORY.rst entry.

### Key principles for generated experiment scripts

1. **Generate complete, runnable scripts** — not fragments
2. **Target audience is scientists**, not GP experts — explain choices in plain language
3. **Never pick the acquisition function silently, and never default to `"expected improvement"`.** It's the choice that decides where the instrument goes. Name your recommendation, state its one tradeoff, and confirm with the user. "Find the best conditions" is ambiguous — ask whether they want the single best point or a trustworthy map containing it. Several built-ins are maximization-only and fail silently for minimization. Read `skills/acquisition-functions/SKILL.md` first.
4. **Always document the hyperparameter layout** — which index maps to what
5. **Hyperparameter bounds must match** the total hyperparameter count across kernel + mean + noise
6. **Default kernel is usually fine** — only suggest custom when there's a clear reason
7. **Use vectorized numpy** — no Python loops over data points
8. **Return dict keys**: `posterior_mean()` returns `"m(x)"`, `posterior_covariance()` returns `"v(x)"` and `"S"`. NOT `"f(x)"`.
9. **`get_data()` keys use spaces**: `"x data"`, `"y data"`, `"hyperparameters"`, `"measurement variances"`. NOT underscores.
10. **Hyperparameter index layout is kernel → mean → noise**, and the ranges must be disjoint. Derive the start index as `K = 1 + D` for the default kernel; never hardcode `K = 3` (correct only for 2-D input) and never read `hps[-1]` in a prior mean (that is the noise function's slot).
11. **The prior mean keyword is `prior_mean_function=`**, and the default prior mean is a constant equal to `mean(y_data)` — not zero.
12. **`args` is never a call argument.** It is passed positionally only to kernel/mean/noise (dispatched by parameter count). Cost functions are called as `cost_function(origin, x)` and acquisition functions as `f(x, gp_obj)` — neither receives `args`, but a custom acquisition function can read `gp_obj.args` (that is how the built-in `"target probability"` gets its `a`/`b`, and how KG/NEI read their tuning keys). For anything else, use `functools.partial`.
13. **`cost_function` is applied only when `ask(..., position=...)` is given.** Without `position`, gpCAM silently ignores it.
14. **`ask(n>1)` silently rewrites your request** — see Acquisition functions below. For `n` genuinely independent suggestions from a named acquisition function, ask one at a time or pass a candidate list.

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
- The whole test suite is `tests/test_gpCAM.py` — 10 tests. `test_basic_1task`, `test_basic_multi_task`, and `test_acq_funcs` take the `client` fixture from `distributed.utils_test` and spin up a **local Dask cluster**, so they are slow and start real worker processes; the rest (`test_optimizers`, `test_pickle`, `test_transformed_gp`, and the acquisition regression tests) are fast and client-free — prefer those for quick iteration. Multiprocessing entrypoint guards (`if __name__ == "__main__":`) matter in user-facing example scripts.
- `tox.ini` and the `[bumpversion]` block in `setup.cfg` are **stale** (reference py35–py38 and a removed `setup.py`). Don't trust them; GitHub Actions + hatch-vcs are authoritative. Do not hand-edit `gpcam/_version.py` — it is generated by hatch-vcs at install time, gitignored, and the copy in a working tree is often behind the latest tag.
- Stray `.*.un~` files at the repo root and in `gpcam/` are editor undo files, not source. Ignore them. `obsolete/` holds superseded notebooks and an old `gp_optimizerBCK.py`; it is not packaged or importable — don't cite it as current API.

### Version state

The repo is on the **8.4.x beta line** (latest tag `8.4.2`, `fvgp ~= 4.8.4`); **8.3.9 is the stable line** users may still be pinned to. HISTORY.rst and the README warning are the migration reference. 8.4.1 → 8.4.2 was a docs/skills-only release (identical wheel), so a HISTORY entry does not imply library changes. Constructor/method kwargs renamed in 8.4:

| Old (8.3.x) | New (8.4.x) |
|---|---|
| `gp2Scale_dask_client` | `dask_client` |
| `gp2Scale_linalg_mode` | `linalg_mode` |
| `calc_inv=True` | `linalg_mode="CholInv"` |
| `tell(..., gp_rank_n_update=...)` | `tell(..., rank_n_update=...)` |

When answering user questions or writing examples, use the new names — but recognize the old ones as 8.3-era code, not typos.

### The big picture

**gpCAM is a thin Bayesian-optimization / autonomous-data-acquisition layer on top of the [`fvgp`](https://github.com/lbl-camera/fvgp) package** (`fvgp ~= 4.8.4` in `pyproject.toml` — the newest 4.8.x; `~=` allows any later 4.8.z but not 4.9). The actual GP math — kernels, hyperparameter training, posterior evaluation, MCMC, deep kernels — lives in `fvgp`. gpCAM adds the `ask`/`tell`/`train`/`optimize` loop and acquisition-function optimization on top.

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
- `optimize(...)` in `gp_optimizer_base.py` is a keyword-only convenience wrapper that runs the whole tell/ask/train loop for a known `func`; it returns `{'trace f(x)', 'trace x', 'f(x)', 'x'}`. Note `func` must return a **`(y, noise_variances)` tuple**, not a bare `y` — the loop unpacks `y_new, v_new = func(x_new)` — and it seeds itself with 10 random points when `x0` is None, discarding existing data (`tell(..., append=False)`).
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
- `"local"` → scipy `minimize` (L-BFGS-B with finite-difference gradient). Returns exactly **one** point regardless of `n`, and on `success is False` it silently substitutes the random start point `x0` (logged as a warning only).
- `"hgdl"` → the in-house `hgdl` hybrid optimizer (needs a Dask client; used automatically when `n>1` with a callable acq func). Near-duplicate optima are filtered, so it can return **fewer than `n`** points.

Sign convention to keep straight: acquisition functions are written to be **maximized**, but internally `evaluate_acquisition_function` returns the value **negated and divided by `cost_function`** (so the scipy minimizers maximize). `find_acquisition_function_maxima` flips the sign back in its `return opti, -func_eval, opt_obj` — not `ask()`, which passes `f_a(x)` through untouched. A user-supplied acquisition function `f(x, gp_obj)` must return a 1d array of length `len(x)` and is maximized.

Built-in acquisition strings are dispatched in `evaluate_gp_acquisition_function`, with **separate single-task vs multi-task branches** (`x_out is not None`) that do *not* support the same set — e.g. `"maximum"`, `"minimum"`, `"gradient"`, `"probability of improvement"`, and `"target probability"` exist only on the single-task branch. `"target probability"` requires `args={'a': lower, 'b': upper}`. The same string can also mean different things across branches: single-task `"variance"` returns the posterior **standard deviation**, multi-task `"variance"` returns the **variance summed over the output dimension**. Ranking is unaffected, but `'f_a(x)'` values are not comparable between modes.

`"knowledge gradient"` (KGCP) and `"noisy expected improvement"` (Monte-Carlo NEI) are lookahead optimization acquisitions available on **both** branches; the multi-task form scalarizes to the task-summed objective `sum_t f(x,t)`, like the multi-task EI/UCB. Both live as standalone functions in `surrogate_model.py` (`knowledge_gradient`, `noisy_expected_improvement`) with helpers (`_expected_max_of_affine` — the exact correlated-KG line integral; `_scalarized_blocks` — reference/candidate posterior blocks from `"S_flat"`, task-major `k = point + Npts*task` ordering). They need the cross-covariance between candidates and reference points, so unlike the point-wise acquisitions they call the full (non-`variance_only`) `posterior_covariance`. Tunable via `args` keys `kg_reference_set_size`/`kg_seed` and `nei_samples`/`nei_reference_set_size`/`nei_seed`.

The **task-major product-space ordering** (`k = point + Npts*task`) is the thing to get right whenever multi-task code touches raw arrays. `_scalarized_blocks` and `_assumed_observation_noise` both reshape `(P*T, P*T)` matrices as `(T, P, T, P)` because of it, and `_observed_task_sums` reads the original `(N, No)` array via `fvgp_y_data` rather than reshaping the flat `y_data` — a `reshape(-1, No)` of the product-space vector interleaves points and tasks and is silently wrong.

**`ask()` rewrites its own arguments in several cases** (all with a `warnings.warn`, easy to miss in a loop). With a Euclidean bounds array and `n > 1`:

- callable acquisition function → `method` forced to `"hgdl"`, and a `Client()` is constructed and closed inside the call if none was passed;
- string acquisition function and `method != "hgdl"` → `method` forced to `"global"`, the bounds are **tiled `n` times** into one `n*D`-dimensional joint optimization, and the acquisition function is **replaced by `"total correlation"`** unless it was already `"total correlation"` or `"relative information entropy"`.

So `ask(bounds, n=5, acquisition_function="ucb")` does not maximize UCB at all. `"total correlation"` and `"relative information entropy"` also force `vectorized=False`, and `ask(args=...)` **persistently overwrites** `self._args` for all later calls.

`ask()` returns `{'x': ..., 'f_a(x)': ..., 'opt_obj': ...}` (`opt_obj` is non-`None` only for `hgdl`). Non-Euclidean / mixed spaces are supported by passing `input_set` as a **list of candidates** instead of a bounds array — but that path bypasses the optimizers entirely: it evaluates every candidate and sorts, so `method`, `x0`, `pop_size`, `max_iter`, and `constraints` are all ignored, and `n` just truncates the sorted list.

### Pickling

`GPOptimizerBase.__getstate__`/`__setstate__` are hand-written (Dask clients and some fvgp internals don't pickle). `LogitGPOptimizer` overrides `__getstate__` again for its transform state. Callables handed to the optimizer (cost/acquisition functions) must be **module-level** to pickle by reference — `tests/test_gpCAM.py::test_pickle` depends on this.

### Deprecated code

`gpcam/autonomous_experimenter.py` — `AutonomousExperimenterGP` and `AutonomousExperimenterFvGP` are still exported from `gpcam/__init__.py` but both **raise on construction**. They are deprecated in favor of using `GPOptimizer`/`fvGPOptimizer` directly. Don't extend or revive them; point users at the optimizer classes (or the Tsuchinoko package).

## Reference materials

- [gpCAM documentation](https://gpcam.readthedocs.io) — full API reference and mathematical background (much of it the inherited `fvgp` API); sources in `docs/source/`, built with sphinx + myst-nb (`pip install -e .[docs]`)
- `examples/` — runnable notebooks covering the main paths: minimal loop, single/multi-task, gp2Scale, non-Euclidean input spaces, Log/Logit optimizers, plus `.py` templates for custom acquisition/cost/kernel functions
- The installed `fvgp` package source — authoritative for kernels, training, and posterior math
