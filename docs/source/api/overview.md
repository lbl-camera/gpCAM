# API Reference

```{toctree}
:hidden:
:maxdepth: 1

gpOptimizer.md
fvgpOptimizer.md
gpMCMC.md
kernels.md
logging.md
```

gpCAM comes with two main classes: the [gpOptimizer](gpOptimizer.md), for single-task
Gaussian processes and Bayesian optimization, and the [fvgpOptimizer](fvgpOptimizer.md)
for the multi-task equivalent. Both are capable of dealing with non-Euclidean input
spaces and world-record-holding scaling of exact GPs.

These optimizer classes are a thin Bayesian-optimization layer on top of the
[fvGP](https://fvgp.readthedocs.io) engine — most of the GP machinery (kernels,
training, posterior evaluation) is inherited from `fvgp.GP`, and acquisition-function
optimization can run on supercomputers via [HGDL](https://hgdl.readthedocs.io).

To get to know gpCAM, check out the [examples](../examples/index.md), download the
repository and look in `./tests`, or visit the project
[website](https://gpcam.lbl.gov/).

## New in fvgp 4.8 (gpCAM 8.4)

Because the optimizers inherit from `fvgp.GP`, the following fvgp 4.8 additions are
available directly on any `GPOptimizer`/`fvGPOptimizer` instance and are listed on the
[gpOptimizer](gpOptimizer.md) / [fvgpOptimizer](fvgpOptimizer.md) pages:

- **Model-validation metrics** — `mae`, `mape`, `msll`, `interval_score`, `mpiw`, and
  `coverage_curve` for quantifying predictive accuracy and calibration, plus
  `plot_observed_vs_predicted` for a quick diagnostic plot.
- **New kernels** — `bump` and `sle_kernel` (in addition to the existing library), all
  re-exported through `gpcam.kernels` and documented on the [Kernels](kernels.md) page.
- **Linear-algebra modes** — the `linalg_mode` argument now accepts `CholInv`/`Inv`
  (the replacement for the removed `calc_inv` option) and preconditioned sparse solvers
  such as `sparseCGpre`/`sparseMINRESpre` for large gp2Scale problems.

## See Also

- [Repository](https://github.com/lbl-camera/gpCAM/)
- The [fvGP](https://fvgp.readthedocs.io/en/latest/) package
- The [HGDL](https://hgdl.readthedocs.io/en/latest/) package

```{div} centered-heading
Have suggestions for the API or found a bug?
```

```{div} text-center
Please submit an issue on [GitHub](https://github.com/lbl-camera/gpCAM/).
```
