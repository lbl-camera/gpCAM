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
