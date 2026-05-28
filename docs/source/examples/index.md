# Examples

```{toctree}
:hidden:
:maxdepth: 1

GPOptimizer_Minimal
1dSingleTaskAcqFuncTest
GPOptimizer_SingleTaskTest
GPOptimizer_NonEuclideanInputSpaces
GPOptimizer_gp2ScaleTest
GPOptimizer_MultiTaskTest
GPOptimizer_Optimization
```

The notebooks below walk through common gpCAM use cases.
Each can be downloaded and run locally after `pip install gpcam`.

- **[Minimal GPOptimizer](GPOptimizer_Minimal.ipynb)** — the shortest possible `ask`/`tell`/`train` loop
- **[Acquisition functions (1D)](1dSingleTaskAcqFuncTest.ipynb)** — comparing built-in acquisition functions on a 1D problem
- **[Single-task GP](GPOptimizer_SingleTaskTest.ipynb)** — single-task regression and autonomous data acquisition
- **[Non-Euclidean inputs](GPOptimizer_NonEuclideanInputSpaces.ipynb)** — Bayesian optimization over candidate sets (strings, graphs, materials)
- **[gp2Scale](GPOptimizer_gp2ScaleTest.ipynb)** — large-scale sparse GP with Dask distributed computation
- **[Multi-task GP](GPOptimizer_MultiTaskTest.ipynb)** — multi-output regression with the `fvGPOptimizer` class
- **[Function optimization](GPOptimizer_Optimization.ipynb)** — minimizing a known function with the `optimize` loop
