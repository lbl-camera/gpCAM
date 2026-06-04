# Transformed Optimizers

Single-task optimizers that fit a Gaussian process to observations after a fixed
output transform (link function), and use
:py:meth:`gpcam.GPOptimizer.evaluate_posterior` to push the latent Gaussian
posterior back through the inverse link to the original observation space.

Use :py:class:`gpcam.LogGPOptimizer` for strictly positive observations in
``(0, inf)`` — predictions and credible intervals are guaranteed positive, and
the original-scale mean/std are available in closed form (lognormal).

Use :py:class:`gpcam.LogitGPOptimizer` for observations bounded in ``[0, 1]`` —
predictions and credible intervals are guaranteed inside ``(0, 1)``; the
original-scale mean/std are estimated by Monte Carlo (logistic-normal has no
closed form). Observations are clipped to ``[eps, 1 - eps]`` because
``logit(0)``/``logit(1)`` are infinite.

The inherited :py:meth:`posterior_mean` / :py:meth:`posterior_covariance`
operate in the GP's modeling space (log / logit); use
:py:meth:`evaluate_posterior` to obtain a posterior on the original scale.

## LogGPOptimizer

```{eval-rst}
.. autoclass:: gpcam.gp_optimizer.LogGPOptimizer
    :members:
```

## LogitGPOptimizer

```{eval-rst}
.. autoclass:: gpcam.gp_optimizer.LogitGPOptimizer
    :members:
```
