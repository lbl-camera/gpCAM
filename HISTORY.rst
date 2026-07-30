=======
History
=======

Unreleased
----------

Bug fixes
~~~~~~~~~

* ``"expected improvement"`` no longer degenerates to a scaled ``"variance"``
  acquisition (`#55 <https://github.com/lbl-camera/gpCAM/issues/55>`_). The
  improvement was clipped to zero *before* being divided by the standard
  deviation, so at every candidate whose posterior mean sat below the incumbent
  the score collapsed to the constant ``0.3989 * sigma`` and carried no
  information about the posterior mean. EI now standardizes first,
  ``imp * Phi(z) + sigma * phi(z)``, which needs no clipping. Scores above the
  incumbent are unchanged; below it, EI now exploits as intended. Anyone who ran
  a single-task EI campaign on 8.4.2 or earlier was effectively running pure
  exploration.
* Multi-task ``"expected improvement"`` is scalarized coherently. It compared a
  posterior mean summed over tasks against ``np.max(y_data)`` — a single
  ``(point, task)`` entry of the flattened product-space vector — and used the
  *sum of the per-task standard deviations* as the spread. The incumbent is now
  the best task-summed observation and the spread is ``sqrt(sum_t Var[f(x,t)])``,
  matching ``ucb``/``lcb``. This assumes independent tasks; ``"knowledge
  gradient"`` and ``"noisy expected improvement"`` remain the accurate choice for
  correlated tasks, since they take the exact cross-task covariance.
* ``"knowledge gradient"`` now handles matrix-valued observation noise. fvgp lets a
  ``noise_function`` return a full ``(N, N)`` noise covariance rather than a 1-d
  vector of variances, and the assumed-noise helper averaged the whole matrix --
  understating the noise by a factor of N for uncorrelated noise merely expressed as
  a diagonal matrix, which inflated the KG update slopes and over-valued noisy
  candidates. It now reads the diagonal, and for the multi-task task-summed objective
  a full covariance yields ``Var(sum_t eps)`` exactly, including cross-task noise
  correlation.

8.4.2 — 2026-07-24
------------------

Documentation / Claude Code skills release (no library code changes; the
installed wheel is identical to 8.4.1).

* Corrected and expanded the bundled Claude Code skill set (kernels, prior
  means, noise, acquisition, cost, multi-task, gp2Scale, transformed
  optimizers, experiment designer). Fixes include the deep-kernel recipe now
  actually loading the network weights, a consistent ``K = 1 + D`` hyperparameter
  index convention (kernel → mean → noise), the documented ``prior_mean_function``
  constructor keyword, and removal of a few APIs that never existed.
* Added two new skills: ``uncertainty-calibration`` (scoring rules, coverage
  curves, over/under-confidence diagnosis) and ``troubleshooting`` (error-message
  to cause/fix decision tree).
* Every code recipe in the skills is verified to run against fvgp 4.8.1.

8.4.0 (beta) — 2026
-------------------

This release tracks ``fvgp ~= 4.8`` and renames a few constructor kwargs.
**8.3.9 remains the stable line**; pin ``gpcam==8.3.9`` if you encounter
issues on 8.4 and please open a GitHub issue.

API migration
~~~~~~~~~~~~~

==================================== ====================================
Old (8.3.x)                          New (8.4.x)
==================================== ====================================
``gp2Scale_dask_client``             ``dask_client``
``gp2Scale_linalg_mode``             ``linalg_mode``
``calc_inv=True``                    ``linalg_mode="CholInv"``
``tell(..., gp_rank_n_update=...)``  ``tell(..., rank_n_update=...)``
==================================== ====================================

New
~~~

* ``LogGPOptimizer`` for strictly positive observations (lognormal closed-form moments).
* ``LogitGPOptimizer`` for bounded observations, with a ``range=(lower, upper)`` argument for any closed interval.
* ``evaluate_posterior(x, return_samples=True, n_samples=N)`` on every optimizer — original-space posterior summary plus optional raw samples.

6.0.0 (2020-10-26)
------------------

* First release on PyPI.
