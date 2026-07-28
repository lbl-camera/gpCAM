=======
History
=======

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
