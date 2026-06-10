=======
History
=======

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
