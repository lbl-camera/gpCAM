=======
History
=======

Unreleased
----------

Removals
~~~~~~~~

* ``AutonomousExperimenterGP`` and ``AutonomousExperimenterFvGP`` are gone, along with
  ``gpcam/autonomous_experimenter.py`` and the two notebooks in ``obsolete/`` that used
  them. Both classes had raised on construction for several releases, so nothing that
  worked stops working -- but the *names* are no longer exported, so
  ``from gpcam import AutonomousExperimenterGP`` now raises ``ImportError`` rather than
  importing successfully and failing later with a deprecation message.

  The replacement is to use ``GPOptimizer``/``fvGPOptimizer`` directly in an ask/tell/train
  loop, which is what the class wrapped and what every example already shows; the
  Tsuchinoko package is the alternative for a full experiment-orchestration layer.

Dependencies
~~~~~~~~~~~~

* Require ``fvgp ~= 4.8.6``, which is the release that carries ``gp2Scale_distribution``
  and Python 3.10 support (was ``~= 4.8.5``). ``~=`` already admitted any later 4.8.z, so
  this only raises the floor -- but it has to be raised, because passing the new keyword
  to an older fvGP is a ``TypeError`` rather than a graceful no-op.

* Require ``fvgp ~= 4.8.5``, the newest release in the 4.8 line (was ``~= 4.8.1``).
  This raises the floor only; ``~=`` already allowed any later 4.8.z. 4.8.5 carries the
  corrected sparse-solver guidance: preconditioner reuse is decided by how far K+V has
  drifted rather than by a refresh interval, and Krylov warm starts are honored only for
  ``train(method='mcmc')``.

Documentation
~~~~~~~~~~~~~

* The ``kernel_function`` docstrings now state what the default kernel actually is -- a
  stationary anisotropic Matern kernel of first-order differentiability with one length
  scale per input dimension -- and that ``gp2Scale`` switches the default to a compactly
  supported Wendland kernel chosen by ``compute_device``, with pointers into
  ``fvgp.kernels``. They previously referenced ``fvgp.GP.default_kernel``, which does not
  exist, so the cross-reference never resolved. Kept in step with the same change in fvGP.

New features
~~~~~~~~~~~~

* ``gp2Scale_distribution`` is forwarded to fvGP by ``GPOptimizer``, ``fvGPOptimizer``
  and ``GPOptimizerBase``. It chooses how the distributed covariance is cut across the
  workers: ``"blockwise"`` (the default, and the historical behavior) maps (row block,
  column block) pairs and schedules only the upper triangle of a symmetric covariance, so
  the cluster does half the kernel evaluations; ``"rowwise"`` maps whole row strips and
  has each worker return a finished sparse strip, moving the assembly sort onto the
  workers and reducing the host's job to a concatenation. Row-wise cannot exploit symmetry
  and so doubles the kernel evaluations; it is the choice when host assembly rather than
  kernel evaluation is the bottleneck. The value is validated by fvGP and survives
  pickling; state pickled before this parameter existed unpickles as ``"blockwise"``.

* ``ask()`` selects a *jointly* informative batch from a candidate list. Previously,
  ``n > 1`` with a candidate list scored every candidate on its own, sorted, and returned
  the top ``n`` — which says nothing about whether those points are useful together, and
  in practice returned near-duplicates, since nothing stops the highest individual scorers
  from sitting on top of one another.

  ``"total correlation"`` and ``"relative information entropy"`` score a whole set with a
  single number, so for those the batch is now chosen jointly and ``'f_a(x)'`` is one
  value describing the set rather than one value per point. The best subset is
  combinatorial, so it is built by greedy forward selection — the best single candidate,
  then the one that best complements it, and so on — which costs ``n`` passes over the
  candidates instead of enumerating subsets. Criteria of this kind are approximately
  submodular, for which greedy selection is a standard and well-founded choice.

  Every other acquisition, including any callable (which gpCAM cannot introspect), still
  takes the point-by-point path, and ``ask()`` now warns that the returned points are not
  mutually optimal.

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

8.4.0 — 2026
------------

This release tracks ``fvgp ~= 4.8`` and renames a few constructor kwargs. It went out
as a beta; the 8.4 line is now the recommended one and 8.3.9 is no longer maintained
as a fallback.

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
