=======
History
=======

Unreleased
----------

* The Claude Code skills moved from ``skills/`` to ``gpcam/skills/``, inside the package,
  so wheels and ``pip install`` now include them (``importlib.resources.files("gpcam.skills")``).
  ``.claude-plugin/plugin.json`` points the marketplace plugin at the new folder.
* New ``agent_skills`` entry point (``gpcam = "gpcam.skills"``) lets agent hosts discover the
  skills: each subfolder of the entry point's package that contains a ``SKILL.md`` is a skill.

8.7.0 — 2026-10-28
------------------

Requires fvgp 4.11.1. fvgp 4.11 merged its multi-task class into ``fvgp.GP``, and gpCAM follows:
``GPOptimizer`` now does single- and multi-task GPs. 4.11.1 hands a callable training ``method``
the GP, as documented (4.11.0 passed fvgp's internal trainer).

Deprecations
~~~~~~~~~~~~

* **``fvGPOptimizer`` is deprecated** (``FutureWarning``) and will be removed when gpCAM moves
  to fvgp 5.0.0. It is now a ``GPOptimizer`` with ``multi_task_mode="cartesian"``:
  ``fvGPOptimizer(x, y, ...)`` is ``GPOptimizer(x, y, ..., multi_task_mode="cartesian")``,
  and existing code keeps running.

Changes
~~~~~~~

* **One class for single- and multi-task GPs.** The shape of ``y_data`` decides: (V) or
  (V, 1) is a single-task GP, (V, No) with No > 1 a multi-task GP, modeled as the new
  ``GPOptimizer`` argument ``multi_task_mode`` says: ``'independent'`` (the default, as in
  fvgp: No GPs sharing one kernel and its hyperparameters, which is what ``GPOptimizer`` did
  with 2-d ``y_data``), ``'cartesian'`` (a single-task GP over input × output space, any
  kernel, missing outputs allowed: the former ``fvGPOptimizer``), or ``'kronecker'`` (a
  separable kernel on a full grid; by default a learned output covariance). Kernels, means and
  noise functions written for ``[x, task]`` points need ``'cartesian'``. New read-only
  properties ``multi_task_mode`` (None for a single-task GP) and, from fvgp, ``output_num``;
  ``multi_task`` is now read-only.

* **What changes for former ``fvGPOptimizer`` code:**

  - a ``y_data`` with a single column is a single-task GP (it was a one-task Cartesian GP);
  - ``get_data()`` returns the data as the GP stores it, under the same keys in every mode:
    for ``'cartesian'``, ``"x data"`` and ``"y data"`` are the (V * No, D + 1) and (V * No, 1)
    product-space arrays (they were the (V, D) and (V, No) arrays passed in), and the
    ``"transformed x data"`` / ``"transformed y data"`` keys are gone. New key:
    ``"multi-task mode"``. ``"original y data"`` is now there in every mode;
  - ``fvgp_x_data``, ``fvgp_y_data`` and ``fvgp_noise_variances`` no longer exist (fvgp
    4.11 keeps one copy of the data);
  - ``isinstance(gpo, GPOptimizer)`` is True.

* **``x_out``** selects outputs by index for ``'independent'`` and ``'kronecker'`` and by
  output coordinate for ``'cartesian'``; all outputs by default, in every mode. Passing an
  ``x_out`` to a single-task GP raises (fvgp 4.11).

* **Posterior results follow fvgp 4.11's contract**, the same keys and shapes in every mode
  (``fvgp.GP`` Notes): single-task results have no output axis; multi-task ``m(x)``, ``v(x)``
  are (V, n), ``S`` is (V, V, n, n), flat arrays are task-major; ``variance_only=True`` gives
  ``S = None``; the ``x_pred`` key is gone. Cartesian ``m(x)``, ``v(x)`` and ``S`` keep their
  shapes and values.
  ``evaluate_posterior`` returns (V, n) arrays for a multi-task GP and ``"samples"`` of shape
  (V, n, n_samples) (were flattened point-major).

* **The built-in multi-task acquisition functions work in every mode,** checked by the
  identity that independent outputs are a Cartesian GP whose kernel is zero between outputs.
  ``total correlation`` is not available for ``'kronecker'``; there ``ask(n > 1)`` on bounds
  switches to ``relative information entropy`` instead. ``knowledge gradient`` and
  ``noisy expected improvement`` read the structured ``m(x)`` and ``S``, so they no longer
  depend on the flat arrays' order.

* **Pickles.** An optimizer pickled with gpCAM < 8.7 is not reliably restored: fvgp 4.11
  restores multi-task GPs pickled before it as single-task GPs. Rebuild it from its data.
  An uninitialized one (pickled before ``tell()``) keeps its multi-task class.

Bug fixes
~~~~~~~~~

* ``GPOptimizer`` with 2-d ``y_data`` left ``x_out`` unset, so the single-task acquisition
  functions received (V, n) arrays: ``variance`` and ``ucb`` returned (V, n) instead of one
  score per point, ``noisy expected improvement`` returned the wrong number of values, and
  ``expected improvement`` raised.
* The multi-task ``expected improvement`` incumbent summed the observations, which is NaN
  wherever an output is missing (``'cartesian'``) and made every EI value NaN. It is now the
  best task-summed posterior mean at the data points, in every mode.
* ``knowledge gradient``'s noise estimate read a (V, V) noise matrix of a non-Cartesian GP as
  (task, point) blocks whenever V was divisible by the number of outputs.
* ``optimize()`` with a vector-valued ``func`` on an optimizer without data decided the shape
  from ``multi_task``, which is unknown before the first ``tell()``.

Changes inherited from fvgp 4.10 and 4.11
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

gpCAM overrides none of these methods; the constructor docstrings are fvgp's.

* ``linalg_mode="low_rank_<r>"`` for kernels of known rank r.
* Kronecker multi-task GPs with a coregionalization default kernel.
* The default prior mean is each output's average.
* The multi-output likelihood gradient averages over the outputs (it followed output 0).
* ``train(asynchronous=True)`` defaults to ``method='local'`` (was ``mcmc``).
* A matrix-valued noise function is deprecated (``FutureWarning``).
* ``posterior_mean_grad``'s ``component`` is deprecated; the gradient covers every
  requested output. ``dS/dx`` was wrong off the diagonal.

See fvgp's HISTORY for the details.

8.6.1 — 2026-10-14
------------------

Requires fvgp 4.9.2. That release never runs CG on a broken preconditioner: a factor
containing NaN or inf (an incomplete factorization that broke down silently) counts as a
failed build, any CG solve that fails is redone with unpreconditioned MINRES, and CG is
capped at 5000 iterations, 500 with a preconditioner (``args["sparse_cg_maxiter"]``). The
constructor docstrings describe this. No gpCAM code changes.

8.6.0 — 2026-10-14
------------------

Requires fvgp 4.9.1, and passes its new options through.

Breaking changes
~~~~~~~~~~~~~~~~

* The constructor argument ``ram_economy`` of ``GPOptimizer`` and ``fvGPOptimizer`` is
  now ``gradient_per_direction``, following fvgp 4.9. It never governed RAM alone: it
  decides whether a user's ``kernel_function_grad`` and ``noise_function_grad`` take a
  ``direction`` argument and return one hyperparameter's slice, or return the full stack.
  Optimizers pickled with the old name load unchanged.

Changes inherited from fvgp 4.9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

gpCAM overrides none of these methods, so they reach every ``GPOptimizer``; the docstrings
are fvgp's.

* **The default trainer is local optimization with restarts.** ``gpo.train()`` without a
  ``method`` runs ``method='local'`` from ``n_restarts=5`` starting points and keeps the
  best -- as accurate on held-out data as MCMC in fvgp's measurements, 2-4x faster. Under
  gp2Scale and with ``asynchronous=True`` it stays ``mcmc``. Default fits therefore change:
  they return the maximum-likelihood point rather than the MCMC posterior median, and
  ``mcmc_info`` is not filled. Explicit ``method='local'`` calls now also restart five
  times; ``n_restarts=1`` restores a single run. ``GPOptimizer.optimize()`` uses the
  default for its first training too; its in-loop re-training (``training_method``,
  default ``"global"``) is unchanged.
* **MCMC fixed.** The proposal ignored its adapted step size, the stopping rule fired at
  random, the result was the median of the last 1% of the chain, and the proposal's shape
  was learned from the last 10 states only. ``train(method='mcmc')`` now reaches its
  target acceptance, stops on a convergence check, returns the median after burn-in, and
  samples positive scale-like hyperparameters in log space. New keywords:
  ``mcmc_burn_in``, ``mcmc_break_condition``, ``mcmc_seed``, ``mcmc_log_scale``; new
  ``gpcam.gp_mcmc.ess`` and ``split_rhat`` for multi-chain checks.
* **gp2Scale.** Preconditioned modes build their preconditioner lazily during training
  (``args["sparse_preconditioner_lazy"]``, ``"sparse_preconditioner_settle_steps"``), and
  the stochastic log-determinant's 1% target (``random_logdet_error_rtol``) now holds for
  negative log-determinants, which used to run all 5000 probes; ``random_logdet_error_atol``
  sets an absolute target.
* Measured noise is no longer added to prediction points that merely match the data in
  number, and heteroscedastic ``noise_variances`` warn when their mean is used as the noise
  at new points.

Documentation
~~~~~~~~~~~~~

* Constructor docstrings re-synchronized with fvgp 4.9 (``gradient_per_direction``, the new
  ``args`` keys).
* The experiment-designer, troubleshooting and gp2Scale skills describe the new trainer,
  the MCMC and the gp2Scale changes. The gp2Scale skill's ``gpMCMC`` example passed the
  prior function where ``bounds`` belongs and is corrected.

8.5.0 — 2026-08-12
------------------

A minor-version bump rather than a patch: ``from gpcam import AutonomousExperimenterGP``
now raises ``ImportError``. Nothing that *worked* stops working -- those classes had
raised on construction for several releases -- but an import-level break does not belong
in a patch release. The release also carries the Python 3.10--3.14 support and the
dependency ranges, which existed only in the repository until now.

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

* Require ``fvgp ~= 4.8.8`` (was ``~= 4.8.6``). ``~=`` already admitted any later 4.8.z,
  so this raises the floor only, but 4.8.7/4.8.8 are where the behavior gpCAM's docstrings
  now describe actually lives. Three things come with it, all inherited rather than wrapped:

  - **Row-wise distribution is genuinely row-wise.** A ``"rowwise"`` worker evaluates one
    strip against every column in a single kernel call, so ``gp2Scale_batch_size`` is a
    *strip width* there and a *block side* under ``"blockwise"`` -- two different memory
    budgets, and not comparable numbers. Both are maxima with a remainder rather than
    exact divisors. The budget also decides whether to distribute at all, which is what
    stops a tall thin cross-covariance from being shredded into hundreds of tiny tasks;
    cross-covariances are striped regardless of the setting.
  - **``train(accept_only_if_improved=True)``**, the new default: synchronous training
    with ``method`` in ``'local'``, ``'hgdl'``, ``'adam'`` or a callable now keeps the
    incumbent hyperparameters when the optimizer returns worse ones, and warns.
    ``'global'``, ``'mcmc'``, ``'bo'`` and asynchronous training are unaffected.
    ``GPOptimizer`` and ``fvGPOptimizer`` inherit ``train`` from fvGP, so this needs no
    gpCAM code and is available on both.
  - **``method='bo'`` now says so when it returns the hyperparameters it started from.**
    Nothing beating the warm start is a legitimate answer, but it is indistinguishable
    from a run that silently did nothing. The warning reports how many points were
    evaluated, why the run stopped, and recommends ``method='local'`` or ``'mcmc'``,
    which follow the variance/length-scale ridge of a marginal likelihood that a
    space-filling design and an axis-aligned surrogate both miss. Worth knowing before
    tuning many hyperparameters through ``GPOptimizer.train``.

* gpCAM supports Python 3.10 through 3.14, and CI tests all five (was 3.11 and 3.12).
  The obstacle was never the code but the ``~=`` pins: ``scipy ~= 1.16.0`` requires
  Python >= 3.11 and has no cp314 wheel, so no single pin can span the range. The
  scipy/numpy/dask/distributed requirements are ranges now and pip resolves per
  interpreter -- 3.10 lands on scipy 1.13 / dask 2024.1 through hgdl 2.2.3, while 3.11
  and up get the 2025-era stack. In practice these versions are decided upstream anyway:
  fvgp depends on hgdl, and hgdl ``~=``-pins the whole scientific stack, so hgdl is what
  picks the versions a given Python gets. ``fvgp ~= 4.8.6`` is unchanged and is what made
  3.10 reachable at all -- 4.8.5 declared ``requires_python >= 3.11``.

  The workflow also moves off ``actions/checkout@v2`` and ``actions/setup-python@v5``'s
  predecessor, which cannot install 3.13 or 3.14, and gains ``fail-fast: false`` so one
  version failing no longer hides the others.

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
