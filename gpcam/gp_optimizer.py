#!/usr/bin/env python
import warnings
import numpy as np
from scipy.special import expit, logit
from fvgp import fvGP
from .gp_optimizer_base import GPOptimizerBase


# TODO (for gpCAM)
#    see fvgp "gp.py" for TODOs

class GPOptimizer(GPOptimizerBase):
    """
    This class is an optimization extension of the :doc:`fvgp <fvgp:index>` package
    for single-task (scalar-valued) Gaussian Processes.
    Gaussian Processes can be initialized, trained, and conditioned; also
    the posterior can be evaluated and used via an acquisition function,
    and plugged into optimizers to find maxima.

    V ... number of input points

    D ... input space dimensionality

    N ... arbitrary integers (N1, N2,...)


    Parameters
    ----------
    x_data : np.ndarray or list, optional
        The input point positions. Shape (V x D), where D is the :py:attr:`fvgp.GP.index_set_dim`.
        For single-task GPs, the index set dimension = input space dimension.
        For multi-task GPs, the index set dimension = input space dimension + 1.
        If dealing with non-Euclidean inputs
        x_data should be a list, not a numpy array.
        In this case, both the index set and the input space dim are set to 1.
        If not provided here the GP will be initiated after ``tell()``.
    y_data : np.ndarray, optional
        The values of the data points. Shape (V) or (V, N). If shape (V,N) the algorithm will run N independent GPs.
        This is not to be confused with multi-task learning. In this case, all GPs have to have the same prior
        mean function.
        If not provided here the GP will be initiated after ``tell()``.
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used to initiate the GP.
        The default is an array of ones with the right length for the anisotropic Matern
        kernel with automatic relevance determination (ARD). If ``gp2Scale`` is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
        The full hyperparameter vector is passed to the kernel, mean, and noise callables,
        but the index ranges used by each callable are **disjoint and user-defined**.
        Each callable must only read the indices reserved for it. The gradient
        computation relies on this: when a hyperparameter index belongs to the mean
        function its kernel derivative is assumed zero, and vice versa.
    noise_variances : np.ndarray, optional
        A numpy array defining the uncertainties/noise in the
        ``y_data`` in form of a point-wise variance. Shape (V).
        Note: if no noise_variances are provided here, the noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to ``abs(np.mean(y_data)) / 100.0``. If
        noise covariances are required (correlated noise), make use of the ``noise_function``.
        Only provide a noise function OR ``noise_variances``, not both.
        If the shape of ``y_data`` is (V,N) the noise is still of shape (V), e.g., the outputs
        must have the same noise in this scenario.
    compute_device : str, optional
        One of ``cpu`` or ``gpu``, determines how linear algebra computations are executed. The default is ``cpu``.
        For ``gpu``, pytorch or cupy has to be installed manually. For advanced options see ``args``.
        If ``gp2Scale`` is enabled but no kernel is provided, the choice of the ``compute_device``
        will be particularly important. In that case, the default Wendland kernel will be computed on
        the cpu or the gpu which will significantly change the compute time depending on the compute
        architecture.
    kernel_function : Callable, optional
        A symmetric positive definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters, [args]).
        ``args`` is optional and is used to make :py:attr:`fvgp.GP.args` available.
        The input ``x1`` is a N1 x D array of positions, ``x2`` is a N2 x D
        array of positions, the hyperparameters argument
        is a 1d array of length D+1 for the default kernel and of a different
        length for user-defined kernels.
        The default is a stationary anisotropic kernel
        (:py:meth:`fvgp.GP.default_kernel`) which performs automatic relevance determination (ARD).
        The output is a matrix, an N1 x N2 numpy array.
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the kernel (disjoint from mean and noise indices).
    kernel_function_grad : Callable, optional
        A function that calculates the derivative of the ``kernel_function`` with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input ``x1`` (a N1 x D array of positions),
        ``x2`` (a N2 x D array of positions) and
        ``hyperparameters`` (a 1d array of length D+1 for the default kernel).
        The default is an analytical gradient for the default kernel or a finite difference calculation otherwise.
        If ``ram_economy`` is True, the function's input is x1, x2, hyperparameters (numpy array), and a direction (int).
        The output is a numpy array of shape (len(hps) x N).
        If ``ram_economy`` is ``False``, the function's input is x1, x2, and hyperparameters.
        The output is a numpy array of shape (len(hyperparameters) x N1 x N2). See ``ram_economy``.
    prior_mean_function : Callable, optional
        A function f(x, hyperparameters, [args]) that evaluates the prior mean at a set of input position.
        It accepts as input
        an array of positions (of shape N1 x D) and hyperparameters (a 1d array of length D+1 for the default kernel).
        Optionally, the third argument ``args`` can be defined.
        The return value is a 1d array of length N1.
        If prior_mean_function is not provided, :py:meth:`fvgp.GP._default_mean_function` is used,
        which is the average of the ``y_data``.
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the mean function (disjoint from kernel and noise indices).
    prior_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the ``prior_mean_function`` at
        a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x D) and hyperparameters
        (a 1d array of length D+1 for the default kernel).
        The return value is a 2d array of
        shape (len(hyperparameters) x N1). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparameters,
        or a finite-difference approximation
        is used if ``prior_mean_function`` is provided.
    noise_function : Callable, optional
        The noise function is a callable f(x,hyperparameters, [args]) that returns a
        vector (1d np.ndarray) of len(x), a matrix of shape (length(x),length(x)) or a sparse matrix
        of the same shape.
        The third argument ``args`` is optional.
        The input ``x`` is a numpy array of shape (N x D). The hyperparameter array is the same
        that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the noise function (disjoint from kernel and mean indices).
    noise_function_grad : Callable, optional
        A function that evaluates the gradient of the ``noise_function``
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x D) and
        hyperparameters (a 1d array of length D+1 for the default kernel).
        The return value is a 2d np.ndarray of
        shape (len(hyperparameters) x N) or a 3d np.ndarray of shape (len(hyperparameters) x N x N).
        If None is provided, either
        zeros are returned since the default noise function does not depend on
        hyperparameters, or, if ``noise_function`` is provided but no noise function gradient,
        a finite-difference approximation will be used.
        The same rules regarding ``ram_economy`` as for the kernel definition apply here.
        That means the function will have an additional ``direction`` parameter.
    gp2Scale: bool, optional
        Turns on gp2Scale. This will distribute the covariance computations across multiple workers.
        This is an advanced feature for HPC GPs up to 10
        million data points. If gp2Scale is used, the default kernel is an anisotropic
        Wendland kernel which is compactly supported. There are a few
        things to consider (read on); this is an advanced option.
        If no kernel is provided, the ``compute_device`` option should be revisited.
        The default kernel will use the specified device to compute covariances.
        The default is False.
    gp2Scale_batch_size : int, optional
        Matrix batch size for distributed computing in gp2Scale. The default is 10000.
    dask_client : dask.distributed.Client, optional
        A dask client for gp2Scale, asynchronous training, and certain linear algebra operations.
        On HPC architecture, this client is provided by the job script. Please have a look at the examples.
        A local client is used as the default.
    linalg_mode : str, optional
        Controls the linear-algebra backend used to solve (K+V)x=b and compute log|K+V|.
        The default is ``None``, which selects ``"Chol"`` for standard GPs and automatically
        picks the best sparse mode for gp2Scale GPs.

        **Recommended for standard (non-gp2Scale) GPs:**

        * ``"Chol"`` *(default)* — Cholesky factorization; numerically stable and memory-efficient.
        * ``"CholInv"`` — Cholesky factorization, then explicitly stores the inverse; speeds up posterior
          covariance evaluation 3–10×. Avoid for datasets larger than ~5 000 points due to memory
          and numerical cost. Training always uses the Cholesky factor for stability.
        * ``"Inv"`` — computes and stores the explicit inverse directly (no Cholesky) even during training. Only suitable for
          very small datasets where posterior covariance is computed many times.

        **Specialized for gp2Scale (sparse covariance matrices):**

        * ``"sparseLU"`` — sparse LU factorization; good default for sparse systems up to ~50 000 points.
        * ``"sparseCG"`` — sparse conjugate-gradient iterative solver.
        * ``"sparseMINRES"`` — sparse MINRES iterative solver.
        * ``"sparseSolve"`` — direct sparse solve via scipy.
        * ``"sparseCGpre"`` — preconditioned conjugate-gradient. The preconditioner type
          is selected by ``args["sparse_preconditioner_type"]`` (default ``"ilu"``;
          also ``"ic"``/``"incomplete_cholesky"``, ``"block_jacobi"``,
          ``"schwarz"``/``"additive_schwarz"``, or ``"amg"`` (requires pyamg)).
        * ``"sparseMINRESpre"`` — preconditioned MINRES; same preconditioner choices.
        * ``"sparseCGpre_<type>"`` / ``"sparseMINRESpre_<type>"`` — shortcut that sets
          ``args["sparse_preconditioner_type"]`` to ``<type>`` (e.g. ``"sparseCGpre_amg"``).

        **Custom solver (any GP):**

        Pass an iterable of three callables ``[f_factor, f_solve, f_logdet]``:

        * ``f_factor(K)`` — receives the covariance matrix and returns a factorization object
          (or the matrix itself if no factorization is needed).
        * ``f_solve(obj, b)`` — solves the linear system and returns the solution vector.
        * ``f_logdet(obj)`` — returns the log-determinant as a scalar.

        **Migration note:** the ``calc_inv`` option from earlier gpCAM versions was removed;
        use ``linalg_mode="CholInv"`` (or ``"Inv"``) for the equivalent stored-inverse behavior.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the log marginal likelihood is/are used for the training.
        If True, components of the derivative of the log marginal likelihood are
        calculated sequentially, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (and noise function) with
        respect to the hyperparameters (kernel_function_grad) is
        going to be provided, it has to be tailored: for ``ram_economy=True`` it should be
        of the form f(x, hyperparameters, direction)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If ``ram_economy=False``, the function should be of the form f(x, hyperparameters)
        and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters.
        CAUTION: This array will be stored and is very large.
    cost_function : Callable, optional
        A function encoding the cost of motion through the input
        space and the cost of a measurement. Its inputs
        are an ``origin`` (np.ndarray of size V x D), ``x``
        (np.ndarray of size V x D), and the value of ``cost_func_params``;
        ``origin`` is the starting position, and ``x`` is the
        destination position. The return value is a 1d array of
        length V describing the costs as floats. The 'score' from
        acquisition_function is divided by this
        returned cost to determine the next measurement point.
        The default is a no-op.
    logging : bool, optional
        If True, logging is enabled. The default is False.
    args: dict, optional
        Advanced options. Recognized keys are:

        Stochastic-Lanczos logdet (sparse modes):

        - "random_logdet_lanczos_degree" : int; default = 20
        - "random_logdet_error_rtol" : float; default = 0.01
        - "random_logdet_verbose" : True/False; default = False
        - "random_logdet_print_info" : True/False; default = False
        - "random_logdet_lanczos_compute_device" : str; default = "cpu"/"gpu"

        Sparse iterative solver tolerances and iteration limits:

        - "sparse_cg_tol" : float; default = 1e-5
        - "sparse_minres_tol" : float; default = 1e-5
        - "sparse_cg_maxiter" : int; default = None (use scipy default)
        - "sparse_minres_maxiter" : int; default = None (use scipy default)
        - "sparse_krylov_maxiter" : int; default = None (applies to both if the
          solver-specific key is not set)
        - "sparse_block_krylov" : True/False; default = False — use a block CG
          variant when there are multiple RHS columns
        - "sparse_krylov_mode" : "single"/"block"; equivalent toggle
        - "sparse_krylov_block_size" : int — RHS block size for block CG

        Iterative-solver acceleration (``sparseCG``/``sparseMINRES`` and the
        ``*pre`` variants):

        - "sparse_krylov_warm_start" : True/False; default = False — feed the
          previous training iteration's ``KVinvY`` as ``x0`` to the next solve
        - "sparse_preconditioner_type" : str; default = "ilu". One of "ilu",
          "ic"/"ichol"/"incomplete_cholesky", "block_jacobi", "schwarz"/
          "additive_schwarz", "amg" (requires pyamg)
        - "sparse_preconditioner_refresh_interval" : int; default = 1 —
          reuse the cached preconditioner for up to N consecutive solves
          before rebuilding. ``set_KV`` always force-refreshes.
        - "sparse_preconditioner_block_size" : int — block size for block_jacobi
          and additive_schwarz partitions
        - "sparse_preconditioner_schwarz_overlap" : int — overlap layers for
          additive Schwarz
        - "sparse_preconditioner_drop_tol" / "sparse_preconditioner_fill_factor"
          — forwarded to scipy ``spilu`` for "ilu"
        - "sparse_preconditioner_amg_*" — forwarded to pyamg
          (``max_levels``, ``max_coarse``, ``strength``, ``cycle``, etc.)
        - "sparse_preconditioner_shift" / "_growth" / "_attempts" — diagonal
          shift retry knobs for "ic" / "block_jacobi" / "additive_schwarz" when
          a local Cholesky encounters a non-PD block

        Cholesky compute-device routing:

        - "Chol_factor_compute_device" : str; default = "cpu"/"gpu"
        - "update_Chol_factor_compute_device": str; default = "cpu"/"gpu"
        - "Chol_solve_compute_device" : str; default = "cpu"/"gpu"
        - "Chol_logdet_compute_device" : str; default = "cpu"/"gpu"

        GPU backend:

        - "GPU_engine" : "torch"/"cupy"; default = first available
        - "GPU_device" : str; e.g. "cuda:1" or "mps"
        - "GPU_device_index" : int — explicit CUDA device index

        All other keys will be stored and are available as part of the object instance and
        in kernel, mean, and noise functions.

    Attributes
    ----------
    x_data : np.ndarray
        Datapoint positions.
    y_data : np.ndarray
        Datapoint values.
    noise_variances : np.ndarray
        Datapoint observation variances.
    hyperparameters : np.ndarray
        Current hyperparameters in use.
    K : np.ndarray
        Current prior covariance matrix of the GP.
    m : np.ndarray
        Current prior mean vector.
    V : np.ndarray
        the noise covariance matrix.
    """

    def __init__(
            self,
            x_data=None,
            y_data=None,
            init_hyperparameters=None,
            noise_variances=None,
            compute_device="cpu",
            kernel_function=None,
            kernel_function_grad=None,
            noise_function=None,
            noise_function_grad=None,
            prior_mean_function=None,
            prior_mean_function_grad=None,
            gp2Scale=False,
            dask_client=None,
            gp2Scale_batch_size=10000,
            linalg_mode=None,
            ram_economy=False,
            cost_function=None,
            logging=False,
            args=None
    ):
        super().__init__(x_data=x_data,
                         y_data=y_data,
                         init_hyperparameters=init_hyperparameters,
                         noise_variances=noise_variances,
                         compute_device=compute_device,
                         kernel_function=kernel_function,
                         kernel_function_grad=kernel_function_grad,
                         noise_function=noise_function,
                         noise_function_grad=noise_function_grad,
                         prior_mean_function=prior_mean_function,
                         prior_mean_function_grad=prior_mean_function_grad,
                         gp2Scale=gp2Scale,
                         dask_client=dask_client,
                         gp2Scale_batch_size=gp2Scale_batch_size,
                         linalg_mode=linalg_mode,
                         ram_economy=ram_economy,
                         cost_function=cost_function,
                         logging=logging,
                         multi_task=False,
                         args=args)


class fvGPOptimizer(GPOptimizerBase, fvGP):
    """
    This class is an optimization extension of the :doc:`fvgp <fvgp:index>`
    package for multi-task (vector-valued) Gaussian Processes.
    Gaussian Processes can be initialized, trained, and conditioned; also
    the posterior can be evaluated and used via an acquisition function,
    and plugged into optimizers to find maxima.

    V ... number of input points

    Di... input space dimensionality

    No... number of outputs

    N ... arbitrary integers (N1, N2,...)


    The main logic of :doc:`fvgp <fvgp:index>` is that any multi-task GP is just a single-task GP
    over a Cartesian product space of input and output space, as long as the kernel
    is flexible enough, so prepare to work on your kernel. This is the best
    way to give the user optimal control and power. At various instances, for example
    prior-mean function, noise function, and kernel function definitions, you will
    see that the input `x` is defined over this combined space.
    For example, if your input space is a Euclidean 2d space and your output
    is labelled [0,1], the input to the mean, kernel, and noise functions might be

    x =

    [[0.2, 0.3,0],[0.9,0.6,0],

    [0.2, 0.3,1],[0.9,0.6,1]]

    This has to be understood and taken into account when customizing gpCAM for multi-task
    use. The examples will provide deeper insights.


    Parameters
    ----------
    x_data : np.ndarray | list, optional
        The input point positions. Shape (V x Di), where Di is the :py:attr:`fvgp.fvGP.input_set_dim`.
        For multi-task GPs, the index set dimension = input space dimension + 1.
        If dealing with non-Euclidean inputs
        x_data should be a list, not a numpy array.
        In this case, both the index set and the input space dim are set to 1.
        If not provided here the GP will be initiated after ``tell()``.
    y_data : np.ndarray, optional
        The values of the data points. Shape (V,No).
        It is possible that not every entry in ``x_data``
        has all corresponding tasks available. In that case ``y_data`` may have np.nan as the corresponding entries.
        If not provided here the GP will be initiated after ``tell()``.
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used to initiate the GP.
        The default is an array of ones with the right length for the anisotropic Matern
        kernel with automatic relevance determination (ARD). The task direction is
        simply considered a separate dimension. If ``gp2Scale`` is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
        The full hyperparameter vector is passed to the kernel, mean, and noise callables,
        but the index ranges used by each callable are **disjoint and user-defined**.
        Each callable must only read the indices reserved for it. The gradient
        computation relies on this: when a hyperparameter index belongs to the mean
        function its kernel derivative is assumed zero, and vice versa.
    noise_variances : np.ndarray, optional
        A numpy array defining the uncertainties/noise in the
        ``y_data`` in form of a point-wise variance. Shape (V, No).
        If ``y_data`` has np.nan entries, the corresponding
        ``noise_variances`` have to be np.nan.
        Note: if no noise_variances are provided here, the noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to ``abs(np.mean(y_data)) / 100.0``. If
        noise covariances are required (correlated noise), make use of the ``noise_function``.
        Only provide a noise function OR ``noise_variances``, not both.
    compute_device : str, optional
        One of ``cpu`` or ``gpu``, determines how linear algebra computations are executed. The default is ``cpu``.
        For ``gpu``, pytorch or cupy has to be installed manually. For advanced options see ``args``.
        If ``gp2Scale`` is enabled but no kernel is provided, the choice of the ``compute_device``
        will be particularly important. In that case, the default Wendland kernel will be computed on
        the cpu or the gpu which will significantly change the compute time depending on the compute
        architecture.
    kernel_function : Callable, optional
        A symmetric positive definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters, [args]).
        ``args`` is optional and is used to make :py:attr:`fvgp.GP.args` available.
        The input ``x1`` is a N1 x Di+1 array of positions, ``x2`` is a N2 x Di+1
        array of positions, the hyperparameters argument
        is a 1d array of length N depending on how many hyperparameters are initialized.
        The default is a stationary anisotropic kernel
        (:py:meth:`fvgp.GP.default_kernel`) which performs automatic relevance determination (ARD). The task
        direction is simply considered an additional dimension. This kernel should only be used for tests and in the
        simplest of cases.
        The output is a matrix, an N1 x N2 numpy array.
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the kernel (disjoint from mean and noise indices).
    kernel_function_grad : Callable, optional
        A function that calculates the derivative of the ``kernel_function`` with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input ``x1`` (a N1 x Di + 1 array of positions),
        ``x2`` (a N2 x Di + 1 array of positions) and
        ``hyperparameters`` (a 1d array of length Di+2 for the default kernel).
        The default is an analytical gradient for the default kernel or a finite difference calculation otherwise.
        If ``ram_economy`` is True, the function's input is x1, x2, hyperparameters (numpy array), and a direction (int).
        The output is a numpy array of shape (len(hps) x N).
        If ``ram_economy`` is ``False``, the function's input is x1, x2, and hyperparameters.
        The output is a numpy array of shape (len(hyperparameters) x N1 x N2). See ``ram_economy``.
    prior_mean_function : Callable, optional
        A function f(x, hyperparameters, [args]) that evaluates the prior mean at a set of input position.
        It accepts as input
        an array of positions (of shape N1 x Di+1) and
        hyperparameters (a 1d array of length Di+2 for the default kernel).
        Optionally, the third argument ``args`` can be defined.
        The return value is a 1d array of length N1. If None is provided,
        :py:meth:`fvgp.GP._default_mean_function` is used, which is the average of the ``y_data``.
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the mean function (disjoint from kernel and noise indices).
    prior_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the ``prior_mean_function`` at
        a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x Di+1) and hyperparameters
        (a 1d array of length Di+2 for the default kernel).
        The return value is a 2d array of
        shape (len(hyperparameters) x N1). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparameters,
        or a finite-difference approximation
        is used if ``prior_mean_function`` is provided.
    noise_function : Callable, optional
        The noise function is a callable f(x,hyperparameters, [args]) that returns a
        vector (1d np.ndarray) of len(x), a matrix of shape (length(x),length(x)) or a sparse matrix
        of the same shape.
        The third argument ``args`` is optional.
        The input ``x`` is a numpy array of shape (N x Di+1). The hyperparameter array is the same
        that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
        This callable receives the full hyperparameter vector but must only use
        the indices reserved for the noise function (disjoint from kernel and mean indices).
    noise_function_grad : Callable, optional
        A function that evaluates the gradient of the ``noise_function``
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x Di+1) and
        hyperparameters (a 1d array of length Di+1 for the default kernel).
        The return value is a 2d np.ndarray of
        shape (len(hyperparameters) x N) or a 3d np.ndarray of shape (len(hyperparameters) x N x N).
        If None is provided, either
        zeros are returned since the default noise function does not depend on
        hyperparameters, or, if ``noise_function`` is provided but no noise function gradient,
        a finite-difference approximation will be used.
        The same rules regarding ``ram_economy`` as for the kernel definition apply here.
        That means the function will have an additional ``direction`` parameter.
    gp2Scale: bool, optional
        Turns on gp2Scale. This will distribute the covariance computations across multiple workers.
        This is an advanced feature for HPC GPs up to 10
        million data points. If gp2Scale is used, the default kernel is an anisotropic
        Wendland kernel which is compactly supported. There are a few
        things to consider (read on); this is an advanced option.
        If no kernel is provided, the ``compute_device`` option should be revisited.
        The default kernel will use the specified device to compute covariances.
        The default is False.
    gp2Scale_batch_size : int, optional
        Matrix batch size for distributed computing in gp2Scale. The default is 10000.
    dask_client : dask.distributed.Client, optional
        A dask client for gp2Scale, asynchronous training, and certain linear algebra operations.
        On HPC architecture, this client is provided by the job script. Please have a look at the examples.
        A local client is used as the default.
    linalg_mode : str, optional
        Controls the linear-algebra backend used to solve (K+V)x=b and compute log|K+V|.
        The default is ``None``, which selects ``"Chol"`` for standard GPs and automatically
        picks the best sparse mode for gp2Scale GPs.

        **Recommended for standard (non-gp2Scale) GPs:**

        * ``"Chol"`` *(default)* — Cholesky factorization; numerically stable and memory-efficient.
        * ``"CholInv"`` — Cholesky factorization, then explicitly stores the inverse; speeds up posterior
          covariance evaluation 3–10×. Avoid for datasets larger than ~5 000 points due to memory
          and numerical cost. Training always uses the Cholesky factor for stability.
        * ``"Inv"`` — computes and stores the explicit inverse directly (no Cholesky). Only suitable for
          very small datasets where posterior covariance is computed many times.

        **Specialized for gp2Scale (sparse covariance matrices):**

        * ``"sparseLU"`` — sparse LU factorization; good default for sparse systems up to ~50 000 points.
        * ``"sparseCG"`` — sparse conjugate-gradient iterative solver.
        * ``"sparseMINRES"`` — sparse MINRES iterative solver.
        * ``"sparseSolve"`` — direct sparse solve via scipy.
        * ``"sparseCGpre"`` — preconditioned conjugate-gradient. The preconditioner type
          is selected by ``args["sparse_preconditioner_type"]`` (default ``"ilu"``;
          also ``"ic"``/``"incomplete_cholesky"``, ``"block_jacobi"``,
          ``"schwarz"``/``"additive_schwarz"``, or ``"amg"`` (requires pyamg)).
        * ``"sparseMINRESpre"`` — preconditioned MINRES; same preconditioner choices.
        * ``"sparseCGpre_<type>"`` / ``"sparseMINRESpre_<type>"`` — shortcut that sets
          ``args["sparse_preconditioner_type"]`` to ``<type>`` (e.g. ``"sparseCGpre_amg"``).

        **Custom solver (any GP):**

        Pass an iterable of three callables ``[f_factor, f_solve, f_logdet]``:

        * ``f_factor(K)`` — receives the covariance matrix and returns a factorization object
          (or the matrix itself if no factorization is needed).
        * ``f_solve(obj, b)`` — solves the linear system and returns the solution vector.
        * ``f_logdet(obj)`` — returns the log-determinant as a scalar.

        **Migration note:** the ``calc_inv`` option from earlier gpCAM versions was removed;
        use ``linalg_mode="CholInv"`` (or ``"Inv"``) for the equivalent stored-inverse behavior.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the log marginal likelihood is/are used for the training.
        If True, components of the derivative of the log marginal likelihood are
        calculated sequentially, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (and noise function) with
        respect to the hyperparameters (kernel_function_grad) is
        going to be provided, it has to be tailored: for ``ram_economy=True`` it should be
        of the form f(x, hyperparameters, direction)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If ``ram_economy=False``, the function should be of the form f(x, hyperparameters)
        and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters.
        CAUTION: This array will be stored and is very large.
    cost_function : Callable, optional
        A function encoding the cost of motion through the input
        space and the cost of a measurement. Its inputs
        are an ``origin`` (np.ndarray of size V x D), ``x``
        (np.ndarray of size V x D), and the value of ``cost_func_params``;
        ``origin`` is the starting position, and ``x`` is the
        destination position. The return value is a 1d array of
        length V describing the costs as floats. The 'score' from
        acquisition_function is divided by this
        returned cost to determine the next measurement point.
        The default is a no-op.
    logging : bool, optional
        If True, logging is enabled. The default is False.
    args: dict, optional
        Advanced options. Recognized keys are:

        Stochastic-Lanczos logdet (sparse modes):

        - "random_logdet_lanczos_degree" : int; default = 20
        - "random_logdet_error_rtol" : float; default = 0.01
        - "random_logdet_verbose" : True/False; default = False
        - "random_logdet_print_info" : True/False; default = False
        - "random_logdet_lanczos_compute_device" : str; default = "cpu"/"gpu"

        Sparse iterative solver tolerances and iteration limits:

        - "sparse_cg_tol" : float; default = 1e-5
        - "sparse_minres_tol" : float; default = 1e-5
        - "sparse_cg_maxiter" : int; default = None (use scipy default)
        - "sparse_minres_maxiter" : int; default = None (use scipy default)
        - "sparse_krylov_maxiter" : int; default = None (applies to both if the
          solver-specific key is not set)
        - "sparse_block_krylov" : True/False; default = False — use a block CG
          variant when there are multiple RHS columns
        - "sparse_krylov_mode" : "single"/"block"; equivalent toggle
        - "sparse_krylov_block_size" : int — RHS block size for block CG

        Iterative-solver acceleration (``sparseCG``/``sparseMINRES`` and the
        ``*pre`` variants):

        - "sparse_krylov_warm_start" : True/False; default = False — feed the
          previous training iteration's ``KVinvY`` as ``x0`` to the next solve
        - "sparse_preconditioner_type" : str; default = "ilu". One of "ilu",
          "ic"/"ichol"/"incomplete_cholesky", "block_jacobi", "schwarz"/
          "additive_schwarz", "amg" (requires pyamg)
        - "sparse_preconditioner_refresh_interval" : int; default = 1 —
          reuse the cached preconditioner for up to N consecutive solves
          before rebuilding. ``set_KV`` always force-refreshes.
        - "sparse_preconditioner_block_size" : int — block size for block_jacobi
          and additive_schwarz partitions
        - "sparse_preconditioner_schwarz_overlap" : int — overlap layers for
          additive Schwarz
        - "sparse_preconditioner_drop_tol" / "sparse_preconditioner_fill_factor"
          — forwarded to scipy ``spilu`` for "ilu"
        - "sparse_preconditioner_amg_*" — forwarded to pyamg
          (``max_levels``, ``max_coarse``, ``strength``, ``cycle``, etc.)
        - "sparse_preconditioner_shift" / "_growth" / "_attempts" — diagonal
          shift retry knobs for "ic" / "block_jacobi" / "additive_schwarz" when
          a local Cholesky encounters a non-PD block

        Cholesky compute-device routing:

        - "Chol_factor_compute_device" : str; default = "cpu"/"gpu"
        - "update_Chol_factor_compute_device": str; default = "cpu"/"gpu"
        - "Chol_solve_compute_device" : str; default = "cpu"/"gpu"
        - "Chol_logdet_compute_device" : str; default = "cpu"/"gpu"

        GPU backend:

        - "GPU_engine" : "torch"/"cupy"; default = first available
        - "GPU_device" : str; e.g. "cuda:1" or "mps"
        - "GPU_device_index" : int — explicit CUDA device index

        All other keys will be stored and are available as part of the object instance and
        in kernel, mean, and noise functions.

    Attributes
    ----------
    x_data : np.ndarray | list
        Datapoint positions
    y_data : np.ndarray
        Datapoint values
    noise_variances : np.ndarray
        Datapoint observation variances.
    hyperparameters : np.ndarray
        Current hyperparameters in use.
    K : np.ndarray
        Current prior covariance matrix of the GP
    m : np.ndarray
        Current prior mean vector.
    V : np.ndarray
        the noise covariance matrix or a vector.
    """

    def __init__(
            self,
            x_data=None,
            y_data=None,
            init_hyperparameters=None,
            noise_variances=None,
            compute_device="cpu",
            kernel_function=None,
            kernel_function_grad=None,
            noise_function=None,
            noise_function_grad=None,
            prior_mean_function=None,
            prior_mean_function_grad=None,
            gp2Scale=False,
            dask_client=None,
            gp2Scale_batch_size=10000,
            linalg_mode=None,
            ram_economy=False,
            cost_function=None,
            logging=False,
            args=None
    ):
        super().__init__(x_data=x_data,
                         y_data=y_data,
                         init_hyperparameters=init_hyperparameters,
                         noise_variances=noise_variances,
                         compute_device=compute_device,
                         kernel_function=kernel_function,
                         kernel_function_grad=kernel_function_grad,
                         noise_function=noise_function,
                         noise_function_grad=noise_function_grad,
                         prior_mean_function=prior_mean_function,
                         prior_mean_function_grad=prior_mean_function_grad,
                         gp2Scale=gp2Scale,
                         dask_client=dask_client,
                         gp2Scale_batch_size=gp2Scale_batch_size,
                         linalg_mode=linalg_mode,
                         ram_economy=ram_economy,
                         cost_function=cost_function,
                         logging=logging,
                         multi_task=True,
                         args=args)


class LogGPOptimizer(GPOptimizer):
    """
    A single-task :py:class:`GPOptimizer` for strictly positive observations in (0, inf).

    Observations are modeled in log-space (the GP sees ``log(y)``), and posterior
    predictions are mapped back to the original scale with ``exp`` via
    :py:meth:`evaluate_posterior`, which guarantees strictly positive predictions and
    credible intervals. ``exp`` of a Gaussian is lognormal, so the original-scale mean
    and standard deviation are available in closed form.

    All constructor arguments are identical to :py:class:`GPOptimizer`. Note that the
    inherited :py:meth:`posterior_mean` / :py:meth:`posterior_covariance` operate in
    log-space; use :py:meth:`evaluate_posterior` for the original (positive) scale.

    Acquisition functions: :py:meth:`ask` optimizes the GP in log-space. Because ``log``
    is monotone increasing, ranking acquisitions (``variance``, ``ucb``, ``lcb``,
    ``maximum``, ``minimum``) still identify the same locations as on the original scale.
    For ``target probability``, pass bounds already in log-space (``np.log(a)``, ``np.log(b)``).
    """

    def _prepare(self, y):
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0.0):
            raise ValueError("LogGPOptimizer requires strictly positive observations (y > 0).")
        return y

    def _forward(self, y):
        return np.log(y)

    def _inverse(self, z):
        return np.exp(z)

    def _forward_deriv(self, y):
        return 1.0 / y

    def _moments(self, mu, var):
        # exp(Normal(mu, var)) is lognormal
        mean = np.exp(mu + var / 2.0)
        std = np.sqrt((np.exp(var) - 1.0) * np.exp(2.0 * mu + var))
        return mean, std


class LogitGPOptimizer(GPOptimizer):
    """
    A single-task :py:class:`GPOptimizer` for observations bounded in [0, 1].

    Observations are modeled in logit (log-odds) space (the GP sees ``logit(y)``), and
    posterior predictions are mapped back with the logistic/sigmoid via
    :py:meth:`evaluate_posterior`, which guarantees predictions and credible intervals
    inside (0, 1). Because ``logit(0)`` / ``logit(1)`` are infinite, observations are
    clipped to ``[eps, 1 - eps]`` (a warning is emitted when clipping occurs). The
    logistic-normal distribution has no closed-form moments, so the original-scale mean
    and standard deviation are estimated by Monte-Carlo.

    Parameters
    ----------
    eps : float, optional
        Clipping margin for the open interval; observations are clipped to
        ``[eps, 1 - eps]`` before the logit transform. The default is 1e-6.
    n_samples : int, optional
        Number of Monte-Carlo samples used to estimate the original-scale mean/std in
        :py:meth:`evaluate_posterior`. The default is 10000.

    Notes
    -----
    All other constructor arguments are identical to :py:class:`GPOptimizer`. The
    inherited :py:meth:`posterior_mean` / :py:meth:`posterior_covariance` operate in
    logit-space; use :py:meth:`evaluate_posterior` for the original (0, 1) scale. The
    acquisition-function note for :py:class:`LogGPOptimizer` applies here too (pass
    ``target probability`` bounds in logit-space).
    """

    def __init__(self, x_data=None, y_data=None, eps=1e-6, n_samples=10000, **kwargs):
        self.eps = eps
        self.n_samples = n_samples
        super().__init__(x_data=x_data, y_data=y_data, **kwargs)

    def _prepare(self, y):
        y = np.asarray(y, dtype=float)
        if np.any(y < self.eps) or np.any(y > 1.0 - self.eps):
            warnings.warn("LogitGPOptimizer clipped observations to "
                          f"[{self.eps}, {1.0 - self.eps}] before the logit transform.")
        return np.clip(y, self.eps, 1.0 - self.eps)

    def _forward(self, y):
        return logit(y)

    def _inverse(self, z):
        return expit(z)

    def _forward_deriv(self, y):
        return 1.0 / (y * (1.0 - y))

    def _moments(self, mu, var):
        # sigmoid(Normal(mu, var)) is logistic-normal -> no closed form, estimate by MC
        mu = np.asarray(mu).reshape(-1)
        sd = np.sqrt(np.asarray(var).reshape(-1))
        samples = expit(np.random.normal(loc=mu[:, None], scale=sd[:, None],
                                         size=(mu.shape[0], self.n_samples)))
        return samples.mean(axis=1), samples.std(axis=1)

    def __getstate__(self):
        state = super().__getstate__()
        state["eps"] = self.eps
        state["n_samples"] = self.n_samples
        return state
