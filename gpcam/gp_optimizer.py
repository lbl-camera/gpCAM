#!/usr/bin/env python
import numpy as np
from fvgp import fvGP
from .gp_optimizer_base import GPOptimizerBase


# TODO (for gpCAM)
#

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
        If x_data is not provided here the GP will be initiated after `tell()`.
    y_data : np.ndarray, optional
        The values of the data points. Shape (V).
        If not provided here the GP will be initiated after `tell()`.
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used to initiate the GP.
        The default is an array of ones with the right length for the anisotropic Matern
        kernel with automatic relevance determination (ARD). If `gp2Scale` is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
    noise_variances : np.ndarray, optional
        An numpy array defining the uncertainties/noise in the
        `y_data` in form of a point-wise variance. Shape (V).
        Note: if no noise_variances are provided here, the noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to `abs(np.mean(y_data)) / 100.0`. If
        noise covariances are required (correlated noise), make use of the `noise_function`.
        Only provide a noise function OR `noise_variances`, not both.
    compute_device : str, optional
        One of `cpu` or `gpu`, determines how linear algebra computations are executed. The default is `cpu`.
        For `gpu`, pytorch or cupy has to be installed manually. For advanced options see `args`.
        If `gp2Scale` is enabled but no kernel is provided, the choice of the `compute_device`
        will be particularly important. In that case, the default Wendland kernel will be computed on
        the cpu or the gpu which will significantly change the compute time depending on the compute
        architecture.
    kernel_function : Callable, optional
        A symmetric positive definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters).
        The input `x1` is a N1 x D array of positions, `x2` is a N2 x D
        array of positions, the hyperparameters argument
        is a 1d array of length D+1 for the default kernel and of a different
        length for user-defined kernels.
        The default is a stationary anisotropic kernel
        (`fvgp.GP.default_kernel`) which performs automatic relevance determination (ARD).
        The output is a matrix, an N1 x N2 numpy array.
    kernel_function_grad : Callable, optional
        A function that calculates the derivative of the `kernel_function` with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input `x1` (a N1 x D array of positions),
        `x2` (a N2 x D array of positions) and
        `hyperparameters` (a 1d array of length D+1 for the default kernel).
        The default is a finite difference calculation.
        If `ram_economy` is True, the function's input is x1, x2, hyperparameters (numpy array), and a direction (int).
        The output is a numpy array of shape (len(hps) x N).
        If `ram_economy` is `False`, the function's input is x1, x2, and hyperparameters.
        The output is a numpy array of shape (len(hyperparameters) x N1 x N2). See `ram_economy`.
    prior_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x D) and hyperparameters (a 1d array of length D+1 for the default kernel).
        The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used, which is the average of the `y_data`.
    prior_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the `prior_mean_function` at
        a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x D) and hyperparameters
        (a 1d array of length D+1 for the default kernel).
        The return value is a 2d array of
        shape (len(hyperparameters) x N1). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparameters,
        or a finite-difference approximation
        is used if `prior_mean_function` is provided.
    noise_function : Callable, optional
        The noise function is a callable f(x,hyperparameters) that returns a
        vector (1d np.ndarray) of len(x), a matrix of shape (length(x),length(x)) or a sparse matrix
        of the same shape.
        The input `x` is a numpy array of shape (N x D). The hyperparameter array is the same
        that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
    noise_function_grad : Callable, optional
        A function that evaluates the gradient of the `noise_function`
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x D) and
        hyperparameters (a 1d array of length D+1 for the default kernel).
        The return value is a 2d np.ndarray of
        shape (len(hyperparameters) x N) or a 3d np.ndarray of shape (len(hyperparameters) x N x N).
        If None is provided, either
        zeros are returned since the default noise function does not depend on
        hyperparameters, or, if `noise_function` is provided but no noise function,
        a finite-difference approximation will be used.
        The same rules regarding `ram_economy` as for the kernel definition apply here.
        That means the function will have an additional `direction` parameter.
    gp2Scale: bool, optional
        Turns on gp2Scale. This will distribute the covariance computations across multiple workers.
        This is an advanced feature for HPC GPs up to 10
        million data points. If gp2Scale is used, the default kernel is an anisotropic
        Wendland kernel which is compactly supported. There are a few
        things to consider (read on); this is an advanced option.
        If no kernel is provided, the `compute_device` option should be revisited.
        The default kernel will use the specified device to compute covariances.
        The default is False.
    gp2Scale_dask_client : dask.distributed.Client, optional
        A dask client for gp2Scale.
        On HPC architecture, this client is provided by the job script. Please have a look at the examples.
        A local client is used as the default.
    gp2Scale_batch_size : int, optional
        Matrix batch size for distributed computing in gp2Scale. The default is 10000.
    gp2Scale_linalg_mode : str, optional
        One of `Chol`, `sparseLU`, `sparseCG`, `sparseMINRES`, `sparseSolve`, `sparseCGpre`
        (incomplete LU preconditioner), or `sparseMINRESpre`. The default is None which amounts to
        an automatic determination of the mode. For advanced customization options
        this can also be an iterable with three callables: the first f(K), where K is the covariance matrix
        to compute a factorization object
        which is available in the second and third callable. The second being the linear solve f(obj, vec),
        and the third being the logdet=f(obj). If a factorization object is not required, the first callable
        should return the matrix itself (K).
    calc_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance
        matrix after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster (3-10 times).
        For larger problems (>5000 data points), the use of inversion should be avoided due
        to computational instability and costs. The default is
        False. Note, the training will not use the
        inverse for stability reasons. Storing the inverse is
        a good option when the dataset is not too large and the posterior covariance is heavily used.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the log marginal likelihood is/are used for the training.
        If True, components of the derivative of the log marginal likelihood are
        calculated sequentially, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (and noise function) with
        respect to the hyperparameters (kernel_function_grad) is
        going to be provided, it has to be tailored: for `ram_economy=True` it should be
        of the form f(x, hyperparameters, direction)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If `ram_economy=False`, the function should be of the form f(x, hyperparameters)
        and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters.
        CAUTION: This array will be stored and is very large.
    cost_function : Callable, optional
        A function encoding the cost of motion through the input
        space and the cost of a measurement. Its inputs
        are an `origin` (np.ndarray of size V x D), `x`
        (np.ndarray of size V x D), and the value of `cost_func_params`;
        `origin` is the starting position, and `x` is the
        destination position. The return value is a 1d array of
        length V describing the costs as floats. The 'score' from
        acquisition_function is divided by this
        returned cost to determine the next measurement point.
        The default in no-op.
    logging : bool
        If true, logging is enabled. The default is False.
    args: dict, optional
        Advanced options. Recognized keys are:

        - "random_logdet_lanczos_degree" : int; default = 20
        - "random_logdet_error_rtol" : float; default = 0.01
        - "random_logdet_verbose" : True/False; default = False
        - "random_logdet_print_info" : True/False; default = False
        - "sparse_minres_tol" : float
        - "cg_minres_tol" : float
        - "random_logdet_lanczos_compute_device" : str; default = "cpu"/"gpu"
        - "Chol_factor_compute_device" : str; default = "cpu"/"gpu"
        - "update_Chol_factor_compute_device": str; default = "cpu"/"gpu"
        - "Chol_solve_compute_device" : str; default = "cpu"/"gpu"
        - "Chol_logdet_compute_device" : str; default = "cpu"/"gpu"
        - "GPU_engine" : str; default = "torch"/"cupy"

        All other keys will be stored and are available as part of the object instance.

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
            gp2Scale_dask_client=None,
            gp2Scale_batch_size=10000,
            gp2Scale_linalg_mode=None,
            calc_inv=False,
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
                         gp2Scale_dask_client=gp2Scale_dask_client,
                         gp2Scale_batch_size=gp2Scale_batch_size,
                         gp2Scale_linalg_mode=gp2Scale_linalg_mode,
                         calc_inv=calc_inv,
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
    is labelled [0,1], the input to the mean, kernel, and noise function might be

    x =

    [[0.2, 0.3,0],[0.9,0.6,0],

    [0.2, 0.3,1],[0.9,0.6,1]]

    This has to be understood and taken into account when customizing gpCAM for multi-task
    use. The examples will provide deeper insights.


    Parameters
    ----------
    x_data : np.ndarray or list, optional
        The input point positions. Shape (V x Di), where Di is the :py:attr:`fvgp.fvGP.input_set_dim`.
        For multi-task GPs, the index set dimension = input space dimension + 1.
        If dealing with non-Euclidean inputs
        x_data should be a list, not a numpy array.
        In this case, both the index set and the input space dim are set to 1.
    y_data : np.ndarray
        The values of the data points. Shape (V,No).
        It is possible that not every entry in `x_data`
        has all corresponding tasks available. In that case `y_data` may have np.nan as the corresponding entries.
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used to initiate the GP.
        The default is an array of ones with the right length for the anisotropic Matern
        kernel with automatic relevance determination (ARD). The task direction is
        simply considered a separate dimension. If gp2Scale is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
    noise_variances : np.ndarray, optional
        An numpy array or list defining the uncertainties/noise in the
        `y_data` in form of a point-wise variance. Shape (V, No).
        If `y_data` has np.nan entries, the corresponding
        `noise_variances` have to be np.nan.
        Note: if no noise_variances are provided here, the noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to `abs(np.mean(y_data)) / 100.0`. If
        noise covariances are required (correlated noise), make use of the `noise_function`.
        Only provide a noise function OR `noise_variances`, not both.
    compute_device : str, optional
        One of `cpu` or `gpu`, determines how linear algebra computations are executed. The default is `cpu`.
        For `gpu`, pytorch or cupy has to be installed manually. For advanced options see `args`.
        If `gp2Scale` is enabled but no kernel is provided, the choice of the `compute_device`
        will be particularly important. In that case, the default Wendland kernel will be computed on
        the cpu or the gpu which will significantly change the compute time depending on the compute
        architecture.
    kernel_function : Callable, optional
        A symmetric positive definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters).
        The input `x1` a N1 x Di+1 array of positions, `x2` is a N2 x Di+1
        array of positions, the hyperparameters argument
        is a 1d array of length N depending on how many hyperparameters are initialized.
        The default is a stationary anisotropic kernel
        (`fvgp.GP.default_kernel`) which performs automatic relevance determination (ARD). The task
        direction is simply considered an additional dimension. This kernel should only be used for tests and in the
        simplest of cases.
        The output is a matrix, an N1 x N2 numpy array.
    kernel_function_grad : Callable, optional
        A function that calculates the derivative of the `gp_kernel_function` with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input `x1` (a N1 x Di + 1 array of positions),
        `x2` (a N2 x Di + 1 array of positions) and
        `hyperparameters` (a 1d array of length Di+2 for the default kernel).
        The default is a finite difference calculation.
        If `ram_economy` is True, the function's input is x1, x2,hyperparameters (numpy array), and a direction (int).
        The output is a numpy array of shape (len(hps) x N).
        If `ram_economy` is `False`, the function's input is x1, x2, and hyperparameters.
        The output is a numpy array of shape (len(hyperparameters) x N1 x N2). See `ram_economy`.
    prior_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x Di+1) and
         hyperparameters (a 1d array of length Di+2 for the default kernel).
        The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used, which is the average of the `y_data`.
    prior_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the `prior_mean_function` at
        a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x Di+1) and hyperparameters
        (a 1d array of length Di+2 for the default kernel).
        The return value is a 2d array of
        shape (len(hyperparameters) x N1). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparameters,
        or a finite-difference approximation
        is used if `prior_mean_function` is provided.
    noise_function : Callable, optional
        The noise function is a callable f(x,hyperparameters) that returns a
        vector (1d np.ndarray) of len(x), a matrix of shape (length(x),length(x)) or a sparse matrix
        of the same shape.
        The input `x` is a numpy array of shape (N x Di+1). The hyperparameter array is the same
        that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
    noise_function_grad : Callable, optional
        A function that evaluates the gradient of the `noise_function`
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x Di+1) and
        hyperparameters (a 1d array of length D+1 for the default kernel).
        The return value is a 2d np.ndarray of
        shape (len(hyperparameters) x N) or a 3d np.ndarray of shape (len(hyperparameters) x N x N).
        If None is provided, either
        zeros are returned since the default noise function does not depend on
        hyperparameters, or, if `noise_function` is provided but no noise function,
        a finite-difference approximation will be used.
        The same rules regarding `ram_economy` as for the kernel definition apply here.
        That means the function will have an additional `direction` parameter.
    gp2Scale: bool, optional
        Turns on gp2Scale. This will distribute the covariance computations across multiple workers.
        This is an advanced feature for HPC GPs up to 10
        million data points. If gp2Scale is used, the default kernel is an anisotropic
        Wendland kernel which is compactly supported. There are a few
        things to consider (read on); this is an advanced option.
        If no kernel is provided, the `compute_device` option should be revisited.
        The default kernel will use the specified device to compute covariances.
        The default is False.
    gp2Scale_dask_client : dask.distributed.Client, optional
        A dask client for gp2Scale.
        On HPC architecture, this client is provided by the job script. Please have a look at the examples.
        A local client is used as the default.
    gp2Scale_batch_size : int, optional
        Matrix batch size for distributed computing in gp2Scale. The default is 10000.
    gp2Scale_linalg_mode : str, optional
        One of `Chol`, `sparseLU`, `sparseCG`, `sparseMINRES`, `sparseSolve`, `sparseCGpre`
        (incomplete LU preconditioner), or `sparseMINRESpre`. The default is None which amounts to
        an automatic determination of the mode. For advanced customization options
        this can also be an iterable with three callables: the first f(K), where K is the covariance matrix
        to compute a factorization object
        which is available in the second and third callable. The second being the linear solve f(obj, vec),
        and the third being the logdet=f(obj). If a factorization object is not required, the first callable
        should return the matrix itself (K).
    calc_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance
        matrix after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster (3-10 times).
        For larger problems (>5000 data points), the use of inversion should be avoided due
        to computational instability and costs. The default is
        False. Note, the training will not use the
        inverse for stability reasons. Storing the inverse is
        a good option when the dataset is not too large and the posterior covariance is heavily used.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the log marginal likelihood is/are used for the training.
        If True, components of the derivative of the log marginal likelihood are
        calculated sequentially, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (and noise function) with
        respect to the hyperparameters (kernel_function_grad) is
        going to be provided, it has to be tailored: for `ram_economy=True` it should be
        of the form f(x, hyperparameters, direction)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If `ram_economy=False`, the function should be of the form f(x, hyperparameters)
        and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters.
        CAUTION: This array will be stored and is very large.
    cost_function : Callable, optional
        A function encoding the cost of motion through the input
        space and the cost of a measurement. Its inputs
        are an `origin` (np.ndarray of size V x D), `x`
        (np.ndarray of size V x D), and the value of `cost_func_params`;
        `origin` is the starting position, and `x` is the
        destination position. The return value is a 1d array of
        length V describing the costs as floats. The 'score' from
        acquisition_function is divided by this
        returned cost to determine the next measurement point.
        The default in no-op.
    logging : bool
        If true, logging is enabled. The default is False.
    args: dict, optional
        Advanced options. Recognized keys are:

        - "random_logdet_lanczos_degree" : int; default = 20
        - "random_logdet_error_rtol" : float; default = 0.01
        - "random_logdet_verbose" : True/False; default = False
        - "random_logdet_print_info" : True/False; default = False
        - "sparse_minres_tol" : float
        - "cg_minres_tol" : float
        - "random_logdet_lanczos_compute_device" : str; default = "cpu"/"gpu"
        - "Chol_factor_compute_device" : str; default = "cpu"/"gpu"
        - "update_Chol_factor_compute_device": str; default = "cpu"/"gpu"
        - "Chol_solve_compute_device" : str; default = "cpu"/"gpu"
        - "Chol_logdet_compute_device" : str; default = "cpu"/"gpu"
        - "GPU_engine" : str; default = "torch"/"cupy"

        All other keys will be stored and are available as part of the object instance.


    Attributes
    ----------
    x_data : np.ndarray or list
        Datapoint positions
    y_data : np.ndarray
        Datapoint values
    noise_variances : np.ndarray
        Datapoint observation variances.
    fvgp_x_data : np.ndarray or list
        Data points from the fvgp point of view.
    fvgp_y_data : np.ndarray
        The data values from the fvgp point of view.
    fvgp_noise_variances : np.ndarray
        Observation variances from the fvgp point of view.
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
            gp2Scale_dask_client=None,
            gp2Scale_batch_size=10000,
            gp2Scale_linalg_mode=None,
            calc_inv=False,
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
                         gp2Scale_dask_client=gp2Scale_dask_client,
                         gp2Scale_batch_size=gp2Scale_batch_size,
                         gp2Scale_linalg_mode=gp2Scale_linalg_mode,
                         calc_inv=calc_inv,
                         ram_economy=ram_economy,
                         cost_function=cost_function,
                         logging=logging,
                         multi_task=True,
                         args=args)
