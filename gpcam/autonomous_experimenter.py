# /usr/bin/env python

import dask
import numpy as np
from loguru import logger


class AutonomousExperimenterGP:
    """
    THE AutonomousExperimenterGP IS DEPRECIATED. PLEASE USE THE GPOptimizer DIRECTLY.
    AN ALTERNATIVE IS THE TSUCHINOKO PACKAGE.

    This class executes the autonomous loop for a single-task Gaussian process.
    Use class :py:class:`gpcam.AutonomousExperimenterFvGP` for multi-task experiments.
    The AutonomousExperimenter is a convenience-driven functionality that does not allow
    as much customization as using the :py:class:`gpcam.GPOptimizer` directly. But it is a great option to
    get started.


    Parameters
    ----------
    input_space : np.ndarray
        A numpy array of floats of shape D x 2 describing the input space (bounds).
        The autonomous experimenter is only able to handle Euclidean spaces.
        Please use the :py:class:`gpcam.GPOptimizer` to deal with non-Euclidean cases.
    hyperparameters : np.ndarray, optional
        Vector of hyperparameters used by the GP initially.
        This class provides methods to train hyperparameters.
        The default is a random draw from a uniform distribution
        within hyperparameter_bounds, with a shape appropriate
        for the default kernel (D + 1), which is an anisotropic Matern
        kernel with automatic relevance determination (ARD). If gp2Scale is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
    hyperparameter_bounds : np.ndarray, optional
        A 2d numpy array of shape (N x 2), where N is the number of needed hyperparameters.
        The default is None, in which case the hyperparameter_bounds are estimated from the domain size
        and the initial y_data. If the data changes significantly,
        the hyperparameters and the bounds should be changed/retrained. Initial hyperparameters and bounds
        can also be set in the train calls. The default only works for the default kernels.
    instrument_function : Callable, optional
         A function that takes data points (a list of dicts), and returns the same
         with the measurement data filled in. The function is
         expected to communicate with the instrument and perform measurements,
         populating fields of the data input. `y_data` and `noise variance` have to be filled in.
    init_dataset_size : int, optional
        If `x` and `y` are not provided and `dataset` is not provided,
        `init_dataset_size` must be provided. An initial
        dataset is constructed randomly with this length. The `instrument_function`
        is immediately called to measure values
        at these initial points.
    acquisition_function : Callable, optional
        The acquisition function accepts as input a numpy array
        of size V x D (such that V is the number of input
        points, and D is the parameter space dimensionality) and
        a `GPOptimizer` object. The return value is 1d array
        of length V providing 'scores' for each position,
        such that the highest scored point will be measured next.
        Built-in functions can be used by one of the following keys:
        `ucb`,`lcb`,`maximum`,
        `minimum`, `variance`,`expected_improvement`,
        `relative information entropy`,`relative information entropy set`,
        `probability of improvement`, `gradient`,`total correlation`,`target probability`.
        If None, the default function `variance`, meaning
        `fvgp.GP.posterior_covariance` with variance_only = True will be used.
        The acquisition function can be a callable function of the form my_func(x,gpcam.GPOptimizer)
        which will be maximized (!!!), so make sure desirable new measurement points
        will be located at maxima.
        Explanations of the acquisition functions:
        variance: simply the posterior variance
        relative information entropy: the KL divergence of the prior over predictions and the posterior
        relative information entropy set: the KL divergence of the prior
        defined over predictions and the posterior point-by-point
        ucb: upper confidence bound, posterior mean + 3. std
        lcb: lower confidence bound, -(posterior mean - 3. std)
        maximum: finds the maximum of the current posterior mean
        minimum: finds the maximum of the current posterior mean
        gradient: puts focus on high-gradient regions
        probability of improvement: as the name would suggest
        expected improvement: as the name would suggest
        total correlation: extension of mutual information to more than 2 random variables
        target probability: probability of a target; needs a dictionary
        GPOptimizer.args = {'a': lower bound, 'b': upper bound} to be defined.
    cost_function : Callable, optional
        A function encoding the cost of motion through the input space and the
        cost of a measurements. Its inputs are an
        `origin` (np.ndarray of size V x D), `x` (np.ndarray of size V x D),
        and the value of `cost_function_parameters`;
        `origin` is the starting position, and `x` is the destination position. The return value is a 1d array of
        length V describing the costs as floats. The 'score' from
        acquisition_function is divided by this returned cost to determine
        the next measurement point. If None, the default is a uniform cost of 1.
    cost_update_function : Callable, optional
        A function that updates the `cost_function_parameters` which are communicated to the `cost_function`.
        This function accepts as input
        costs (a list of cost values determined by `instrument_function`), bounds (a V x 2 numpy array) and a parameters
        object. The default is a no-op.
    cost_function_parameters : Any, optional
        An object that is communicated to the `cost_function` and `cost_update_function`. The default is `{}`.
    online : bool, optional
        The default is True. `online=True` will lead to calls to `gpOptimizer.tell(append=True)` which
        potentially saves a lot of time in the GP update. The GP is updated either with an inversion update
        or a Cholesky factor update.
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
    prior_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x D) and hyperparameters (a 1d array of length D+1 for the default kernel).
        The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used, which is the average of the `y_data`.
    noise_function : Callable optional
        The noise function is a callable f(x,hyperparameters) that returns a
        vector (1d np.ndarray) of length(x).
        The input `x` is a numpy array of shape (N x D). The hyperparameter array is the same
        that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
    run_every_iteration : Callable, optional
        A function that is run at every iteration. It accepts as input a
        `gpcam.AutonomousExperimenterGP` instance. The default is a no-op.
    x_data : np.ndarray, optional
        Initial data point positions.
    y_data : np.ndarray, optional
        Initial data point values.
    noise_variances : np.ndarray, optional
        Initial data point observation variances.
    dataset : string, optional
        A filename of a gpcam-generated file that is used to initialize a new instance.
    communicate_full_dataset : bool, optional
        If True, the full dataset will be communicated to the `instrument_function`
        on each iteration. If False, only the
        newly suggested data points will be communicated. The default is False.
    compute_device : str, optional
        One of `cpu` or `gpu`, determines how linear algebra computations are executed. The default is `cpu`.
    calc_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance
        matrix after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster (3-10 times).
        For larger problems (>2000 data points), the use of inversion should be avoided due
        to computational instability and costs. The default is
        False. Note, the training will not use the
        inverse for stability reasons. Storing the inverse is
        a good option when the dataset is not too large and the posterior covariance is heavily used.
        Caution: this option, together with `append=True` in `tell()` will mean that the inverse of
        the covariance is updated, not recomputed, which can lead to instability.
        In application where data is appended many times, it is recommended to either turn
        `calc_inv` off, or to regularly force the recomputation of the inverse via `gp_rank_n_update` in
        `update_gp_data`.
    training_dask_client : distributed.client.Client, optional
        A Dask Distributed Client instance for distributed training. If None is provided, a new
        `dask.distributed.Client` instance is constructed.
    acq_func_opt_dask_client : distributed.client.Client, optional
        A Dask Distributed Client instance for distributed `acquisition_function`
        optimization. If None is provided, a new
        `dask.distributed.Client` instance is constructed.


    Attributes
    ----------
    x_data : np.ndarray
        Data point positions
    y_data : np.ndarray
        Data point values
    noise_variances : np.ndarray
        Data point observation variances
    data.dataset : list
        All data
    hyperparameter_bounds : np.ndarray
        A 2d array of floats of size J x 2, such that J is the length
        matching the length of `hyperparameters` defining
        the bounds for training.
    gp_optimizer : gpcam.GPOptimizer
        A GPOptimizer instance used for initializing a Gaussian process and performing optimization of the posterior.
    """

    def __init__(self,
                 input_space,
                 hyperparameters=None,
                 hyperparameter_bounds=None,
                 instrument_function=None,
                 init_dataset_size=None,
                 acquisition_function="variance",
                 cost_function=None,
                 cost_update_function=None,
                 cost_function_parameters=None,
                 online=True,
                 kernel_function=None,
                 prior_mean_function=None,
                 noise_function=None,
                 run_every_iteration=None,
                 x_data=None, y_data=None, noise_variances=None, dataset=None,
                 communicate_full_dataset=False,
                 compute_device="cpu",
                 calc_inv=False,
                 training_dask_client=None,
                 acq_func_opt_dask_client=None,
                 gp2Scale=False,
                 gp2Scale_dask_client=None,
                 gp2Scale_batch_size=10000,
                 ram_economy=True,
                 args=None
                 ):

        raise Exception("THE AutonomousExperimenterGP IS DEPRECIATED. PLEASE USE THE GPOptimizer DIRECTLY."
                        "AN ALTERNATIVE IS THE TSUCHINOKO PACKAGE.")


###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
class AutonomousExperimenterFvGP(AutonomousExperimenterGP):
    """
    THE AutonomousExperimenterFvGP IS DEPRECIATED. PLEASE USE THE FvGPOptimizer DIRECTLY.
    AN ALTERNATIVE IS THE TSUCHINOKO PACKAGE.


    Executes the autonomous loop for a multi-task Gaussian process.

    Parameters
    ----------
    input_space : np.ndarray
        A numpy array of floats of shape D x 2 describing the input space (bounds).
        The autonomous experimenter is only able to handle Euclidean spaces.
        Please use the :py:class:`gpcam.fvGPOptimizer`to deal with non-Euclidean cases.
    hyperparameters : np.ndarray, optional
        Vector of hyperparameters used by the GP initially.
        This class provides methods to train hyperparameters.
        The default is a random draw from a uniform distribution
        within hyperparameter_bounds, with a shape appropriate
        for the default kernel (D + 1), which is an anisotropic Matern
        kernel with automatic relevance determination (ARD). If gp2Scale is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
    hyperparameter_bounds : np.ndarray, optional
        A 2d numpy array of shape (N x 2), where N is the number of needed hyperparameters.
        The default is None, in which case the hyperparameter_bounds are estimated from the domain size
        and the initial y_data. If the data changes significantly,
        the hyperparameters and the bounds should be changed/retrained. Initial hyperparameters and bounds
        can also be set in the train calls. The default only works for the default kernels.
    instrument_function : Callable, optional
         A function that takes data points (a list of dicts), and returns the same
         with the measurement data filled in. The function is
         expected to communicate with the instrument and perform measurements,
         populating fields of the data input. `y_data` and `noise variances` have to be filled in.
    init_dataset_size : int, optional
        If `x` and `y` are not provided and `dataset` is not provided,
        `init_dataset_size` must be provided. An initial
        dataset is constructed randomly with this length. The `instrument_function`
        is immediately called to measure values
        at these initial points.
    acquisition_function : Callable, optional
        The acquisition function accepts as input a numpy array
        of size V x D (such that V is the number of input
        points, and D is the parameter space dimensionality) and
        a `GPOptimizer` object. The return value is 1d array
        of length V providing 'scores' for each position,
        such that the highest scored point will be measured next.
        Built-in functions can be used by one of the following keys:
        `variance`, `relative information entropy`,
        `relative information entropy set`, `total correlation`.
        See fvGPOptimizer.ask() for a short explanation of these functions.
        In the multi-task case, it is highly recommended to
        deploy a user-defined acquisition function due to the intricate relationship
        of posterior distributions at different points in the output space.
        If None, the default function `variance`, meaning
        `fvgp.GP.posterior_covariance` with variance_only = True will be used.
        The acquisition function can be a callable function of the form my_func(x,gpcam.GPOptimizer)
        which will be maximized (!!!), so make sure desirable new measurement points
        will be located at maxima.
    cost_function : Callable, optional
        A function encoding the cost of motion through the input space and the
        cost of a measurements. Its inputs are an
        `origin` (np.ndarray of size V x D), `x` (np.ndarray of size V x D),
        and the value of `cost_function_parameters`;
        `origin` is the starting position, and `x` is the destination position. The return value is a 1d array of
        length V describing the costs as floats. The 'score' from
        acquisition_function is divided by this returned cost to determine
        the next measurement point. If None, the default is a uniform cost of 1.
    cost_update_function : Callable, optional
        A function that updates the `cost_func_params` which are communicated to the `cost_function`.
        This function accepts as input
        costs (a list of cost values determined by `instrument_function`), bounds (a V x 2 numpy array) and a parameters
        object. The default is a no-op.
    cost_function_parameters : Any, optional
        An object that is communicated to the `cost_function` and `cost_update_function`. The default is `{}`.
    online : bool, optional
        The default is True. `online=True` will lead to calls to `gpOptimizer.tell(append=True)` which
        potentially saves a lot of time in the GP update. The GP is updated either with an inversion update
        or a Cholesky factor update.
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
    prior_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x Di+1) and
        hyperparameters (a 1d array of length Di+2 for the default kernel).
        The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used, which is the average of the `y_data`.
    noise_function : Callable optional
        The noise function is a callable f(x,hyperparameters) that returns a
        vector (1d np.ndarray) of length(x).
        The input `x` is a numpy array of shape (N x D). The hyperparameter array is the same
        that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
    run_every_iteration : Callable, optional
        A function that is run at every iteration. It accepts as input a
        `gpcam.AutonomousExperimenterGP` instance. The default is a no-op.
    x_data : np.ndarray, optional
        Initial data point positions.
    y_data : np.ndarray, optional
        Initial data point values.
    noise_variances : np.ndarray, optional
        Initial data point observation variances.
    dataset : string, optional
        A filename of a gpcam-generated file that is used to initialize a new instance.
    communicate_full_dataset : bool, optional
        If True, the full dataset will be communicated to the `instrument_function`
        on each iteration. If False, only the
        newly suggested data points will be communicated. The default is False.
    compute_device : str, optional
        One of `cpu` or `gpu`, determines how linear algebra computations are executed. The default is `cpu`.
    calc_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance
        matrix after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster (3-10 times).
        For larger problems (>2000 data points), the use of inversion should be avoided due
        to computational instability and costs. The default is
        False. Note, the training will not use the
        inverse for stability reasons. Storing the inverse is
        a good option when the dataset is not too large and the posterior covariance is heavily used.
        Caution: this option, together with `append=True` in `tell()` will mean that the inverse of
        the covariance is updated, not recomputed, which can lead to instability.
        In application where data is appended many times, it is recommended to either turn
        `calc_inv` off, or to regularly force the recomputation of the inverse via `gp_rank_n_update` in
        `update_gp_data`.
    training_dask_client : distributed.client.Client, optional
        A Dask Distributed Client instance for distributed training. If None is provided, a new
        `dask.distributed.Client` instance is constructed.
    acq_func_opt_dask_client : distributed.client.Client, optional
        A Dask Distributed Client instance for distributed `acquisition_function`
        optimization. If None is provided, a new
        `dask.distributed.Client` instance is constructed.


    Attributes
    ----------
    x_data : np.ndarray
        Data point positions
    y_data : np.ndarray
        Data point values
    noise_variances : np.ndarray
        Data point observation variances
    data.dataset : list
        All data
    hyperparameter_bounds : np.ndarray
        A 2d array of floats of size J x 2, such that J is the length
        matching the length of `hyperparameters` defining
        the bounds for training.
    gp_optimizer : gpcam.GPOptimizer
        A GPOptimizer instance used for initializing a Gaussian process and performing optimization of the posterior.
    """

    def __init__(self,
                 input_space,
                 hyperparameters=None,
                 hyperparameter_bounds=None,
                 instrument_function=None,
                 init_dataset_size=None,
                 acquisition_function="variance",
                 cost_function=None,
                 cost_update_function=None,
                 cost_function_parameters=None,
                 online=True,
                 kernel_function=None,
                 prior_mean_function=None,
                 noise_function=None,
                 run_every_iteration=None,
                 x_data=None, y_data=None, noise_variances=None, dataset=None,
                 communicate_full_dataset=False,
                 compute_device="cpu",
                 calc_inv=False,
                 training_dask_client=None,
                 acq_func_opt_dask_client=None,
                 gp2Scale=False,
                 gp2Scale_dask_client=None,
                 gp2Scale_batch_size=10000,
                 ram_economy=True,
                 args=None
                 ):
        ################################
        # getting the data ready#########
        ################################
        raise Exception("THE AutonomousExperimenterFvGP IS DEPRECIATED. PLEASE USE THE FvGPOptimizer DIRECTLY."
                        "AN ALTERNATIVE IS THE TSUCHINOKO PACKAGE.")
