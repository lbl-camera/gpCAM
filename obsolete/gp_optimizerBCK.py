#!/usr/bin/env python
import numpy as np
from loguru import logger
from fvgp import fvGP
from fvgp import GP
from . import surrogate_model as sm
import warnings
import random


# TODO (for gpCAM)
#   mixed continuous-discrete space via cartesian product of random draws from continuous and candidates,
#                           in this case allow "input_space" and "input set"/"candidates"
#class GPOptimizer(GPOptimizerBase):
class GPOptimizer(GP):
    """
    This class is an optimization extension of the :doc:`fvgp <fvgp:index>` package
    for single-task (scalar-valued) Gaussian Processes.
    Gaussian Processes can be initialized, trained, and conditioned; also
    the posterior can be evaluated and used via acquisition functions,
    and plugged into optimizers to find their optima. This class inherits all methods from
    the :py:class:`fvgp.GP` class.

    V ... number of input points

    D ... input space dimensionality

    N ... arbitrary integers (N1, N2,...)


    All posterior evaluation functions are inherited from :py:class:`fvgp.GP`.
    These include, but are not limited to:

    :py:meth:`fvgp.GP.posterior_mean`

    :py:meth:`fvgp.GP.posterior_covariance`

    :py:meth:`fvgp.GP.posterior_mean_grad`

    :py:meth:`fvgp.GP.posterior_covariance_grad`

    :py:meth:`fvgp.GP.joint_gp_prior`

    :py:meth:`fvgp.GP.joint_gp_prior_grad`

    :py:meth:`fvgp.GP.gp_entropy`

    :py:meth:`fvgp.GP.gp_entropy_grad`

    :py:meth:`fvgp.GP.gp_kl_div`

    :py:meth:`fvgp.GP.gp_mutual_information`

    :py:meth:`fvgp.GP.gp_total_correlation`

    :py:meth:`fvgp.GP.gp_relative_information_entropy`

    :py:meth:`fvgp.GP.gp_relative_information_entropy_set`

    :py:meth:`fvgp.GP.posterior_probability`

    Methods for validation:

    :py:meth:`fvgp.GP.crps`

    :py:meth:`fvgp.GP.rmse`

    :py:meth:`fvgp.GP.make_2d_x_pred`

    :py:meth:`fvgp.GP.make_1d_x_pred`

    :py:meth:`fvgp.GP.log_likelihood`


    Parameters
    ----------
    x_data : np.ndarray or list, optional
        The input point positions. Shape (V x D), where D is the :py:attr:`fvgp.GP.index_set_dim`.
        For single-task GPs, the index set dimension = input space dimension.
        For multi-task GPs, the index set dimension = input space dimension + 1.
        If dealing with non-Euclidean inputs
        x_data should be a list, not a numpy array.
        If x_data is not provided here the GP will be initiated after `tell()`.
    y_data : np.ndarray, optional
        The values of the data points. Shape (V).
        If not provided here the GP will be initiated after `tell()`.
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used to initiate the GP.
        The default is an array of ones with the right length for the anisotropic Matern
        kernel with automatic relevance determination (ARD). If gp2Scale is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
    noise_variances : np.ndarray, optional
        An numpy array defining the uncertainties/noise in the
        `y_data` in form of a point-wise variance. Shape (V).
        Note: if no noise_variances are provided here, the noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to `abs(np.mean(y_data)) / 100.0`. If
        noise covariances are required (correlated noise), make use of the `kernel_function`.
        Only provide a noise function OR `noise_variances`, not both.
    compute_device : str, optional
        One of `cpu` or `gpu`, determines how linear algebra computations are executed. The default is `cpu`.
        For "gpu", pytorch has to be installed manually.
        If gp2Scale is enabled but no kernel is provided, the choice of the `compute_device`
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
        If `ram_economy` is True, the function's input is x1, x2, direction (int), and hyperparameters (numpy array).
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
        vector (1d np.ndarray) of length(x).
        The input `x` is a numpy array of shape (N x D). The hyperparameter array is the same
        that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
        If noise covariances are required (correlated noise), make use of the `kernel_function`.
    noise_function_grad : Callable, optional
        A function that evaluates the gradient of the `noise_function`
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x D) and
        hyperparameters (a 1d array of length D+1 for the default kernel).
        The return value is a 2d array of
        shape (len(hyperparameters) x N). If None is provided, either
        zeros are returned since the default noise function does not depend on
        hyperparameters, or, if `noise_function` is provided but no noise function,
        a finite-difference approximation will be used.
        The same rules regarding `ram_economy` as for the kernel definition apply here.
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
        an automatic determination of the mode.
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
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the log marginal likelihood is/are used for the training.
        If True, components of the derivative of the log marginal likelihood are
        calculated sequentially, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (and noise function) with
        respect to the hyperparameters (kernel_function_grad) is
        going to be provided, it has to be tailored: for `ram_economy=True` it should be
        of the form f(x, direction, hyperparameters)
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
    cost_function_parameters : object, optional
        This object is transmitted to the cost function;
        it can be of any type. The default is None.
    cost_update_function : Callable, optional
        If provided this function will be used when
        :py:meth:`update_cost_function` is called.
        The function `cost_update_function` accepts as
        input costs and a parameter
        object. The default is a no-op.
    logging : bool
        If true, logging is enabled. The default is False.
    args : any, optional
            Arguments will be transmitted to the acquisition function as part of the GPOptimizer
            object instance.

    Attributes
    ----------
    x_data : np.ndarray
        Datapoint positions
    y_data : np.ndarray
        Datapoint values
    noise_variances : np.ndarray
        Datapoint observation variances
    hyperparameters : np.ndarray
        Current hyperparameters in use.
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
            cost_function_parameters=None,
            cost_update_function=None,
            logging=False,
            args=None
    ):
        self.cost_function = cost_function
        self.cost_function_parameters = cost_function_parameters
        self.cost_update_function = cost_update_function
        self.hyperparameters = init_hyperparameters
        self.compute_device = compute_device
        self.kernel_function = kernel_function
        self.kernel_function_grad = kernel_function_grad
        self.noise_function = noise_function
        self.noise_function_grad = noise_function_grad
        self.prior_mean_function = prior_mean_function
        self.prior_mean_function_grad = prior_mean_function_grad
        self.gp2Scale = gp2Scale
        self.gp2Scale_dask_client = gp2Scale_dask_client
        self.gp2Scale_batch_size = gp2Scale_batch_size
        self.gp2Scale_linalg_mode = gp2Scale_linalg_mode
        self.calc_inv = calc_inv
        self.ram_economy = ram_economy
        self.args = args

        self.gp = False
        if x_data is not None and y_data is not None:
            self._initializeGP(x_data, y_data, noise_variances=noise_variances)
        else:
            warnings.warn("GP has not been initialized. Call tell() before using any class methods.")

        if logging is True:
            logger.enable("gpcam")
            logger.enable("fvgp")

    def _initializeGP(self, x_data, y_data, noise_variances=None):
        """
        Function to initialize a GP object.
        If data is prided at initialization this function is NOT needed.
        It has the same parameters as the initialization of the class (except cost related).
        """
        self.gp = True
        self.x_out = None
        super().__init__(
            x_data,
            y_data,
            init_hyperparameters=self.hyperparameters,
            noise_variances=noise_variances,
            compute_device=self.compute_device,
            kernel_function=self.kernel_function,
            kernel_function_grad=self.kernel_function_grad,
            noise_function=self.noise_function,
            noise_function_grad=self.noise_function_grad,
            prior_mean_function=self.prior_mean_function,
            prior_mean_function_grad=self.prior_mean_function_grad,
            gp2Scale=self.gp2Scale,
            gp2Scale_dask_client=self.gp2Scale_dask_client,
            gp2Scale_batch_size=self.gp2Scale_batch_size,
            gp2Scale_linalg_mode=self.gp2Scale_linalg_mode,
            calc_inv=self.calc_inv,
            ram_economy=self.ram_economy,
        )
        self.input_space_dimension = self.index_set_dim

    #########################################################################################
    def get_data(self):
        """
        Function that provides access to the class attributes.

        Return
        ------
        dictionary of class attributes : dict
        """

        return {
            "input dim": self.data.index_set_dim,
            "x data": self.x_data,
            "y data": self.y_data,
            "measurement variances": self.likelihood.V,
            "hyperparameters": self.hyperparameters,
            "cost function parameters": self.cost_function_parameters,
            "cost function": self.cost_function}

    def evaluate_acquisition_function(self, x, acquisition_function="variance", origin=None, args=None):
        """
        Function to evaluate the acquisition function.

        Parameters
        ----------
        x : np.ndarray or list
            Point positions at which the acquisition function is evaluated. np.ndarray of shape (N x D) or list.
        acquisition_function : Callable, optional
            Acquisition function to execute. Callable with inputs (x,gpcam.GPOptimizer),
            where x is a V x D array of input x position. The return value is a 1d array of length V.
            The default is `variance`.
        origin : np.ndarray, optional
            If a cost function is provided this 1d numpy array of length D is used as the origin of motion.
        args : any, optional
            Arguments that will be passed to the acquisition function as part of the gp_optimizer object.
            CAUTION: this will overwrite the args set at initialization.

        Return
        ------
        Evaluation : np.ndarray
            The acquisition function evaluations at all x.
        """
        if args is not None: self.args = args
        if self.cost_function and origin is None:
            warnings.warn("Warning: For the cost function to be active, an origin has to be provided.")
        if origin is not None and self.cost_function is None:
            warnings.warn("Warning: An origin is given but no cost function is defined. Cost function ignored")
        try:
            res = sm.evaluate_acquisition_function(
                x, gpo=self, acquisition_function=acquisition_function, origin=origin, dim=self.input_space_dimension,
                cost_function=self.cost_function, cost_function_parameters=self.cost_function_parameters)

            return -res
        except Exception as ex:
            logger.error(ex)
            logger.error("Evaluating the acquisition function was not successful.")
            raise Exception("Evaluating the acquisition function was not successful.", ex)

    def tell(self, x, y, noise_variances=None, append=True, gp_rank_n_update=None):
        """
        This function can tell() the gp_optimizer class
        the data that was collected. The data will instantly be used to update the GP data.

        Parameters
        ----------
        x : np.ndarray or list
            Point positions to be communicated to the Gaussian Process; either a np.ndarray of shape (U x D)
            or a list.
        y : np.ndarray
            Point values of shape (U) to be communicated to the Gaussian Process.
        noise_variances : np.ndarray, optional
            Point value variances of shape(U) to be communicated to the Gaussian Process.
            If not provided, the GP will 1% of the y values as variances.
        append : bool, optional
            Indication whether to append to or overwrite the existing dataset. Default = True.
            In the default case, data will be appended.
        gp_rank_n_update : bool , optional
            Indicates whether the GP marginal should be rank-n updated or recomputed. The default
            is `gp_rank_n_update=append`, meaning if data is only appended, the rank_n_update will
            be performed.
        """

        if gp_rank_n_update is None: gp_rank_n_update = append
        if self.gp:
            self.update_gp_data(x, y, noise_variances_new=noise_variances,
                                append=append, gp_rank_n_update=gp_rank_n_update)
        else:
            self._initializeGP(x, y, noise_variances=noise_variances)

    ##############################################################
    def train(
            self,
            objective_function=None,
            objective_function_gradient=None,
            objective_function_hessian=None,
            hyperparameter_bounds=None,
            init_hyperparameters=None,
            method="global",
            pop_size=20,
            tolerance=0.0001,
            max_iter=120,
            local_optimizer="L-BFGS-B",
            global_optimizer="genetic",
            constraints=(),
            dask_client=None,
            info=False):

        """
        This function finds the maximum of the log marginal likelihood and therefore trains the GP (synchronously).
        This can be done on a remote cluster/computer by specifying the method to be `hgdl` and
        providing a dask client. However, in that case `fvgp.GP.train_async()` is preferred.
        The GP prior will automatically be updated with the new hyperparameters after the training.


        Parameters
        ----------
        hyperparameter_bounds : np.ndarray, optional
            A 2d numpy array of shape (N x 2), where N is the number of hyperparameters.
            The default means inferring the bounds from the communicated dataset.
            This only works for the default kernel.
        objective_function : callable, optional
            The function that will be MINIMIZED for training the GP. The form of the function is f(hyperparameters=hps)
            and returns a scalar. This function can be used to train via non-standard user-defined objectives.
            The default is the negative log marginal likelihood.
        objective_function_gradient : callable, optional
            The gradient of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a vector of len(hps). This function can be used to train
            via non-standard user-defined objectives.
            The default is the gradient of the negative log marginal likelihood.
        objective_function_hessian : callable, optional
            The Hessian of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a matrix of shape(len(hps),len(hps)). This function can be used to train
            via non-standard user-defined objectives.
            The default is the Hessian of the negative log marginal likelihood.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers.
            The default is a random draw from a uniform distribution within the `hyperparameter_bounds`.
            If you want the training to start at previously trained hyperparameters you have to specify that
            explicitly.
        method : str or Callable, optional
            The method used to train the hyperparameters.
            The options are `global`, `local`, `hgdl`, `mcmc`, and a callable.
            The callable gets a `gp.GP` instance and has to return a 1d np.ndarray of hyperparameters.
            The default is `global` (scipy's differential evolution).
            If method = `mcmc`,
            the attribute `fvgp.GP.mcmc_info` is updated and contains convergence and distribution information.
        pop_size : int, optional
            A number of individuals used for any optimizer with a global component. Default = 20.
        tolerance : float, optional
            Used as termination criterion for local optimizers. Default = 0.0001.
        max_iter : int, optional
            Maximum number of iterations for global and local optimizers. Default = 120.
        local_optimizer : str, optional
            Defining the local optimizer. Default = `L-BFGS-B`, most `scipy.optimize.minimize`
            functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to `method = hgdl`. Default = `genetic`
        constraints : tuple of object instances, optional
            Equality and inequality constraints for the optimization.
            If the optimizer is `hgdl` see `hgdl.readthedocs.io`.
            If the optimizer is a `scipy` optimizer, see the scipy documentation.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if `hgdl` is used. If None is provided, a new
            `dask.distributed.Client` instance is constructed.
        info : bool, optional
            Provides a way how to access information reports during training of the GP. The default is False.
            If other information is needed please utilize `logger` as described in the online
            documentation (separately for HGDL and fvgp if needed).


        Return
        ------
        optimized hyperparameters (only fyi, gp is already updated) : np.ndarray
        """
        self.hyperparameters = super().train(
            objective_function=objective_function,
            objective_function_gradient=objective_function_gradient,
            objective_function_hessian=objective_function_hessian,
            hyperparameter_bounds=hyperparameter_bounds,
            init_hyperparameters=init_hyperparameters,
            method=method,
            pop_size=pop_size,
            tolerance=tolerance,
            max_iter=max_iter,
            local_optimizer=local_optimizer,
            global_optimizer=global_optimizer,
            constraints=constraints,
            dask_client=dask_client,
            info=info)
        return self.hyperparameters

    ##############################################################
    def train_async(
            self,
            objective_function=None,
            objective_function_gradient=None,
            objective_function_hessian=None,
            hyperparameter_bounds=None,
            init_hyperparameters=None,
            max_iter=10000,
            local_optimizer="L-BFGS-B",
            global_optimizer="genetic",
            constraints=(),
            dask_client=None):

        """
        This function finds the maximum of the log marginal likelihood and therefore trains the GP asynchronously.
        This can be done on a remote cluster/computer by
        providing a dask client. This function submits the training and returns
        an object which can be given to `fvgp.GP.update_hyperparameters()`,
        which will automatically update the GP with the new hyperparameters.


        Parameters
        ----------
        hyperparameter_bounds : np.ndarray, optional
            A 2d numpy array of shape (N x 2), where N is the number of hyperparameters.
            The default means inferring the bounds from the communicated dataset.
            This only works for the default kernel.
        objective_function : callable, optional
            The function that will be MINIMIZED for training the GP. The form of the function is f(hyperparameters=hps)
            and returns a scalar. This function can be used to train via non-standard user-defined objectives.
            The default is the negative log marginal likelihood.
        objective_function_gradient : callable, optional
            The gradient of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a vector of len(hps). This function can be used to train
            via non-standard user-defined objectives.
            The default is the gradient of the negative log marginal likelihood.
        objective_function_hessian : callable, optional
            The Hessian of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a matrix of shape(len(hps),len(hps)). This function can be used to train
            via non-standard user-defined objectives.
            The default is the Hessian of the negative log marginal likelihood.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers.
            The default is a random draw from a uniform distribution within the `hyperparameter_bounds`.
            If you want the training to start at previously trained hyperparameters you have to specify that
            explicitly.
        max_iter : int, optional
            Maximum number of iterations for global and local optimizers. Default = 120.
        local_optimizer : str, optional
            Defining the local optimizer. Default = `L-BFGS-B`, most `scipy.optimize.minimize`
            functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to `method = hgdl`. Default = `genetic`
        constraints : tuple of object instances, optional
            Equality and inequality constraints for the optimization.
            If the optimizer is `hgdl` see `hgdl.readthedocs.io`.
            If the optimizer is a `scipy` optimizer, see the scipy documentation.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if `hgdl` is used. If None is provided, a new
            `dask.distributed.Client` instance is constructed.


        Return
        ------
        Optimization object that can be given to `fvgp.GP.update_hyperparameters()` to update the GP : object instance
        """

        opt_obj = super().train_async(
            objective_function=objective_function,
            objective_function_gradient=objective_function_gradient,
            objective_function_hessian=objective_function_hessian,
            hyperparameter_bounds=hyperparameter_bounds,
            init_hyperparameters=init_hyperparameters,
            max_iter=max_iter,
            local_optimizer=local_optimizer,
            global_optimizer=global_optimizer,
            constraints=constraints,
            dask_client=dask_client
        )
        return opt_obj

    ##############################################################
    def stop_training(self, opt_obj):
        """
        Function to stop an asynchronous `hgdl` training.
        This leaves the :py:class:`distributed.client.Client` alive.

        Parameters
        ----------
        opt_obj : object instance
            Object created by :py:meth:`train_async()`.
        """
        super().stop_training(opt_obj)

    def kill_training(self, opt_obj):
        """
        Function to kill an asynchronous training. This shuts down the associated :py:class:`distributed.client.Client`.

        Parameters
        ----------
        opt_obj : object instance
            Object created by :py:meth:`train_async()`.
        """
        super().kill_training(opt_obj)

    ##############################################################
    def update_hyperparameters(self, opt_obj):
        """
        Function to update the Gaussian Process hyperparameters if an asynchronous training is running.

        Parameters
        ----------
        opt_obj : object instance
            Object created by :py:meth:`train_async()`.

        Return
        ------
        hyperparameters : np.ndarray
        """

        hps = super().update_hyperparameters(opt_obj)
        self.hyperparameters = hps
        return hps

    def set_hyperparameters(self, hps):
        """
        Function to set hyperparameters.

        Parameters
        ----------
        hps : np.ndarray
            A 1-d numpy array of hyperparameters.
        """
        self.hyperparameters = hps
        super().set_hyperparameters(hps)

    ##############################################################
    def ask(self,
            input_set,
            position=None,
            n=1,
            acquisition_function="variance",
            method="global",
            pop_size=20,
            max_iter=20,
            tol=1e-6,
            constraints=(),
            x0=None,
            vectorized=True,
            info=False,
            args=None,
            dask_client=None):
        """
        Given that the acquisition device is at `position`, this function `ask()`s for
        `n` new optimal points within a given `input_set` (given as bounds or candidates)  
        using the optimization setup `method`,
        `acquisition_function_pop_size`, `max_iter`, `tol`, `constraints`, and `x0`.
        This function can also choose the best candidate of a candidate set for Bayesian optimization
        on non-Euclidean input spaces.

        Parameters
        ----------
        input_set : np.ndarray or list
            Either a numpy array of floats of shape D x 2 describing the
            search space or a set of candidates in the form of a list. If a candidate list
            is provided, `ask()` will evaluate the acquisition function on each
            element and return a sorted array of length `n`.
            This is usually desirable for non-Euclidean inputs but can be used either way. If candidates are
            Euclidean, they should be provided as a list of 1d np.ndarray`s.
        position : np.ndarray, optional
            Current position in the input space. If a cost function is 
            provided this position will be taken into account
            to guarantee a cost-efficient new suggestion. The default is None.
        n : int, optional
            The algorithm will try to return n suggestions for
            new measurements. This is either done by method = 'hgdl', or otherwise
            by maximizing the collective information gain (default).
        acquisition_function : Callable, optional
            The acquisition function accepts as input a numpy array
            of size V x D (such that V is the number of input
            points, and D is the parameter space dimensionality) and
            a :py:class:`GPOptimizer` object. The return value is 1d array
            of length V providing 'scores' for each position,
            such that the highest scored point will be measured next.
            Built-in functions can be used by one of the following keys: 
            `ucb`,`lcb`,`maximum`,
            `minimum`, `variance`,`expected_improvement`,
            `relative information entropy`,`relative information entropy set`,
            `probability of improvement`, `gradient`,`total correlation`,`target probability`.
            If None, the default function `variance`, meaning
            :py:meth:`fvgp.GP.posterior_covariance` with variance_only = True will be used.
            The acquisition function can be callable and of the form my_func(x,gpcam.GPOptimizer)
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
            args = {'a': lower bound, 'b': upper bound} to be defined.
        method : str, optional
            A string defining the method used to find the maximum
            of the acquisition function. Choose from `global`,
            `local`, `hgdl`. HGDL is an in-house hybrid optimizer
            that is comfortable on HPC hardware.
            The default is `global`.
        pop_size : int, optional
            An integer defining the number of individuals if `global` 
            is chosen as method. The default is 20. For
            :py:mod:`hgdl` this will be overwritten
            by the `dask_client` definition.
        max_iter : int, optional
            This number defined the number of iterations 
            before the optimizer is terminated. The default is 20.
        tol : float, optional
            Termination criterion for the local optimizer.
            The default is 1e-6.
        x0 : np.ndarray, optional
            A set of points as numpy array of shape N x D,
            used as starting location(s) for the optimization
            algorithms. The default is None.
        vectorized : bool, optional
            If your acquisition function is vectorized to return the 
            solution to an array of inquiries as an array,
            this option makes the optimization faster if method = 'global'
            is used. The default is True but will be set to
            False if method is not global.
        info : bool, optional
            Print optimization information. The default is False.
        constraints : tuple of object instances, optional
            scipy constraints instances, depending on the used optimizer.
        args : any, optional
            Arguments that will be passed to the acquisition function as part of the gp_optimizer object.
            CAUTION: this will overwrite the args set at initialization.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed
            `acquisition_function` optimization. If None is provided,
            a new :py:class:`distributed.client.Client` instance is constructed for hgdl.

        Return
        ------
        Solution : {'x': np.array(maxima), "f_a(x)" : np.array(func_evals), "opt_obj" : opt_obj}
            Found maxima of the acquisition function, the associated function values and optimization object
            that, only in case of `method` = `hgdl` can be queried for solutions.
        """

        logger.debug("ask() initiated with hyperparameters: {}", self.hyperparameters)
        logger.debug("optimization method: {}", method)
        logger.debug("input set:\n{}", input_set)
        logger.debug("acq func: {}", acquisition_function)
        if args is not None: self.args = args

        assert isinstance(vectorized, bool)

        #check for bounds or candidate set
        #check that bounds, if they exist, are 2d
        if isinstance(input_set, np.ndarray) and np.ndim(input_set) != 2:
            raise Exception("The input_set parameter has to be a 2d np.ndarray or a list.")
        #for user-defined acquisition functions, use "hgdl" if n>1
        if n > 1 and callable(acquisition_function):
            method = "hgdl"
        #make sure that if there are bounds and n>1, method has to be global, if not hgdl
        if isinstance(input_set, np.ndarray) and n > 1 and method != "hgdl":
            vectorized = False
            method = "global"
            new_optimization_bounds = np.vstack([input_set for i in range(n)])
            input_set = new_optimization_bounds
            if acquisition_function != "total correlation" and acquisition_function != "relative information entropy":
                acquisition_function = "total correlation"
                warnings.warn("You specified n>1 and method != 'hgdl' in ask(). The acquisition function "
                              "has therefore been changed to 'total correlation'")

        if acquisition_function == "total correlation" or acquisition_function == "relative information entropy":
            vectorized = False

        maxima, func_evals, opt_obj = sm.find_acquisition_function_maxima(
            self,
            acquisition_function,
            origin=position,
            number_of_maxima_sought=n,
            input_set=input_set,
            input_set_dim=self.input_space_dimension,
            optimization_method=method,
            optimization_pop_size=pop_size,
            optimization_max_iter=max_iter,
            optimization_tol=tol,
            cost_function=self.cost_function,
            cost_function_parameters=self.cost_function_parameters,
            optimization_x0=x0,
            constraints=constraints,
            vectorized=vectorized,
            info=info,
            dask_client=dask_client)
        if n > 1: return {'x': maxima.reshape(-1, self.input_space_dimension), "f_a(x)": func_evals,
                          "opt_obj": opt_obj}
        return {'x': np.array(maxima), "f_a(x)": np.array(func_evals), "opt_obj": opt_obj}

    def optimize(self,
                 *,
                 func,
                 search_space,
                 hyperparameter_bounds=None,
                 train_at=(10, 20, 50, 100, 200),
                 x0=None,
                 acq_func='lcb',
                 max_iter=100,
                 callback=None,
                 break_condition=None,
                 ask_max_iter=20,
                 ask_pop_size=20,
                 method="global",
                 training_method="global",
                 training_max_iter=20
                 ):
        """
        This function is a light-weight optimization loop, using `tell()` and `ask()` repeatedly
        to optimize a given function, while retraining the GP regularly. For advanced customizations please
        use those three methods in a customized loop.

        Parameters
        ----------
        func : Callable
            The function to be optimized. The callable should be of the form def f(x),
            where `x` is an element of your search space. The return is a tuple of scalars (a,b) where
            `a` is the function evaluation scalar and `b` is the noise variance scalar.
        search_space : np.ndarray or list
            In the Euclidean case this should be a 2d np.ndarray of bounds in each direction of the input space.
            In the non-Euclidean case, this should be a list of all candidates.
        hyperparameter_bounds : np.ndarray
            Bound of the hyperparameters for the training. The default will only work for the default kernel.
            Otherwise, please specify bounds for your hyperparameters.
        train_at : tuple, optional
            The list should contain the integers that indicate the data lengths
            at which to train the GP. The default = [10,20,50,100,200].
        x0 : np.ndarray, optional
            Starting positions. Corresponding to the search space either elements of
            the candidate set in form of a list or elements of the Euclidean search space in the form of a
            2d np.ndarray.
        acq_func : Callable, optional
            Default lower-confidence bound(lcb) which means minimizing the `func`.
            The acquisition function should be formulated such that MAXIMIZING it will lead to the
            desired optimization (minimization or maximization) of `func`.
            For example `lcb` (the default) MAXIMIZES -(mean - 3.0 * standard dev) which is equivalent to minimizing
            (mean - 3.0 * standard dev) which leads to finding a minimum.
        max_iter : int, optional
            The maximum number of iterations. Default=10,000,000.
        callback : Callable, optional
            Function to be called in every iteration. Form: f(x_data, y_data)
        break_condition : Callable, optional
            Callable f(x_data, y_data) that should return `True` if run is complete, otherwise `False`.
        ask_max_iter : int, optional
            Default=20. Maximum number of iteration of the global and hybrid optimizer within `ask()`.
        ask_pop_size : int, optional
            Default=20. Population size of the global and hybrid optimizer.
        method : str, optional
            Default=`global`. Method of optimization of the acquisition function.
            One of `global, `local`, `hybrid`.
        training_method : str, optional
            Default=`global`. See :py:meth:`gpcam.GPOptimizer.train`
        training_max_iter : int, optional
            Default=20. See :py:meth:`gpcam.GPOptimizer.train`


        Return
        ------
        Full traces of function values `f(x)` and arguments `x` and the last entry: dict
            Form {'trace f(x)': self.y_data,
                  'trace x': self.x_data,
                  'f(x)': self.y_data[-1],
                  'x': self.x_data[-1]}
        """
        assert callable(func)
        assert isinstance(search_space, np.ndarray) or isinstance(search_space, list)
        assert isinstance(max_iter, int)

        if not x0:
            if isinstance(search_space, list): x0 = random.sample(search_space, 10)
            if isinstance(search_space, np.ndarray): x0 = np.random.uniform(low=search_space[:, 0],
                                                                            high=search_space[:, 1],
                                                                            size=(10, len(search_space)))
        result = list(map(func, x0))
        y, v = map(np.hstack, zip(*result))
        self.tell(x=x0, y=y, noise_variances=v, append=False)
        self.train(hyperparameter_bounds=hyperparameter_bounds)
        for i in range(max_iter):
            logger.debug("iteration {}", i)
            x_new = self.ask(search_space,
                             acquisition_function=acq_func,
                             method=method,
                             pop_size=ask_pop_size,
                             max_iter=ask_max_iter,
                             tol=1e-6,
                             constraints=(),
                             info=False)["x"]
            y_new, v_new = func(x_new)
            if callable(callback): callback(self.x_data, self.y_data)
            self.tell(x=x_new, y=y_new, noise_variances=v_new, append=True)
            if len(self.x_data) in train_at: self.train(hyperparameter_bounds=hyperparameter_bounds,
                                                        method=training_method, max_iter=training_max_iter)
            if callable(break_condition):
                if break_condition(self.x_data, self.y_data): break
        return {'trace f(x)': self.y_data,
                'trace x': self.x_data,
                'f(x)': self.y_data[-1],
                'x': self.x_data[-1]}

    def __getstate__(self):
        if self.gp2Scale_dask_client:
            raise logger.warn('GPOptimizer cannot be pickled with a dask client in gp2Scale_dask_client.')

        state = dict(x_data=self.x_data,
                     y_data=self.y_data,
                     hyperparameters=self.hyperparameters,
                     noise_variances=self.likelihood.V,
                     compute_device=self.compute_device,
                     kernel_function=self.kernel_function,
                     kernel_function_grad=self.kernel_function_grad,
                     noise_function=self.noise_function,
                     noise_function_grad=self.noise_function_grad,
                     prior_mean_function=self.prior_mean_function,
                     prior_mean_function_grad=self.prior_mean_function_grad,
                     gp2Scale=self.gp2Scale,
                     gp2Scale_batch_size=self.gp2Scale_batch_size,
                     gp2Scale_linalg_mode=self.gp2Scale_linalg_mode,
                     calc_inv=self.calc_inv,
                     ram_economy=self.ram_economy,
                     cost_function=self.cost_function,
                     cost_function_parameters=self.cost_function_parameters,
                     cost_update_function=self.cost_update_function,
                     )
        return state

    def __setstate__(self, state):
        x_data = state.pop('x_data')
        y_data = state.pop('y_data')
        noise_variances = state.pop('noise_variances')
        state['gp2Scale_dask_client'] = None
        # hyperparameters = state.pop('init_hyperparameters')
        self.__dict__.update(state)
        if x_data is not None and y_data is not None:
            self._initializeGP(x_data, y_data, noise_variances=noise_variances)


######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
#class fvGPOptimizer(fvGP, GPOptimizerBase):
class fvGPOptimizer(fvGP):
    """
    This class is an optimization extension of the :doc:`fvgp <fvgp:index>`
    package for multi-task (vector-valued) Gaussian Processes.
    Gaussian Processes can be initialized, trained, and conditioned; also
    the posterior can be evaluated and used via acquisition functions,
    and plugged into optimizers to find their maxima. This class inherits all methods from
    the :py:class:`fvgp.fvGP` class. Check :doc:`fvgp.readthedocs.io <fvgp:index>` for a full list of capabilities.

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

    This has to be understood and taken into account when customizing :doc:`fvgp <fvgp:index>` for multi-task
    use. The examples will provide deeper insights.


    All posterior evaluation functions are inherited from :py:class:`fvgp.GP`.
    Check there for a full list of capabilities. Methods for validation are also  available.
    Inherited include, but are not limited to:

    :py:meth:`fvgp.GP.posterior_mean`

    :py:meth:`fvgp.GP.posterior_covariance`

    :py:meth:`fvgp.GP.posterior_mean_grad`

    :py:meth:`fvgp.GP.posterior_covariance_grad`

    :py:meth:`fvgp.GP.joint_gp_prior`

    :py:meth:`fvgp.GP.joint_gp_prior_grad`

    :py:meth:`fvgp.GP.gp_entropy`

    :py:meth:`fvgp.GP.gp_entropy_grad`

    :py:meth:`fvgp.GP.gp_kl_div`

    :py:meth:`fvgp.GP.gp_mutual_information`

    :py:meth:`fvgp.GP.gp_total_correlation`

    :py:meth:`fvgp.GP.gp_relative_information_entropy`

    :py:meth:`fvgp.GP.gp_relative_information_entropy_set`

    :py:meth:`fvgp.GP.posterior_probability`


    Other methods:

    :py:meth:`fvgp.GP.crps`

    :py:meth:`fvgp.GP.rmse`

    :py:meth:`fvgp.GP.make_2d_x_pred`

    :py:meth:`fvgp.GP.make_1d_x_pred`

    :py:meth:`fvgp.GP.log_likelihood`


    Parameters
    ----------
    x_data : np.ndarray or list, optional
        The input point positions. Shape (V x Di), where Di is the :py:attr:`fvgp.fvGP.input_space_dim`.
        For multi-task GPs, the index set dimension = input space dimension + 1.
        If dealing with non-Euclidean inputs
        x_data should be a list, not a numpy array.
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
        For "gpu", pytorch has to be installed manually.
        If gp2Scale is enabled but no kernel is provided, the choice of the `compute_device`
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
        A function that calculates the derivative of the `kernel_function` with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input `x1` (a N1 x Di + 1 array of positions),
        `x2` (a N2 x Di + 1 array of positions) and
        `hyperparameters` (a 1d array of length Di+2 for the default kernel).
        The default is a finite difference calculation.
        If `ram_economy` is True, the function's input is x1, x2,
        direction (int), and hyperparameters (numpy array).
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
        vector (1d np.ndarray) of length(x).
        The input `x` is a numpy array of shape (N x Di+1). The hyperparameter array is the same
        that is communicated to mean and kernel functions.
        Only provide a noise function OR a noise variance vector, not both.
        If noise covariances are required (correlated noise), make use of the `kernel_function`.
    noise_function_grad : Callable, optional
        A function that evaluates the gradient of the `noise_function`
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x Di+1) and
        hyperparameters (a 1d array of length D+1 for the default kernel).
        The return value is a 2d array of
        shape (len(hyperparameters) x N). If None is provided, either
        zeros are returned since the default noise function does not depend on
        hyperparameters, or, if `noise_function` is provided but no noise function,
        a finite-difference approximation will be used.
        The same rules regarding `ram_economy` as for the kernel definition apply here.
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
        an automatic determination of the mode.
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
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the log marginal likelihood is/are used for the training.
        If True, components of the derivative of the log marginal likelihood are
        calculated sequentially, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (and noise function) with
        respect to the hyperparameters (kernel_function_grad) is
        going to be provided, it has to be tailored: for `ram_economy=True` it should be
        of the form f(x, direction, hyperparameters)
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
    cost_function_parameters : object, optional
        This object is transmitted to the cost function;
        it can be of any type. The default is None.
    cost_update_function : Callable, optional
        If provided this function will be used when
        :py:meth:`update_cost_function` is called.
        The function `cost_update_function` accepts as
        input costs and a parameter
        object. The default is a no-op.
    logging : bool
        If true, logging is enabled. The default is False.
    args : any, optional
            Arguments will be transmitted to the acquisition function as part of the GPOptimizer
            object instance.


    Attributes
    ----------
    x_data : np.ndarray
        Datapoint positions
    y_data : np.ndarray
        Datapoint values
    fvgp_x_data : np.ndarray
        Datapoint positions as seen by fvgp
    fvgp_y_data : np.ndarray
        Datapoint values as seen by fvgp
    noise_variances : np.ndarray
        Datapoint observation variances.
    hyperparameters : np.ndarray
        Current hyperparameters in use.
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
            cost_function_parameters=None,
            cost_update_function=None,
            logging=False,
            args=None
    ):
        self.cost_function = cost_function
        self.cost_function_parameters = cost_function_parameters
        self.cost_update_function = cost_update_function
        self.hyperparameters = init_hyperparameters
        self.compute_device = compute_device
        self.kernel_function = kernel_function
        self.kernel_function_grad = kernel_function_grad
        self.noise_function = noise_function
        self.noise_function_grad = noise_function_grad
        self.prior_mean_function = prior_mean_function
        self.prior_mean_function_grad = prior_mean_function_grad
        self.gp2Scale = gp2Scale
        self.gp2Scale_dask_client = gp2Scale_dask_client
        self.gp2Scale_batch_size = gp2Scale_batch_size
        self.gp2Scale_linalg_mode = gp2Scale_linalg_mode
        self.calc_inv = calc_inv
        self.ram_economy = ram_economy
        self.args = args

        if logging is True:
            logger.enable("gpcam")
            logger.enable("fvgp")

        self.gp = False
        if x_data is not None and y_data is not None:
            self._initializefvGP(x_data, y_data, noise_variances=noise_variances)
        else:
            warnings.warn("GP has not been initialized. Call tell() before using any class methods.")

    def _initializefvGP(self, x_data, y_data, noise_variances=None):
        """
        Function to initialize a GP object.
        If data is prided at initialization this function is NOT needed.
        It has the same parameters as the initialization of the class.
        """
        self.x_out = np.arange(y_data.shape[1])
        self.gp = True
        super().__init__(
            x_data,
            y_data,
            init_hyperparameters=self.hyperparameters,
            noise_variances=noise_variances,
            compute_device=self.compute_device,
            kernel_function=self.kernel_function,
            kernel_function_grad=self.kernel_function_grad,
            noise_function=self.noise_function,
            noise_function_grad=self.noise_function_grad,
            prior_mean_function=self.prior_mean_function,
            prior_mean_function_grad=self.prior_mean_function_grad,
            gp2Scale=self.gp2Scale,
            gp2Scale_dask_client=self.gp2Scale_dask_client,
            gp2Scale_batch_size=self.gp2Scale_batch_size,
            gp2Scale_linalg_mode=self.gp2Scale_linalg_mode,
            calc_inv=self.calc_inv,
            ram_economy=self.ram_economy,
        )
        self.input_space_dimension = self.input_space_dim

    ############################################################################
    def get_data(self):
        """
        Function that provides access to the class attributes.

        Return
        ------
        dictionary of class attributes : dict
        """

        return {
            "input dim": self.input_space_dimension,
            "x data": self.x_data,
            "y data": self.y_data,
            "measurement variances": self.likelihood.V,
            "hyperparameters": self.hyperparameters,
            "cost function parameters": self.cost_function_parameters,
            "cost function": self.cost_function}

    ############################################################################
    def evaluate_acquisition_function(self, x, x_out=None, acquisition_function="variance", origin=None, args=None):
        """
        Function to evaluate the acquisition function.

        Parameters
        ----------
        x : np.ndarray or list
            Point positions at which the acquisition function is evaluated. np.ndarray of shape (N x D) or list.
        x_out : np.ndarray, optional
            Point positions in the output space.
        acquisition_function : Callable, optional
            Acquisition function to execute. Callable with inputs (x,gpcam.gp_optimizer.GPOptimizer),
            where x is a V x D array of input x_data. The return value is a 1d array of length V.
            The default is `variance`.
        origin : np.ndarray, optional
            If a cost function is provided this 1d numpy array of length D is used as the origin of motion.
        args : any, optional
            Arguments that will be passed to the acquisition function as part of the gp_optimizer object.
            CAUTION: this will overwrite the args set at initialization.

        Return
        ------
        The acquisition function evaluations at all points x : np.ndarray
        """
        if x_out is None: x_out = self.x_out
        if args is not None: self.args = args

        if self.cost_function and origin is None:
            warnings.warn("Warning: For the cost function to be active, an origin has to be provided.")
        x = np.array(x)
        try:
            res = sm.evaluate_acquisition_function(
                x, gpo=self, acquisition_function=acquisition_function, origin=origin, dim=self.input_space_dimension,
                cost_function=self.cost_function, cost_function_parameters=self.cost_function_parameters,
                x_out=x_out)
            return -res
        except Exception as ex:
            logger.error(ex)
            logger.error("Evaluating the acquisition function was not successful.")
            raise Exception("Evaluating the acquisition function was not successful.", ex)

    ############################################################################
    def tell(self, x, y, noise_variances=None, append=True, gp_rank_n_update=None):
        """
        This function can tell() the gp_optimizer class
        the data that was collected. The data will instantly be used to update the GP data.

        Parameters
        ----------
        x : np.ndarray or list
            Point positions to be communicated to the Gaussian Process; either a np.ndarray of shape (U x D)
            or a list.
        y : np.ndarray
            The values of the data points. Shape (V,No).
            It is possible that not every entry in `x_new`
            has all corresponding tasks available. In that case `y_new` may contain np.nan entries.
        noise_variances : np.ndarray, optional
            An numpy array or list defining the uncertainties/noise in the
            `y_data` in form of a point-wise variance. Shape (V, No).
            If `y_data` has np.nan entries, the corresponding
            `noise_variances` have to be np.nan.
            Note: if no noise_variances are provided here, the noise_function
            callable will be used; if the callable is not provided, the noise variances
            will be set to `abs(np.mean(y_data)) / 100.0`. If
            noise covariances are required (correlated noise), make use of the noise_function.
            Only provide a noise function OR `noise_variances`, not both.
        append : bool, optional
            Indication whether to append to or overwrite the existing dataset. Default = True.
            In the default case, data will be appended.
        gp_rank_n_update : bool , optional
            Indicates whether the GP marginal should be rank-n updated or recomputed. The default
            is `gp_rank_n_update=append`, meaning if data is only appended, the rank_n_update will
            be performed.
        """

        if gp_rank_n_update is None: gp_rank_n_update = append
        if self.gp:
            self.update_gp_data(x, y, noise_variances_new=noise_variances,
                                append=append, gp_rank_n_update=gp_rank_n_update)
        else:
            self._initializefvGP(x, y, noise_variances=noise_variances)

    ##############################################################
    def train(self,
              objective_function=None,
              objective_function_gradient=None,
              objective_function_hessian=None,
              hyperparameter_bounds=None,
              init_hyperparameters=None,
              method="global",
              pop_size=20,
              tolerance=0.0001,
              max_iter=120,
              local_optimizer="L-BFGS-B",
              global_optimizer="genetic",
              constraints=(),
              dask_client=None):

        """
        This function finds the maximum of the log marginal likelihood and therefore trains the GP (synchronously).
        This can be done on a remote cluster/computer by specifying the method to be `hgdl` and
        providing a dask client. However, in that case `fvgp.GP.train_async()` is preferred.
        The GP prior will automatically be updated with the new hyperparameters after the training.


        Parameters
        ----------
        hyperparameter_bounds : np.ndarray, optional
            A 2d numpy array of shape (N x 2), where N is the number of hyperparameters.
            The default means inferring the bounds from the communicated dataset.
            This only works for the default kernel.
        objective_function : callable, optional
            The function that will be MINIMIZED for training the GP. The form of the function is f(hyperparameters=hps)
            and returns a scalar. This function can be used to train via non-standard user-defined objectives.
            The default is the negative log marginal likelihood.
        objective_function_gradient : callable, optional
            The gradient of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a vector of len(hps). This function can be used to train
            via non-standard user-defined objectives.
            The default is the gradient of the negative log marginal likelihood.
        objective_function_hessian : callable, optional
            The Hessian of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a matrix of shape(len(hps),len(hps)). This function can be used to train
            via non-standard user-defined objectives.
            The default is the Hessian of the negative log marginal likelihood.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers.
            The default is a random draw from a uniform distribution within the `hyperparameter_bounds`.
            If you want the training to start at previously trained hyperparameters you have to specify that
            explicitly.
        method : str or Callable, optional
            The method used to train the hyperparameters.
            The options are `global`, `local`, `hgdl`, `mcmc`, and a callable.
            The callable gets a `gp.GP` instance and has to return a 1d np.ndarray of hyperparameters.
            The default is `global` (scipy's differential evolution).
            If method = `mcmc`,
            the attribute `fvgp.GP.mcmc_info` is updated and contains convergence and distribution information.
        pop_size : int, optional
            A number of individuals used for any optimizer with a global component. Default = 20.
        tolerance : float, optional
            Used as termination criterion for local optimizers. Default = 0.0001.
        max_iter : int, optional
            Maximum number of iterations for global and local optimizers. Default = 120.
        local_optimizer : str, optional
            Defining the local optimizer. Default = `L-BFGS-B`, most `scipy.optimize.minimize`
            functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to `method = hgdl`. Default = `genetic`
        constraints : tuple of object instances, optional
            Equality and inequality constraints for the optimization.
            If the optimizer is `hgdl` see `hgdl.readthedocs.io`.
            If the optimizer is a `scipy` optimizer, see the scipy documentation.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if `hgdl` is used. If None is provided, a new
            `dask.distributed.Client` instance is constructed.


        Return
        ------
        optimized hyperparameters (only fyi, gp is already updated) : np.ndarray
        """
        self.hyperparameters = super().train(
            objective_function=objective_function,
            objective_function_gradient=objective_function_gradient,
            objective_function_hessian=objective_function_hessian,
            hyperparameter_bounds=hyperparameter_bounds,
            init_hyperparameters=init_hyperparameters,
            method=method,
            pop_size=pop_size,
            tolerance=tolerance,
            max_iter=max_iter,
            local_optimizer=local_optimizer,
            global_optimizer=global_optimizer,
            constraints=constraints,
            dask_client=dask_client)
        return self.hyperparameters

    ##############################################################
    def train_async(self,
                    objective_function=None,
                    objective_function_gradient=None,
                    objective_function_hessian=None,
                    hyperparameter_bounds=None,
                    init_hyperparameters=None,
                    max_iter=10000,
                    local_optimizer="L-BFGS-B",
                    global_optimizer="genetic",
                    constraints=(),
                    dask_client=None
                    ):

        """
        This function finds the maximum of the log marginal likelihood and therefore trains the GP asynchronously.
        This can be done on a remote cluster/computer by
        providing a dask client. This function submits the training and returns
        an object which can be given to `fvgp.GP.update_hyperparameters()`,
        which will automatically update the GP with the new hyperparameters.


        Parameters
        ----------
        hyperparameter_bounds : np.ndarray, optional
            A 2d numpy array of shape (N x 2), where N is the number of hyperparameters.
            The default means inferring the bounds from the communicated dataset.
            This only works for the default kernel.
        objective_function : callable, optional
            The function that will be MINIMIZED for training the GP. The form of the function is f(hyperparameters=hps)
            and returns a scalar. This function can be used to train via non-standard user-defined objectives.
            The default is the negative log marginal likelihood.
        objective_function_gradient : callable, optional
            The gradient of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a vector of len(hps). This function can be used to train
            via non-standard user-defined objectives.
            The default is the gradient of the negative log marginal likelihood.
        objective_function_hessian : callable, optional
            The Hessian of the function that will be MINIMIZED for training the GP.
            The form of the function is f(hyperparameters=hps)
            and returns a matrix of shape(len(hps),len(hps)). This function can be used to train
            via non-standard user-defined objectives.
            The default is the Hessian of the negative log marginal likelihood.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers.
            The default is a random draw from a uniform distribution within the `hyperparameter_bounds`.
            If you want the training to start at previously trained hyperparameters you have to specify that
            explicitly.
        max_iter : int, optional
            Maximum number of iterations for global and local optimizers. Default = 120.
        local_optimizer : str, optional
            Defining the local optimizer. Default = `L-BFGS-B`, most `scipy.optimize.minimize`
            functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to `method = hgdl`. Default = `genetic`
        constraints : tuple of object instances, optional
            Equality and inequality constraints for the optimization.
            If the optimizer is `hgdl` see `hgdl.readthedocs.io`.
            If the optimizer is a `scipy` optimizer, see the scipy documentation.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if `hgdl` is used. If None is provided, a new
            `dask.distributed.Client` instance is constructed.


        Return
        ------
        Optimization object that can be given to `fvgp.GP.update_hyperparameters()` to update the GP : object instance
        """

        opt_obj = super().train_async(
            objective_function=objective_function,
            objective_function_gradient=objective_function_gradient,
            objective_function_hessian=objective_function_hessian,
            hyperparameter_bounds=hyperparameter_bounds,
            init_hyperparameters=init_hyperparameters,
            max_iter=max_iter,
            local_optimizer=local_optimizer,
            global_optimizer=global_optimizer,
            constraints=constraints,
            dask_client=dask_client
        )
        return opt_obj

    ##############################################################
    def stop_training(self, opt_obj):
        """
        Function to stop an asynchronous `hgdl` training.
        This leaves the :py:class:`distributed.client.Client` alive.

        Parameters
        ----------
        opt_obj : object instance
            Object created by :py:meth:`train_async()`.
        """
        super().stop_training(opt_obj)

    def kill_training(self, opt_obj):
        """
        Function to kill an asynchronous training. This shuts down the associated :py:class:`distributed.client.Client`.

        Parameters
        ----------
        opt_obj : object instance
            Object created by :py:meth:`train_async()`.
        """
        super().kill_training(opt_obj)

    ##############################################################
    def update_hyperparameters(self, opt_obj):
        """
        Function to update the Gaussian Process hyperparameters if an asynchronous training is running.

        Parameters
        ----------
        opt_obj : object instance
            Object created by :py:meth:`train_async()`.

        Return
        ------
        hyperparameters : np.ndarray
        """

        hps = super().update_hyperparameters(opt_obj)
        self.hyperparameters = hps
        return hps

    def set_hyperparameters(self, hps):
        """
        Function to set hyperparameters.

        Parameters
        ----------
        hps : np.ndarray
            A 1-d numpy array of hyperparameters.
        """
        self.hyperparameters = hps
        super().set_hyperparameters(hps)

    def ask(self,
            input_set,
            x_out=None,
            acquisition_function='variance',
            position=None,
            n=1,
            method="global",
            pop_size=20,
            max_iter=20,
            tol=1e-6,
            constraints=(),
            x0=None,
            vectorized=True,
            info=False,
            args=None,
            dask_client=None):

        """
        Given that the acquisition device is at `position`, this function `ask()`s for
        `n` new optimal points within a given `input_set` (given as bounds or candidates)
        using the optimization setup `method`,
        `acquisition_function_pop_size`, `max_iter`, `tol`, `constraints`, and `x0`.
        This function can also choose the best candidate of a candidate set for Bayesian optimization
        on non-Euclidean input spaces.

        Parameters
        ----------
        input_set : np.ndarray or list
            Either a numpy array of floats of shape D x 2 describing the
            search space or a set of candidates in the form of a list. If a candidate list
            is provided, `ask()` will evaluate the acquisition function on each
            element and return a sorted array of length `n`.
            This is usually desirable for non-Euclidean inputs but can be used either way. If candidates are
            Euclidean, they should be provided as a list of 1d np.ndarray`s.
        x_out : np.ndarray, optional
            The position indicating where in the output space the acquisition function should be evaluated.
            This array is of shape (No).
        position : np.ndarray, optional
            Current position in the input space. If a cost function is
            provided this position will be taken into account
            to guarantee a cost-efficient new suggestion. The default is None.
        n : int, optional
            The algorithm will try to return n suggestions for
            new measurements. This is either done by method = 'hgdl', or otherwise
            by maximizing the collective information gain (default).
        acquisition_function : Callable, optional
            The acquisition function accepts as input a numpy array
            of size V x D (such that V is the number of input
            points, and D is the parameter space dimensionality) and
            a :py:class:`GPOptimizer` object. The return value is 1d array
            of length V providing 'scores' for each position,
            such that the highest scored point will be measured next.
            Built-in functions can be used by one of the following keys:
            `variance`, `relative information entropy`,
            `relative information entropy set`, `total correlation`, `ucb`, `lcb`,
            and `expected improvement`.
            See :py:meth:`gpcam.GPOptimizer.ask` for a short explanation of these functions.
            In the multi-task case, it is highly recommended to
            deploy a user-defined acquisition function due to the intricate relationship
            of posterior distributions at different points in the output space.
            If None, the default function `variance`, meaning
            :py:meth:`fvgp.GP.posterior_covariance` with variance_only = True will be used.
            The acquisition function can be a callable function of the form my_func(x,gpcam.GPOptimizer)
            which will be maximized (!!!), so make sure desirable new measurement points
            will be located at maxima.
        method : str, optional
            A string defining the method used to find the maximum
            of the acquisition function. Choose from `global`,
            `local`, `hgdl`. HGDL is an in-house hybrid optimizer
            that is comfortable on HPC hardware.
            The default is `global`.
        pop_size : int, optional
            An integer defining the number of individuals if `global`
            is chosen as method. The default is 20. For
            :py:mod:`hgdl` this will be overwritten
            by the `dask_client` definition.
        max_iter : int, optional
            This number defined the number of iterations
            before the optimizer is terminated. The default is 20.
        tol : float, optional
            Termination criterion for the local optimizer.
            The default is 1e-6.
        x0 : np.ndarray, optional
            A set of points as numpy array of shape N x D,
            used as starting location(s) for the optimization
            algorithms. The default is None.
        vectorized : bool, optional
            If your acquisition function is vectorized to return the
            solution to an array of inquiries as an array,
            this option makes the optimization faster if method = 'global'
            is used. The default is True but will be set to
            False if method is not global.
        info : bool, optional
            Print optimization information. The default is False.
        constraints : tuple of object instances, optional
            scipy constraints instances, depending on the used optimizer.
        args : any, optional
            Arguments that will be passed to the acquisition function as part of the gp_optimizer object.
            CAUTION: This will overwrite the args set at initialization.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed
            `acquisition_function` optimization. If None is provided,
            a new :py:class:`distributed.client.Client` instance is constructed for hgdl.

        Return
        ------
        Solution : {'x': np.array(maxima), "f_a(x)" : np.array(func_evals), "opt_obj" : opt_obj}
            Found maxima of the acquisition function, the associated function values and optimization object
            that, only in case of `method` = `hgdl` can be queried for solutions.
        """

        logger.debug("ask() initiated with hyperparameters: {}", self.hyperparameters)
        logger.debug("optimization method: {}", method)
        logger.debug("bounds:\n{}", input_set)
        logger.debug("acq func: {}", acquisition_function)
        self.args = args
        if x_out is None: x_out = self.x_out

        assert isinstance(vectorized, bool)
        if isinstance(input_set, np.ndarray) and np.ndim(input_set) != 2:
            raise Exception("The input_set parameter has to be a 2d np.ndarray or a list.")
        #for user-defined acquisition functions, use "hgdl" if n>1
        if n > 1 and callable(acquisition_function):
            method = "hgdl"
        if isinstance(input_set, np.ndarray) and n > 1 and method != "hgdl":
            vectorized = False
            method = "global"
            new_optimization_bounds = np.vstack([input_set for i in range(n)])
            input_set = new_optimization_bounds
            if acquisition_function != "total correlation" and acquisition_function != "relative information entropy":
                acquisition_function = "total correlation"
                warnings.warn("You specified n>1 and method != 'hgdl' in ask(). The acquisition function "
                              "has therefore been changed to 'total correlation'.")
        if acquisition_function == "total correlation" or acquisition_function == "relative information entropy":
            vectorized = False

        maxima, func_evals, opt_obj = sm.find_acquisition_function_maxima(
            self,
            acquisition_function,
            origin=position, number_of_maxima_sought=n,
            input_set=input_set,
            input_set_dim=self.input_space_dimension,
            optimization_method=method,
            optimization_pop_size=pop_size,
            optimization_max_iter=max_iter,
            optimization_tol=tol,
            cost_function=self.cost_function,
            cost_function_parameters=self.cost_function_parameters,
            optimization_x0=x0,
            constraints=constraints,
            vectorized=vectorized,
            x_out=x_out,
            info=info,
            dask_client=dask_client)
        if n > 1: return {'x': maxima.reshape(-1, self.input_space_dimension), "f_a(x)": func_evals,
                          "opt_obj": opt_obj}
        return {'x': np.array(maxima), "f_a(x)": np.array(func_evals), "opt_obj": opt_obj}

    def optimize(self,
                 *,
                 func,
                 search_space,
                 x_out=None,
                 hyperparameter_bounds=None,
                 train_at=(10, 20, 50, 100, 200),
                 x0=None,
                 acq_func='lcb',
                 max_iter=100,
                 callback=None,
                 break_condition=None,
                 ask_max_iter=20,
                 ask_pop_size=20,
                 method="global",
                 training_method="global",
                 training_max_iter=20
                 ):
        """
        This function is a light-weight optimization loop, using `tell()` and `ask()` repeatedly
        to optimize a given function, while retraining the GP regularly. For advanced customizations please
        use those three methods in a customized loop.

        Parameters
        ----------
        func : Callable
            The function to be optimized. The callable should be of the form def f(x),
            where `x` is an element of your search space. The return is a tuple of scalars (a,b) where
            `a` is a vector of function evaluations and `b` is a vector of noise variances.
        search_space : np.ndarray or list
            In the Euclidean case this should be a 2d np.ndarray of bounds in each direction of the input space.
            In the non-Euclidean case, this should be a list of all candidates.
        x_out : np.ndarray, optional
            The position indicating where in the output space the acquisition function should be evaluated.
            This array is of shape (No).
        hyperparameter_bounds : np.ndarray
            Bound of the hyperparameters for the training. The default will only work for the default kernel.
            Otherwise, please specify bounds for your hyperparameters.
        train_at : tuple, optional
            The list should contain the integers that indicate the data lengths
            at which to train the GP. The default = [10,20,50,100,200].
        x0 : np.ndarray, optional
            Starting positions. Corresponding to the search space either elements of
            the candidate set in form of a list or elements of the Euclidean search space in the form of a
            2d np.ndarray.
        acq_func : Callable, optional
            Default lower-confidence bound(lcb) which means minimizing the `func`.
            The acquisition function should be formulated such that MAXIMIZING it will lead to the
            desired optimization (minimization or maximization) of `func`.
            For example `lcb` (the default) MAXIMIZES -(mean - 3.0 * standard dev) which is equivalent to minimizing
            (mean - 3.0 * standard dev) which leads to finding a minimum.
        max_iter : int, optional
            The maximum number of iterations. Default=10,000,000.
        callback : Callable, optional
            Function to be called in every iteration. Form: f(x_data, y_data)
        break_condition : Callable, optional
            Callable f(x_data, y_data) that should return `True` if run is complete, otherwise `False`.
        ask_max_iter : int, optional
            Default=20. Maximum number of iteration of the global and hybrid optimizer within `ask()`.
        ask_pop_size : int, optional
            Default=20. Population size of the global and hybrid optimizer.
        method : str, optional
            Default=`global`. Method of optimization of the acquisition function.
            One of `global, `local`, `hybrid`.
        training_method : str, optional
            Default=`global`. See :py:meth:`gpcam.GPOptimizer.train`
        training_max_iter : int, optional
            Default=20. See :py:meth:`gpcam.GPOptimizer.train`


        Return
        ------
        Full traces of function values `f(x)` and arguments `x` and the last entry: dict
            Form {'trace f(x)': self.y_data,
                  'trace x': self.x_data,
                  'f(x)': self.y_data[-1],
                  'x': self.x_data[-1]}
        """
        assert callable(func)
        assert isinstance(search_space, np.ndarray) or isinstance(search_space, list)
        assert isinstance(max_iter, int)
        if x_out is None: x_out = self.x_out

        if not x0:
            if isinstance(search_space, list): x0 = random.sample(search_space, 10)
            if isinstance(search_space, np.ndarray): x0 = np.random.uniform(low=search_space[:, 0],
                                                                            high=search_space[:, 1],
                                                                            size=(10, len(search_space)))
        result = list(map(func, x0))
        y = np.asarray(list(map(np.hstack, zip(*result)))).reshape(-1, len(x_out))[0:len(result)]
        v = np.asarray(list(map(np.hstack, zip(*result)))).reshape(-1, len(x_out))[len(result):]
        self.tell(x=x0, y=y, noise_variances=v, append=False)
        self.train(hyperparameter_bounds=hyperparameter_bounds)
        for i in range(max_iter):
            logger.debug("iteration {}", i)
            x_new = self.ask(search_space,
                             x_out,
                             acquisition_function=acq_func,
                             method=method,
                             pop_size=ask_pop_size,
                             max_iter=ask_max_iter,
                             tol=1e-6,
                             constraints=(),
                             info=False)["x"]
            y_new, v_new = func(x_new)
            if callable(callback): callback(self.x_data, self.y_data)
            self.tell(x=x_new, y=y_new, noise_variances=v_new, append=True)
            if len(self.x_data) in train_at: self.train(hyperparameter_bounds=hyperparameter_bounds,
                                                        method=training_method, max_iter=training_max_iter)
            if callable(break_condition):
                if break_condition(self.x_data, self.y_data): break
        return {'trace f(x)': self.y_data,
                'trace x': self.x_data,
                'f(x)': self.y_data[-1],
                'x': self.x_data[-1]}

    def __getstate__(self):
        if self.gp2Scale_dask_client:
            raise logger.warn('GPOptimizer cannot be pickled with a dask client in gp2Scale_dask_client.')

        state = dict(x_data=self.x_data,
                     y_data=self.y_data,
                     hyperparameters=self.hyperparameters,
                     noise_variances=self.likelihood.V,
                     compute_device=self.compute_device,
                     kernel_function=self.kernel_function,
                     kernel_function_grad=self.kernel_function_grad,
                     noise_function=self.noise_function,
                     noise_function_grad=self.noise_function_grad,
                     prior_mean_function=self.prior_mean_function,
                     prior_mean_function_grad=self.prior_mean_function_grad,
                     gp2Scale=self.gp2Scale,
                     gp2Scale_batch_size=self.gp2Scale_batch_size,
                     gp2Scale_linalg_mode=self.gp2Scale_linalg_mode,
                     calc_inv=self.calc_inv,
                     ram_economy=self.ram_economy,
                     cost_function=self.cost_function,
                     cost_function_parameters=self.cost_function_parameters,
                     cost_update_function=self.cost_update_function,
                     )
        return state

    def __setstate__(self, state):
        x_data = state.pop('x_data')
        y_data = state.pop('y_data')
        noise_variances = state.pop('noise_variances')
        state['gp2Scale_dask_client'] = None
        # hyperparameters = state.pop('init_hyperparameters')
        self.__dict__.update(state)
        if x_data is not None and y_data is not None:
            self._initializeGP(x_data, y_data, noise_variances=noise_variances)
