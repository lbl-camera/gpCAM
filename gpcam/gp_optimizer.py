#!/usr/bin/env python
import numpy as np
from loguru import logger
from fvgp import fvGP
from fvgp import GP
from . import surrogate_model as sm
import warnings
import sys


# TODO (for gpCAM)
#   should the gpoptimizer class have an "optimize" method?
#   add a test that shows online mode up to 10000 points, for this the inv_update rounding error can't destroy us.
#   in AE, can we ask including candidates?
#   how is append vs overwrite handled?
#   should we store the original non transformed data in fvgpOptimizer
#   needs `set_hyperparameters`


class GPOptimizer:
    """
    This class is an optimization wrapper around the :doc:`fvgp <fvgp:index>` package
    for single-task (scalar-valued) Gaussian Processes.
    Gaussian Processes can be initialized, trained, and conditioned; also
    the posterior can be evaluated and used via acquisition functions,
    and plugged into optimizers to find its optima. This class inherits many methods from
    the :py:class:`fvgp.GP` class.

    V ... number of input points

    D ... input space dimensionality

    N ... arbitrary integers (N1, N2,...)


    All posterior evaluation functions are inherited from :py:class:`fvgp.GP`.
    Check there for a full list of capabilities. Methods for validation are also  available.
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

    :py:meth:`fvgp.GP.gp_kl_div_grad`

    :py:meth:`fvgp.GP.gp_mutual_information`

    :py:meth:`fvgp.GP.gp_total_correlation`

    :py:meth:`fvgp.GP.gp_relative_information_entropy`

    :py:meth:`fvgp.GP.gp_relative_information_entropy_set`

    :py:meth:`fvgp.GP.posterior_probability`

    :py:meth:`fvgp.GP.posterior_probability_grad`

    Other methods:

    :py:meth:`fvgp.GP.crps`

    :py:meth:`fvgp.GP.rmse`

    :py:meth:`fvgp.GP.make_2d_x_pred`

    :py:meth:`fvgp.GP.make_1d_x_pred`

    :py:meth:`fvgp.GP.log_likelihood`

    :py:meth:`fvgp.GP.neg_log_likelihood`

    :py:meth:`fvgp.GP.neg_log_likelihood_gradient`

    :py:meth:`fvgp.GP.neg_log_likelihood_hessian`


    Parameters
    ----------
    x_data : np.ndarray or list, optional
        The input point positions. Shape (V x D), where D is the :py:attr:`fvgp.GP.index_set_dim`.
        For single-task GPs, the index set dimension = input space dimension.
        For multi-task GPs, the index set dimension = input space dimension + 1.
        If dealing with non-Euclidean inputs
        x_data should be a list, not a numpy array.
        If not provided here the GP will be initiated after `tell()`.
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
        Note: if no noise_variances are provided here, the gp_noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to `abs(np.mean(y_data)) / 100.0`. If
        noise covariances are required (correlated noise), make use of the gp_noise_function.
        Only provide a noise function OR `noise_variances`, not both.
    compute_device : str, optional
        One of `cpu` or `gpu`, determines how linear system solves are executed. The default is `cpu`.
        For "gpu", pytorch has to be installed manually.
        If gp2Scale is enabled but no kernel is provided, the choice of the compute_device
        becomes much more important. In that case, the default Wendland kernel will be computed on
        the cpu or the gpu which will significantly change the compute time depending on the compute
        architecture.
    gp_kernel_function : Callable, optional
        A symmetric positive definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters, obj).
        The input `x1` is a N1 x D array of positions, `x2` is a N2 x D
        array of positions, the hyperparameters argument
        is a 1d array of length D+1 for the default kernel and of a different
        length for user-defined kernels.
        `obj` is an `fvgp.GP` instance. The default is a stationary anisotropic kernel
        (`fvgp.GP.default_kernel`) which performs automatic relevance determination (ARD).
        The output is a matrix, an N1 x N2 numpy array.
    gp_kernel_function_grad : Callable, optional
        A function that calculates the derivative of the `gp_kernel_function` with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input `x1` (a N1 x D array of positions),
        `x2` (a N2 x D array of positions),
        `hyperparameters` (a 1d array of length D+1 for the default kernel), and a
        `fvgp.GP` instance. The default is a finite difference calculation.
        If `ram_economy` is True, the function's input is x1, x2, direction (int), hyperparameters (numpy array), and a
        `fvgp.GP` instance, and the output
        is a numpy array of shape (len(hps) x N).
        If `ram_economy` is `False`, the function's input is x1, x2, hyperparameters, and a
        `fvgp.GP` instance. The output is
        a numpy array of shape (len(hyperparameters) x N1 x N2). See `ram_economy`.
    gp_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x D), hyperparameters (a 1d array of length D+1 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used, which is the average of the `y_data`.
    gp_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the `gp_mean_function` at
        a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x D), hyperparameters
        (a 1d array of length D+1 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 2d array of
        shape (len(hyperparameters) x N1). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparameters,
        or a finite-difference approximation
        is used if `gp_mean_function` is provided.
    gp_noise_function : Callable, optional
        The noise function is a callable f(x,hyperparameters,obj) that returns a
        positive symmetric definite matrix of shape(len(x),len(x)).
        The input `x` is a numpy array of shape (N x D). The hyperparameter array is the same
        that is communicated to mean and kernel functions. The obj is a `fvgp.GP` instance.
        Only provide a noise function OR a noise variance vector, not both.
    gp_noise_function_grad : Callable, optional
        A function that evaluates the gradient of the `gp_noise_function`
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x D),
        hyperparameters (a 1d array of length D+1 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 3-D array of
        shape (len(hyperparameters) x N x N). If None is provided, either
        zeros are returned since the default noise function does not depend on
        hyperparameters, or, if `gp_noise_function` is provided but no gradient function,
        a finite-difference approximation will be used.
        The same rules regarding `ram_economy` as for the kernel definition apply here.
    gp2Scale: bool, optional
        Turns on gp2Scale. This will distribute the covariance computations across multiple workers.
        This is an advanced feature for HPC GPs up to 10
        million data points. If gp2Scale is used, the default kernel is an anisotropic
        Wendland kernel which is compactly supported. The noise function will have
        to return a `scipy.sparse` matrix instead of a numpy array. There are a few more
        things to consider (read on); this is an advanced option.
        If no kernel is provided, the `compute_device` option should be revisited.
        The kernel will use the specified device to compute covariances.
        The default is False.
    gp2Scale_dask_client : dask.distributed.Client, optional
        A dask client for gp2Scale.
        On HPC architecture, this client is provided by the job script. Please have a look at the examples.
        A local client is used as the default.
    gp2Scale_batch_size : int, optional
        Matrix batch size for distributed computing in gp2Scale. The default is 10000.
    calc_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance
        matrix after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster (5-10 times).
        For larger problems (>2000 data points), the use of inversion should be avoided due
        to computational instability and costs. The default is
        False. Note, the training will not the
        inverse for stability reasons. Storing the inverse is
        a good option when the dataset is not too large and the posterior covariance is heavily used.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the log marginal likelihood is/are used for the training.
        If True, components of the derivative of the log marginal likelihood are
        calculated sequentially, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (and noise function) with
        respect to the hyperparameters (gp_kernel_function_grad) is
        going to be provided, it has to be tailored: for `ram_economy=True` it should be
        of the form f(x1[, x2], direction, hyperparameters, obj)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If `ram_economy=False`, the function should be of the form f(x1[, x2,] hyperparameters, obj)
        and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters.
        CAUTION: This array will be stored and is very large.
    args : any, optional
        args will be a class attribute and therefore available to kernel, noise and prior mean functions.
    info : bool, optional
        Provides a way how to access various information reports. The default is False
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

    Attributes
    ----------
    x_data : np.ndarray
        Datapoint positions
    y_data : np.ndarray
        Datapoint values
    noise_variances : np.ndarray
        Datapoint observation (co)variances
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
            gp_kernel_function=None,
            gp_kernel_function_grad=None,
            gp_noise_function=None,
            gp_noise_function_grad=None,
            gp_mean_function=None,
            gp_mean_function_grad=None,
            gp2Scale=False,
            gp2Scale_dask_client=None,
            gp2Scale_batch_size=10000,
            calc_inv=True,
            ram_economy=False,
            args=None,
            info=False,
            cost_function=None,
            cost_function_parameters=None,
            cost_update_function=None
    ):
        self.gp = None
        if x_data is not None and y_data is not None:
            self._initializeGP(
                x_data,
                y_data,
                init_hyperparameters=init_hyperparameters,
                noise_variances=noise_variances,
                compute_device=compute_device,
                gp_kernel_function=gp_kernel_function,
                gp_kernel_function_grad=gp_kernel_function_grad,
                gp_noise_function=gp_noise_function,
                gp_noise_function_grad=gp_noise_function_grad,
                gp_mean_function=gp_mean_function,
                gp_mean_function_grad=gp_mean_function_grad,
                gp2Scale=gp2Scale,
                gp2Scale_dask_client=gp2Scale_dask_client,
                gp2Scale_batch_size=gp2Scale_batch_size,
                calc_inv=calc_inv,
                ram_economy=ram_economy,
                args=args,
                info=info)
        else:
            warnings.warn("You have not provided data yet. You have to manually initialize the GP "
                          "before any class method can be used!")
        self.cost_function = cost_function
        self.cost_function_parameters = cost_function_parameters
        self.cost_update_function = cost_update_function
        self.hyperparameters = init_hyperparameters
        if info:
            logger.enable('gpcam')
            logger.add(sys.stdout, filter="gpcam", level="INFO")
            logger.enable('fvgp')
            logger.add(sys.stdout, filter="fvgp", level="INFO")
            logger.enable('hgdl')
            logger.add(sys.stdout, filter="hgdl", level="INFO")
        else:
            logger.disable('gpcam')
            logger.disable('fvgp')
            logger.disable('hgdl')

    def __getattr__(self, attr):
        if not self.gp:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'. "
                                 f"You may need to initialize the GP.")
        elif self.gp and hasattr(self.gp, attr):
            return getattr(self.gp, attr)
        else:
            raise Exception(f"Attribute '{attr}' not available.")

    def _initializeGP(self,
                      x_data,
                      y_data,
                      init_hyperparameters=None,
                      noise_variances=None,
                      compute_device="cpu",
                      gp_kernel_function=None,
                      gp_kernel_function_grad=None,
                      gp_noise_function=None,
                      gp_noise_function_grad=None,
                      gp_mean_function=None,
                      gp_mean_function_grad=None,
                      gp2Scale=False,
                      gp2Scale_dask_client=None,
                      gp2Scale_batch_size=10000,
                      calc_inv=True,
                      ram_economy=False,
                      args=None,
                      info=False):
        """
        Function to initialize a GP object.
        If data is prided at initialization this function is NOT needed.
        It has the same parameters as the initialization of the class (except cost related).
        """
        self.gp = GP(
            x_data,
            y_data,
            init_hyperparameters=init_hyperparameters,
            noise_variances=noise_variances,
            compute_device=compute_device,
            gp_kernel_function=gp_kernel_function,
            gp_kernel_function_grad=gp_kernel_function_grad,
            gp_noise_function=gp_noise_function,
            gp_noise_function_grad=gp_noise_function_grad,
            gp_mean_function=gp_mean_function,
            gp_mean_function_grad=gp_mean_function_grad,
            gp2Scale=gp2Scale,
            gp2Scale_dask_client=gp2Scale_dask_client,
            gp2Scale_batch_size=gp2Scale_batch_size,
            calc_inv=calc_inv,
            ram_economy=ram_economy,
            args=args,
            info=info,
        )
        self.index_set_dim = self.gp.index_set_dim
        self.input_space_dim = self.gp.index_set_dim

    def get_data(self):
        """
        Function that provides access to the class attributes.

        Return
        ------
        dictionary of class attributes : dict
        """

        return {
            "input dim": self.prior.index_set_dim,
            "x data": self.x_data,
            "y data": self.y_data,
            "measurement variances": self.likelihood.V,
            "hyperparameters": self.hyperparameters,
            "cost function parameters": self.cost_function_parameters,
            "cost function": self.cost_function}

    def evaluate_acquisition_function(self, x, acquisition_function="variance", origin=None):
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

        Return
        ------
        Evaluation : np.ndarray
            The acquisition function evaluations at all x.
        """
        if self.cost_function and origin is None:
            warnings.warn("Warning: For the cost function to be active, an origin has to be provided.")
        if origin is not None and self.cost_function is None:
            warnings.warn("Warning: An origin is given but no cost function is defined. Cost function ignored")
        try:
            res = sm.evaluate_acquisition_function(
                x, gpo=self.gp, acquisition_function=acquisition_function, origin=origin, dim=self.input_space_dim,
                cost_function=self.cost_function, cost_function_parameters=self.cost_function_parameters)

            return -res
        except Exception as ex:
            logger.error(ex)
            logger.error("Evaluating the acquisition function was not successful.")
            raise Exception("Evaluating the acquisition function was not successful.", ex)

    def tell(self, x, y, noise_variances=None, append=False):
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
            The default is True. Indicates if existent data should be appended by or overwritten with the new data.
        """
        self.update_gp_data(x, y, noise_variances_new=noise_variances, append=append)

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
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is a random draw from a uniform distribution within the `hyperparameter_bounds`.
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
        self.hyperparameters = self.gp.train(
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
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is a random draw from a uniform distribution within the `hyperparameter_bounds`.
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

        opt_obj = self.gp.train_async(
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
        self.gp.stop_training(opt_obj)

    def kill_training(self, opt_obj):
        """
        Function to kill an asynchronous training. This shuts down the associated :py:class:`distributed.client.Client`.

        Parameters
        ----------
        opt_obj : object instance
            Object created by :py:meth:`train_async()`.
        """
        self.gp.kill_training(opt_obj)

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

        hps = self.gp.update_hyperparameters(opt_obj)
        self.hyperparameters = hps
        return hps

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
            GPOptimizer.args = {'a': lower bound, 'b': upper bound} to be defined.
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
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed
            `acquisition_function` optimization. If None is provided,
            a new :py:class:`distributed.client.Client` instance is constructed for hgdl.

        Return
        ------
        Solution : {'x': np.array(maxima), "f(x)" : np.array(func_evals), "opt_obj" : opt_obj}
            Found maxima of the acquisition function, the associated function values and optimization object
            that, only in case of `method` = `hgdl` can be queried for solutions.
        """

        logger.info("ask() initiated with hyperparameters: {}", self.hyperparameters)
        logger.info("optimization method: {}", method)
        logger.info("input set:\n{}", input_set)
        logger.info("acq func: {}", acquisition_function)

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
            new_optimization_bounds = np.row_stack([input_set for i in range(n)])
            input_set = new_optimization_bounds
            if acquisition_function != "total correlation" and acquisition_function != "relative information entropy":
                acquisition_function = "total correlation"
                warnings.warn("You specified n>1 and method != 'hgdl' in ask(). The acquisition function "
                              "has therefore been changed to 'total correlation'")

        if acquisition_function == "total correlation" or acquisition_function == "relative information entropy":
            vectorized = False
        if method != "global": vectorized = False

        maxima, func_evals, opt_obj = sm.find_acquisition_function_maxima(
            self.gp,
            acquisition_function,
            origin=position,
            number_of_maxima_sought=n,
            input_set=input_set,
            input_set_dim=self.input_space_dim,
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
        if n > 1: return {'x': maxima.reshape(-1, self.input_space_dim), "f(x)": func_evals,
                          "opt_obj": opt_obj}
        return {'x': np.array(maxima), "f(x)": np.array(func_evals), "opt_obj": opt_obj}

    ##############################################################
    def update_cost_function(self, measurement_costs):
        """
        This function updates the parameters for the user-defined cost function
        It essentially calls the user-given cost_update_function which
        should return the new parameters.

        Parameters
        ----------
        measurement_costs: object
            An arbitrary object that describes the costs when moving in the parameter space.
            It can be arbitrary because the cost function using the parameters and the cost_update_function
            updating the parameters are both user-defined and this object has to be in accordance
            with those definitions.
        """

        if self.cost_function_parameters is None: warnings.warn(
            "No cost_function_parameters specified. Cost update failed.")
        if callable(self.cost_update_function):
            self.cost_function_parameters = self.cost_update_function(measurement_costs, self.cost_function_parameters)
        else:
            warnings.warn("No cost_update_function available. Cost update failed.")


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
class fvGPOptimizer:
    """
    This class is an optimization wrapper around the :doc:`fvgp <fvgp:index>`
    package for multi-task (scalar-valued) Gaussian Processes.
    Gaussian Processes can be initialized, trained, and conditioned; also
    the posterior can be evaluated and used via acquisition functions,
    and plugged into optimizers to find its maxima. This class inherits many methods from
    the :py:class:`fvgp.GP` class. Check :doc:`fvgp.readthedocs.io <fvgp:index>` for a full list of capabilities.

    V ... number of input points

    Di... input space dimensionality

    No... number of outputs

    N ... arbitrary integers (N1, N2,...)


    The main logic of :doc:`fvgp <fvgp:index>` is that any multi-task GP is just a single-task GP
    over a Cartesian product space of input and output space, as long as the kernel
    is flexible enough, so prepare to work on your kernel. This is the best
    way to give the user optimal control and power. At various instances, for instances
    prior-mean function, noise function, and kernel function definitions, you will
    see that the input `x` is defined over this combined space.
    For example, if your input space is a Euclidean 2d space and your output
    is labelled [[0],[1]], the input to the mean, kernel, and noise function might be

    x =

    [[0.2, 0.3,0],[0.9,0.6,0],

    [0.2, 0.3,1],[0.9,0.6,1]]
    
    This has to be understood and taken into account when customizing :doc:`fvgp <fvgp:index>` for multi-task
    use. The examples will provide deeper insights.


    All posterior evaluation functions are inherited from :py:class:`fvgp.GP`.
    Check there for a full list of capabilities. Methods for validation are also  available.
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

    :py:meth:`fvgp.GP.gp_kl_div_grad`

    :py:meth:`fvgp.GP.gp_mutual_information`

    :py:meth:`fvgp.GP.gp_total_correlation`

    :py:meth:`fvgp.GP.gp_relative_information_entropy`

    :py:meth:`fvgp.GP.gp_relative_information_entropy_set`

    :py:meth:`fvgp.GP.posterior_probability`

    :py:meth:`fvgp.GP.posterior_probability_grad`

    Other methods:

    :py:meth:`fvgp.GP.crps`

    :py:meth:`fvgp.GP.rmse`

    :py:meth:`fvgp.GP.make_2d_x_pred`

    :py:meth:`fvgp.GP.make_1d_x_pred`

    :py:meth:`fvgp.GP.log_likelihood`

    :py:meth:`fvgp.GP.neg_log_likelihood`

    :py:meth:`fvgp.GP.neg_log_likelihood_gradient`

    :py:meth:`fvgp.GP.neg_log_likelihood_hessian`


    Parameters
    ----------
    x_data : np.ndarray or list
        The input point positions. Shape (V x Di), where Di is the :py:attr:`fvgp.fvGP.input_space_dim`.
        For multi-task GPs, the index set dimension = input space dimension + 1.
        If dealing with non-Euclidean inputs
        x_data should be a list, not a numpy array.
    y_data : np.ndarray
        The values of the data points. Shape (V,No).
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used to initiate the GP.
        The default is an array of ones with the right length for the anisotropic Matern
        kernel with automatic relevance determination (ARD). The task direction is
        simply considered a separate dimension. If sparse_node or gp2Scale is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
    output_positions : np.ndarray, optional
        A 2d numpy array of shape (V x output_number), so that for each measurement position, the outputs
        are clearly defined by their positions in the output space. The default is
        np.array([[0,1,2,3,...,output_number - 1],[0,1,2,3,...,output_number - 1],...]).
    noise_variances : np.ndarray, optional
        An numpy array defining the uncertainties/noise in the
        `y_data` in form of a point-wise variance. Shape (V, No).
        Note: if no noise_variances are provided here, the gp_noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to `abs(np.mean(y_data)) / 100.0`. If
        noise covariances are required (correlated noise), make use of the gp_noise_function.
        Only provide a noise function OR `noise_variances`, not both.
    compute_device : str, optional
        One of `cpu` or `gpu`, determines how linear system solves are executed. The default is `cpu`.
        For "gpu", pytorch has to be installed manually.
        If gp2Scale is enabled but no kernel is provided, the choice of the compute_device
        becomes much more important. In that case, the default Wendland kernel will be computed on
        the cpu or the gpu which will significantly change the compute time depending on the compute
        architecture.
    gp_kernel_function : Callable, optional
        A symmetric positive definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters, obj).
        The input `x1` a N1 x Di+1 array of positions, `x2` is a N2 x Di+1
        array of positions, the hyperparameters argument
        is a 1d array of length D+1 for the default kernel and of a different
        length for user-defined kernels.
        `obj` is an `fvgp.GP` instance.
        is a 1d array of length N depending on how many hyperparameters are initialized, and
        obj is an `fvgp.GP` instance. The default is a stationary anisotropic kernel
        (`fvgp.GP.default_kernel`) which performs automatic relevance determination (ARD). The task
        direction is simply considered an additional dimension. This kernel should only be used for tests and in the
        simplest of cases.
        The output is a matrix, an N1 x N2 numpy array.
    gp_kernel_function_grad : Callable, optional
        A function that calculates the derivative of the `gp_kernel_function` with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input `x1` (a N1 x Di + 1 array of positions),
        `x2` (a N2 x Di + 1 array of positions),
        `hyperparameters` (a 1d array of length Di+2 for the default kernel), and a
        `fvgp.GP` instance. The default is a finite difference calculation.
        If `ram_economy` is True, the function's input is x1, x2,
        direction (int), hyperparameters (numpy array), and a
        `fvgp.GP` instance, and the output
        is a numpy array of shape (len(hps) x N).
        If `ram_economy` is `False`, the function's input is x1, x2, hyperparameters, and a
        `fvgp.GP` instance. The output is
        a numpy array of shape (len(hyperparameters) x N1 x N2). See `ram_economy`.
    gp_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x Di+1), hyperparameters (a 1d array of length Di+2 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used, which is the average of the `y_data`.
    gp_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the `gp_mean_function` at
        a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x Di+1), hyperparameters
        (a 1d array of length Di+2 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 2d array of
        shape (len(hyperparameters) x N1). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparameters,
        or a finite-difference approximation
        is used if `gp_mean_function` is provided.
    gp_noise_function : Callable, optional
        The noise function is a callable f(x,hyperparameters,obj) that returns a
        positive symmetric definite matrix of shape(len(x),len(x)).
        The input `x` is a numpy array of shape (N x Di+1). The hyperparameter array is the same
        that is communicated to mean and kernel functions. The obj is a `fvgp.GP` instance.
        Only provide a noise function OR a noise variance vector, not both.
    gp_noise_function_grad : Callable, optional
        A function that evaluates the gradient of the `gp_noise_function`
        at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x Di+1),
        hyperparameters (a 1d array of length D+1 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 3-D array of
        shape (len(hyperparameters) x N x N). If None is provided, either
        zeros are returned since the default noise function does not depend on
        hyperparameters, or, if `gp_noise_function` is provided but no gradient function,
        a finite-difference approximation will be used.
        The same rules regarding `ram_economy` as for the kernel definition apply here.
    gp2Scale: bool, optional
        Turns on gp2Scale. This will distribute the covariance computations across multiple workers.
        This is an advanced feature for HPC GPs up to 10
        million data points. If gp2Scale is used, the default kernel is an anisotropic
        Wendland kernel which is compactly supported. The noise function will have
        to return a `scipy.sparse` matrix instead of a numpy array. There are a few more
        things to consider (read on); this is an advanced option.
        If no kernel is provided, the `compute_device` option should be revisited.
        The kernel will use the specified device to compute covariances.
        The default is False.
    gp2Scale_dask_client : dask.distributed.Client, optional
        A dask client for gp2Scale.
        On HPC architecture, this client is provided by the job script. Please have a look at the examples.
        A local client is used as the default.
    gp2Scale_batch_size : int, optional
        Matrix batch size for distributed computing in gp2Scale. The default is 10000.
    calc_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance
        matrix after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster (5-10 times).
        For larger problems (>2000 data points), the use of inversion should be avoided due
        to computational instability and costs. The default is
        False. Note, the training will not the
        inverse for stability reasons. Storing the inverse is
        a good option when the dataset is not too large and the posterior covariance is heavily used.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the log marginal likelihood is/are used for the training.
        If True, components of the derivative of the log marginal likelihood are
        calculated sequentially, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (and noise function) with
        respect to the hyperparameters (gp_kernel_function_grad) is
        going to be provided, it has to be tailored: for `ram_economy=True` it should be
        of the form f(x1[, x2], direction, hyperparameters, obj)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If `ram_economy=False`, the function should be of the form f(x1[, x2,] hyperparameters, obj)
        and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters.
        CAUTION: This array will be stored and is very large.
    args : any, optional
        args will be a class attribute and therefore available to kernel, noise and prior mean functions.
    info : bool, optional
        Provides a way how to access various information reports. The default is False.
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
        Datapoint observation (co)variances.
    hyperparameters : np.ndarray
        Current hyperparameters in use.

    """

    def __init__(
            self,
            x_data=None,
            y_data=None,
            init_hyperparameters=None,
            output_positions=None,
            noise_variances=None,
            compute_device="cpu",
            gp_kernel_function=None,
            gp_kernel_function_grad=None,
            gp_noise_function=None,
            gp_noise_function_grad=None,
            gp_mean_function=None,
            gp_mean_function_grad=None,
            gp2Scale=False,
            gp2Scale_dask_client=None,
            gp2Scale_batch_size=10000,
            calc_inv=True,
            ram_economy=False,
            args=None,
            info=False,
            cost_function=None,
            cost_function_parameters=None,
            cost_update_function=None,
    ):
        self.gp = None
        if x_data is not None and y_data is not None:
            self._initializefvGP(
                x_data,
                y_data,
                init_hyperparameters=init_hyperparameters,
                output_positions=output_positions,
                noise_variances=noise_variances,
                compute_device=compute_device,
                gp_kernel_function=gp_kernel_function,
                gp_kernel_function_grad=gp_kernel_function_grad,
                gp_noise_function=gp_noise_function,
                gp_noise_function_grad=gp_noise_function_grad,
                gp_mean_function=gp_mean_function,
                gp_mean_function_grad=gp_mean_function_grad,
                gp2Scale=gp2Scale,
                gp2Scale_dask_client=gp2Scale_dask_client,
                gp2Scale_batch_size=gp2Scale_batch_size,
                calc_inv=calc_inv,
                ram_economy=ram_economy,
                args=args,
                info=info)
        else:
            warnings.warn("You have not provided data yet. You have to manually initialize the GP "
                          "before any class method can be used!")

        self.cost_function = cost_function
        self.cost_function_parameters = cost_function_parameters
        self.cost_update_function = cost_update_function
        self.hyperparameters = init_hyperparameters

        if info:
            logger.enable('gpcam')
            logger.add(sys.stdout, filter="gpcam", level="INFO")
            logger.enable('fvgp')
            logger.add(sys.stdout, filter="fvgp", level="INFO")
            logger.enable('hgdl')
            logger.add(sys.stdout, filter="hgdl", level="INFO")
        else:
            logger.disable('gpcam')
            logger.disable('fvgp')
            logger.disable('hgdl')

    def __getattr__(self, attr):
        if not self.gp:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'. "
                                 f"You may need to initialize the GP.")
        elif self.gp and hasattr(self.gp, attr):
            return getattr(self.gp, attr)
        else:
            raise Exception(f"Attribute '{attr}' not available.")

    def _initializefvGP(self,
                        x_data,
                        y_data,
                        init_hyperparameters=None,
                        output_positions=None,
                        noise_variances=None,
                        compute_device="cpu",
                        gp_kernel_function=None,
                        gp_kernel_function_grad=None,
                        gp_noise_function=None,
                        gp_noise_function_grad=None,
                        gp_mean_function=None,
                        gp_mean_function_grad=None,
                        gp2Scale=False,
                        gp2Scale_dask_client=None,
                        gp2Scale_batch_size=10000,
                        calc_inv=True,
                        ram_economy=False,
                        args=None,
                        info=False):
        """
        Function to initialize a GP object.
        If data is prided at initialization this function is NOT needed.
        It has the same parameters as the initialization of the class.
        """
        self.gp = fvGP(
            x_data,
            y_data,
            init_hyperparameters=init_hyperparameters,
            output_positions=output_positions,
            noise_variances=noise_variances,
            compute_device=compute_device,
            gp_kernel_function=gp_kernel_function,
            gp_kernel_function_grad=gp_kernel_function_grad,
            gp_noise_function=gp_noise_function,
            gp_noise_function_grad=gp_noise_function_grad,
            gp_mean_function=gp_mean_function,
            gp_mean_function_grad=gp_mean_function_grad,
            gp2Scale=gp2Scale,
            gp2Scale_dask_client=gp2Scale_dask_client,
            gp2Scale_batch_size=gp2Scale_batch_size,
            calc_inv=calc_inv,
            ram_economy=ram_economy,
            args=args,
            info=info,
        )
        self.index_set_dim = self.gp.index_set_dim
        self.input_space_dim = self.gp.input_space_dim

    ############################################################################
    def get_data(self):
        """
        Function that provides access to the class attributes.

        Return
        ------
        dictionary of class attributes : dict
        """

        return {
            "input dim": self.input_space_dim,
            "x data": self.x_data,
            "y data": self.y_data,
            "measurement variances": self.likelihood.V,
            "hyperparameters": self.hyperparameters,
            "cost function parameters": self.cost_function_parameters,
            "cost function": self.cost_function}

    ############################################################################
    def evaluate_acquisition_function(self, x, x_out, acquisition_function="variance", origin=None):
        """
        Function to evaluate the acquisition function.

        Parameters
        ----------
        x : np.ndarray or list
            Point positions at which the acquisition function is evaluated. np.ndarray of shape (N x D) or list.
        x_out : np.ndarray
            Point positions in the output space.
        acquisition_function : Callable, optional
            Acquisition function to execute. Callable with inputs (x,gpcam.gp_optimizer.GPOptimizer),
            where x is a V x D array of input x_data. The return value is a 1d array of length V.
            The default is `variance`.
        origin : np.ndarray, optional
            If a cost function is provided this 1d numpy array of length D is used as the origin of motion.

        Return
        ------
        The acquisition function evaluations at all points x : np.ndarray
        """

        if self.cost_function and origin is None:
            warnings.warn("Warning: For the cost function to be active, an origin has to be provided.")
        x = np.array(x)
        cost_function = self.cost_function
        try:
            res = sm.evaluate_acquisition_function(
                x, gpo=self.gp, acquisition_function=acquisition_function, origin=origin, dim=self.input_space_dim,
                cost_function=self.cost_function, cost_function_parameters=self.cost_function_parameters,
                x_out=x_out)
            return -res
        except Exception as ex:
            logger.error(ex)
            logger.error("Evaluating the acquisition function was not successful.")
            raise Exception("Evaluating the acquisition function was not successful.", ex)

    ############################################################################
    def tell(self, x, y, noise_variances=None, output_positions=None, append=False):
        """
        This function can tell() the gp_optimizer class
        the data that was collected. The data will instantly be used to update the GP data.

        Parameters
        ----------
        x : np.ndarray or list
            Point positions to be communicated to the Gaussian Process; either a np.ndarray of shape (U x D)
            or a list.
        y : np.ndarray
            Point values (of shape U x 1 or U) to be communicated to the Gaussian Process.
        noise_variances : np.ndarray, optional
            Point value variances (of shape U x 1 or U) to be communicated 
            to the Gaussian Process.
            If not provided, the GP will 1% of the y values as variances.
        output_positions : np.ndarray, optional
            A 2d numpy array of shape (V x output_number), so that for each measurement position, the outputs
            are clearly defined by their positions in the output space. The default is
            np.array([[0,1,2,3,...,output_number - 1],[0,1,2,3,...,output_number - 1],...]).
        append : bool, optional
            The default is True. Indicates if existent data should be appended by or overwritten with the new data.
        """
        self.update_gp_data(x, y, noise_variances_new=noise_variances,
                            output_positions_new=output_positions, append=append)

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
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is a random draw from a uniform distribution within the `hyperparameter_bounds`.
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
        self.hyperparameters = self.gp.train(
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
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is a random draw from a uniform distribution within the `hyperparameter_bounds`.
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

        opt_obj = self.gp.train_async(
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
        self.gp.stop_training(opt_obj)

    def kill_training(self, opt_obj):
        """
        Function to kill an asynchronous training. This shuts down the associated :py:class:`distributed.client.Client`.

        Parameters
        ----------
        opt_obj : object instance
            Object created by :py:meth:`train_async()`.
        """
        self.gp.kill_training(opt_obj)

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

        hps = self.gp.update_hyperparameters(opt_obj)
        self.hyperparameters = hps
        return hps

    def ask(self,
            input_set,
            x_out,
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
            dask_client=None):

        """
        Given that the acquisition device is at `position`, this function `ask()`s for
        `n` new optimal points within a given `input_set` (given as bounds or candidates)
        using the optimization setup `method`,
        `acquisition_function_pop_size`, `max_iter`, `tol`, `constraints", and `x0`.
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
        x_out : np.ndarray
            The position indicating where in the output space the acquisition function should be evaluated.
            This array is of shape (No, Do).
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
            `relative information entropy set`, `total correlation`, `ucb`,
            and `expected improvement`.
            See `GPOptimizer.ask()` for a short explanation of these functions.
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
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed
            `acquisition_function` optimization. If None is provided,
            a new :py:class:`distributed.client.Client` instance is constructed for hgdl.

        Return
        ------
        Solution : {'x': np.array(maxima), "f(x)" : np.array(func_evals), "opt_obj" : opt_obj}
            Found maxima of the acquisition function, the associated function values and optimization object
            that, only in case of `method` = `hgdl` can be queried for solutions.
        """

        logger.info("ask() initiated with hyperparameters: {}", self.hyperparameters)
        logger.info("optimization method: {}", method)
        logger.info("bounds:\n{}", input_set)
        logger.info("acq func: {}", acquisition_function)

        assert isinstance(vectorized, bool)
        if isinstance(input_set, np.ndarray) and np.ndim(input_set) != 2:
            raise Exception("The input_set parameter has to be a 2d np.ndarray or a list.")
        #for user-defined acquisition functions, use "hgdl" if n>1
        if n > 1 and callable(acquisition_function):
            method = "hgdl"
        if isinstance(input_set, np.ndarray) and n > 1 and method != "hgdl":
            vectorized = False
            method = "global"
            new_optimization_bounds = np.row_stack([input_set for i in range(n)])
            input_set = new_optimization_bounds
            if acquisition_function != "total correlation" and acquisition_function != "relative information entropy":
                acquisition_function = "total correlation"
                warnings.warn("You specified n>1 and method != 'hgdl' in ask(). The acquisition function "
                              "has therefore been changed to 'total correlation'.")
        if acquisition_function == "total correlation" or acquisition_function == "relative information entropy":
            vectorized = False
        if method != "global": vectorized = False

        maxima, func_evals, opt_obj = sm.find_acquisition_function_maxima(
            self.gp,
            acquisition_function,
            origin=position, number_of_maxima_sought=n,
            input_set=input_set,
            input_set_dim=self.input_space_dim,
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
        if n > 1: return {'x': maxima.reshape(-1, self.gp.input_space_dim), "f(x)": func_evals,
                          "opt_obj": opt_obj}
        return {'x': np.array(maxima), "f(x)": np.array(func_evals), "opt_obj": opt_obj}

    ##############################################################
    def update_cost_function(self, measurement_costs):
        """
        This function updates the parameters for the user-defined cost function.
        It essentially calls the user-given cost_update_function which
        should return the new parameters.

        Parameters
        ----------
        measurement_costs: object
            An arbitrary object that describes the costs when moving in the parameter space.
            It can be arbitrary because the cost function using the parameters and the cost_update_function
            updating the parameters are both user-defined and this object has to be in
            accordance with those definitions.
        """

        if self.cost_function_parameters is None: warnings.warn("No cost_function_parameters "
                                                                "specified. Cost update failed.")
        if callable(self.cost_update_function):
            self.cost_function_parameters = self.cost_update_function(
                measurement_costs, self.cost_function_parameters)
        else:
            warnings.warn("No cost_update_function available. Cost update failed.")
