#!/usr/bin/env python

import numpy as np
from loguru import logger
from fvgp.fvgp import fvGP
from fvgp.gp import GP
from gpcam import surrogate_model as sm
import warnings



#TODO
#   check all docstrings for fvgp specific stuff (fvgp.GP...)
#   double check ask() in single-task
#   do ALL multi-task
#   remember the (fv)GPautonomous_experimenter

class GPOptimizer(GP):
    """
    This class is an optimization wrapper around the fvgp package for single-task (scalar-valued) Gaussian Processes.
    Gaussian Processes can be initialized, trained, and conditioned; also
    the posterior can be evaluated and used via acquisition functions,
    and plugged into optimizers to find its maxima. This class inherits many methods from
    the fvgp.GP class. Check fvgp.readthedocs.io for a full list of capabilities.

    V ... number of input points
    D ... input space dimensionality
    N ... arbitrary integers (N1, N2,...)


    Parameters
    ----------
    x_data : np.ndarray
        The input point positions. Shape (V x D), where D is the `input_space_dim`.
    y_data : np.ndarray
        The values of the data points. Shape (V,1) or (V).
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used by the GP initially.
        This class provides methods to train hyperparameters.
        The default is an array of ones, with a shape appropriate
        for the default kernel (D + 1), which is an anisotropic Matern
        kernel with automatic relevance determination (ARD).
    noise_variances : np.ndarray, optional
        An numpy array defining the uncertainties/noise in the data
        `y_data` in form of a point-wise variance. Shape (len(y_data), 1) or (len(y_data)).
        Note: if no noise_variances are provided here, the gp_noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to `abs(np.mean(y_data) / 100.0`. If
        noise covariances are required, also make use of the gp_noise_function.
    compute_device : str, optional
        One of "cpu" or "gpu", determines how linear system solves are run. The default is "cpu".
        For "gpu", pytoch has to be installed manually.
        If gp2Scale is enabled but no kernel is provided, the choice of the compute_device
        becomes much more important. In that case, the default kernel will be computed on
        the cpu or the gpu which will significantly change the compute time depending on the compute
        architecture.
    gp_kernel_function : Callable, optional
        A symmetric positive semi-definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters, obj).
        The input x1 is a N1 x D array of positions, x2 is a N2 x D
        array of positions, the hyperparameters argument
        is a 1d array of length D+1 for the default kernel and of a different
        user-defined length for other kernels
        obj is an `fvgp.GP` instance. The default is a stationary anisotropic kernel
        (`fvgp.GP.default_kernel`) which performs automatic relevance determination (ARD).
        The output is a covariance matrix, an N1 x N2 numpy array.
    gp_kernel_function_grad : Callable, optional
        A function that calculates the derivative of the ``gp_kernel_function'' with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input x1 (a N1 x D array of positions),
        x2 (a N2 x D array of positions),
        hyperparameters (a 1d array of length D+1 for the default kernel), and a
        `fvgp.GP` instance. The default is a finite difference calculation.
        If 'ram_economy' is True, the function's input is x1, x2, direction (int), hyperparameters (numpy array), and a
        `fvgp.GP` instance, and the output
        is a numpy array of shape (len(hps) x N).
        If 'ram economy' is False,the function's input is x1, x2, hyperparameters, and a
        `fvgp.GP` instance. The output is
        a numpy array of shape (len(hyperparameters) x N1 x N2). See 'ram_economy'.
    gp_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x D), hyperparameters (a 1d array of length D+1 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used.
    gp_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the ``gp_mean_function'' at a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x D), hyperparameters (a 1d array of length D+1 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 2d array of shape (len(hyperparameters) x N1). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparametes, or a finite-difference approximation
        is used if ``gp_mean_function'' is provided.
    gp_noise_function : Callable optional
        The noise function is a callable f(x,hyperparameters,obj) that returns a
        positive symmetric definite matrix of shape(len(x),len(x)).
        The input x is a numpy array of shape (N x D). The hyperparameter array is the same
        that is communicated to mean and kernel functions. The obj is a fvgp.GP instance.
    gp_noise_function_grad : Callable, optional
        A function that evaluates the gradient of the ``gp_noise_function'' at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x D), hyperparameters (a 1d array of length D+1 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 3-D array of shape (len(hyperparameters) x N x N). If None is provided, either
        zeros are returned since the default noise function does not dpeend on hyperparametes. If ``gp_noise_function'' is provided but no gradient function,
        a finite-difference approximation will be used.
        The same rules regarding ram economoy as for the kernel definition apply here.
    normalize_y : bool, optional
        If True, the data values ``y_data'' will be normalized to max(y_data) = 1, min(y_data) = 0. The default is False.
        Variances will be updated accordingly.
    sparse_mode : bool, optional
        When sparse_mode is enabled, the algorithm will use a user-defined kernel function or, if that's not provided, an anisotropic Wendland kernel
        and check for sparsity in the prior covariance. If sparsity is present, sparse operations will be used to speed up computations.
        Caution: the covariace is still stored at first in a dense format. For more extreme scaling, check out the gp2Scale option.
    gp2Scale: bool, optional
        Turns on gp2Scale. This will distribute the covariance computations across multiple workers. This is an advaced feature for HPC GPs up to 10
        million datapoints. If gp2Scale is used, the default kernel is an anisotropic Wemsland kernel which is compactly supported. The noise function will have
        to return a scipy.sparse matrix instead of a numpy array. There are a few more things to consider (read on); this is an advanced option.
        If no kernel is provided, the compute_device option should be revisited. The kernel will use the specified device to compute covariances.
        The default is False.
    gp2Scale_dask_client : dask.distributed.Client, optional
        A dask client for gp2Scale to distribute covariance computations over. Has to contain at least 3 workers.
        On HPC architecture, this client is provided by the jobscript. Please have a look at the examples.
        A local client is used as default.
    gp2Scale_batch_size : int, optional
        Matrix batch size for distributed computing in gp2Scale. The default is 10000.
    store_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance matrix after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster.
        For larger problems (>2000 data points), the use of inversion should be avoided due to computational instability and costs. The default is
        True. Note, the training will always use Cholesky or LU decomposition instead of the inverse for stability reasons. Storing the inverse is
        a good option when the dataset is not too large and the posterior covariance is heavily used.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the marginal log_likelihood is/are used for the training.
        If True, components of the derivative of the marginal log-likelihood are calculated subsequently, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (or noise function) with respect to the hyperparameters (gp_kernel_function_grad) is
        going to be provided, it has to be tailored: for ram_economy=True it should be of the form f(x1[, x2], direction, hyperparameters, obj)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If ram_economy=False, the function should be of the form f(x1[, x2,] hyperparameters, obj) and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters. CAUTION: This array will be stored and is very large.
    args : any, optional
        args will be a class attribute and therefore available to kernel, noise and prior mean functions.
    info : bool, optional
        Provides a way how to see the progress of gp2Scale, Default is False
    cost_function : Callable, optional
        A function encoding the cost of motion through the input 
        space and the cost of a measurement. Its inputs
        are an `origin` (np.ndarray of size V x D), `x` 
        (np.ndarray of size V x D), and the value of `cost_func_params`;
        `origin` is the starting position, and `x` is the 
        destination position. The return value is a 1-D array of
        length V describing the costs as floats. The 'score' from 
        acquisition_function is divided by this
        returned cost to determine the next measurement point. 
        The default in no-op.
    cost_function_parameters : object, optional
        This object is transmitted to the cost function; 
        it can be of any type. The default is None.
    cost_update_function : Callable, optional
        If provided this function will be used when 
        `gpcam.gp_optimizer.GPOptimizer.update_cost_function` is called.
        The function `cost_update_function` accepts as 
        input costs (a list of cost values usually determined by
        `instrument_func`) and a parameter
        object. The default is a no-op.

    Attributes
    ----------
    x_data : np.ndarray
        Datapoint positions
    y_data : np.ndarray
        Datapoint values
    variances : np.ndarray
        Datapoint observation variances
    input_dim : int
        Dimensionality of the input space
    input_space_bounds : np.ndarray
        Bounds of the input space
    hyperparameters : np.ndarray
        Only available after training is executed.
    """

    def __init__(
        self,
        x_data,
        y_data,
        init_hyperparameters = None,
        noise_variances = None,
        compute_device = "cpu",
        gp_kernel_function = None,
        gp_kernel_function_grad = None,
        gp_noise_function = None,
        gp_noise_function_grad = None,
        gp_mean_function = None,
        gp_mean_function_grad = None,
        sparse_mode = False,
        gp2Scale = False,
        gp2Scale_dask_client = None,
        gp2Scale_batch_size = 10000,
        normalize_y = False,
        store_inv = True,
        ram_economy = False,
        args = None,
        info = False,
        cost_function = None,
        cost_function_parameters = None,
        cost_update_function = None
        ):
        if isinstance(x_data,np.ndarray):
            if np.ndim(x_data) == 1: x_data = x_data.reshape(-1,1)
            input_dim = x_data.shape[1]
        else: input_dim = 1
        super().__init__(
                input_dim,
                x_data,
                y_data,
                init_hyperparameters = init_hyperparameters,
                noise_variances = noise_variances,
                compute_device = compute_device,
                gp_kernel_function = gp_kernel_function,
                gp_kernel_function_grad = gp_kernel_function_grad,
                gp_noise_function = gp_noise_function,
                gp_noise_function_grad = gp_noise_function_grad,
                gp_mean_function = gp_mean_function,
                gp_mean_function_grad = gp_mean_function_grad,
                sparse_mode = sparse_mode,
                gp2Scale = gp2Scale,
                gp2Scale_dask_client = gp2Scale_dask_client,
                gp2Scale_batch_size = gp2Scale_batch_size,
                normalize_y = normalize_y,
                store_inv = store_inv,
                ram_economy = ram_economy,
                args = args,
                info = info,
                )
        self.cost_function = cost_function
        self.cost_function_parameters = cost_function_parameters
        self.cost_update_function = cost_update_function

    def get_data(self):
        """
        Function that provides a way to access the class attributes.

        Return
        ------
        dictionary of class attributes : dict
        """

        return {
            "input dim": self.input_space_dim,
            "x data": self.x_data,
            "y data": self.y_data,
            "measurement variances": self.measurmenet_noise,
            "hyperparameters": self.hyperparameters,
            "cost function parameters": self.cost_function_parameters,
            "cost function": self.cost_function}

    def evaluate_acquisition_function(self, x, acquisition_function="variance", origin=None, args = None):
        """
        Function to evaluate the acquisition function.

        Parameters
        ----------
        x : np.ndarray
            Point positions at which the acquisition function is evaluated. Shape (N x D).
        acquisition_function : Callable, optional
            Acquisition function to execute. Callable with inputs (x,gpcam.GPOptimizer),
            where x is a V x D array of input x position. The return value is a 1-D array of length V.
            The default is `variance`.
        origin : np.ndarray, optional
            If a cost function is provided this 1-D numpy array of length D is used as the origin of motion.
        args : any, optional
            Arguments that will be communicated to your acquisition function.

        Return
        ------
        np.ndarray
            The acquisition function evaluations at all points `x`.
        """
        if self.cost_function and origin is None: warnings.warn("Warning: For the cost function to be active, an origin has to be provided.")
        if origin is not None and self.cost_function is None: warnings.warn("Warning: An origin is given but no cost function is defined. Cost function ignored")
        try:
            res = sm.evaluate_acquisition_function(
                x, self, acquisition_function, origin = origin, number_of_maxima_sought = None, cost_function = self.cost_function, cost_function_parameters = self.cost_function_parameters, args = args)
            return -res
        except Exception as ex:
            raise Exception("Evaluating the acquisition function was not successful.", ex)
            logger.error(ex)
            logger.error("Evaluating the acquisition function was not successful.")

    def tell(self, x, y, noise_variances=None):
        """
        This function can tell() the gp_optimizer class
        the data that was collected. The data will instantly be used to update the gp data.
        IMPORTANT: This call does not append data. The entire dataset, including the updates,
        has to be provided.

        Parameters
        ----------
        x : np.ndarray
            Point positions (of shape U x D) to be communicated to the Gaussian Process.
        y : np.ndarray
            Point values (of shape U x 1 or U) to be communicated to the Gaussian Process.
        noise_variances : np.ndarray, optional
            Point value variances (of shape U x 1 or U) to be communicated to the Gaussian Process.
            If not provided, the GP will 1% of the y values as variances.
        """
        super().update_gp_data(x,y,noise_variances = noise_variances)

    ##############################################################
    def init_gp(
            self,
            init_hyperparameters,
            compute_device="cpu",
            gp_kernel_function=None,
            gp_mean_function=None,
            gp_kernel_function_grad = None,
            gp_mean_function_grad = None,
            normalize_y = False,
            use_inv=False,
            ram_economy=True):
        ####DEPRECIATED
        ####DEPRECIATED
        ####DEPRECIATED
        ####DEPRECIATED
        ####DEPRECIATED
        ####DEPRECIATED
        warnings.warn("Call 'init_gp' depreciated. Theinitialization happens now at the time of the gp_optimizer initialization.")


    ##############################################################
    def train_gp(self,
        hyperparameter_bounds = None,
        init_hyperparameters = None,
        method = "global",
        pop_size = 20,
        tolerance = 0.0001,
        max_iter = 120,
        local_optimizer = "L-BFGS-B",
        global_optimizer = "genetic",
        constraints = (),
        dask_client = None):
        """
        This function finds the maximum of the log marginal likelihood and therefore trains the GP (synchronously).
        This can be done on a remote cluster/computer by specifying the method to be be 'hgdl' and
        providing a dask client. However, in that case fvgp.GP.train_aync() is preferred.
        The GP prior will automatically be updated with the new hyperparameters after the training.

        Parameters
        ----------
        hyperparameter_bounds : np.ndarray, optional
            A numpy array of shape (D x 2), defining the bounds for the optimization.
            The default is an array of bounds of the length of the initial hyperparameters
            with all bounds defined practically as [0.00001, inf].
            The initial hyperparameters are either defined by the user at initialization, or in this function call,
            or are defined as np.ones((input_space_dim + 1)).
            This choice is only recommended in very basic scenarios and
            can lead to suboptimal results. It is better to provide
            hyperparameter bounds.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is reusing the initial hyperparameters given at initialization
        method : str or Callable, optional
            The method used to train the hyperparameters. The options are `'global'`, `'local'`, `'hgdl'`, `'mcmc'`, and a callable.
            The callable gets an gp.GP instance and has to return a 1d np.array of hyperparameters.
            The default is `'global'` (scipy's differential evolution).
            If method = "mcmc",
            the attribute fvgp.GP.mcmc_info is updated and contains convergence and distribution information.
        pop_size : int, optional
            A number of individuals used for any optimizer with a global component. Default = 20.
        tolerance : float, optional
            Used as termination criterion for local optimizers. Default = 0.0001.
        max_iter : int, optional
            Maximum number of iterations for global and local optimizers. Default = 120.
        local_optimizer : str, optional
            Defining the local optimizer. Default = "L-BFGS-B", most scipy.opimize.minimize functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to method = hgdl. Default = `genetic`
        constraints : tuple of object instances, optional
            Equality and inequality constraints for the optimization. If the optimizer is ``hgdl'' see ``hgdl.readthedocs.io''.
            If the optimizer is a scipy optimizer, see the scipy documentation.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if HGDL is used. If None is provided, a new
            `dask.distributed.Client` instance is constructed.
        """
        super().train(
        hyperparameter_bounds = hyperparameter_bounds,
        init_hyperparameters = init_hyperparameters,
        method = method,
        pop_size = pop_size,
        tolerance = tolerance,
        max_iter = max_iter,
        local_optimizer = local_optimizer,
        global_optimizer = global_optimizer,
        constraints = constraints,
        dask_client = dask_client)

        return self.hyperparameters
    ##############################################################
    def train_gp_async(self,
        hyperparameter_bounds = None,
        init_hyperparameters = None,
        max_iter = 10000,
        local_optimizer = "L-BFGS-B",
        global_optimizer = "genetic",
        constraints = (),
        dask_client = None
        ):

        """
        This function asynchronously finds the maximum of the log marginal likelihood and therefore trains the GP.
        This can be done on a remote cluster/computer by
        providing a dask client. This function submits the training and returns
        an object which can be given to `fvgp.GP.update_hyperparameters()`, 
        which will automatically update the GP prior with the new hyperparameters.

        Parameters
        ----------
        hyperparameter_bounds : np.ndarray, optional
            A numpy array of shape (D x 2), defining the bounds for the optimization. The default is an array of bounds for the default kernel D = input_space_dim + 1
            with all bounds defined practically as [0.00001, inf]. This choice is only recommended in very basic scenarios.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is reusing the initial hyperparameters given at initialization
        max_iter : int, optional
            Maximum number of epochs for HGDL. Default = 10000.
        local_optimizer : str, optional
            Defining the local optimizer. Default = "L-BFGS-B", most scipy.opimize.minimize functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to method = hgdl. Default = `genetic`
        constraints : tuple of hgdl.NonLinearConstraint instances, optional
            Equality and inequality constraints for the optimization. See ``hgdl.readthedocs.io''
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if HGDL is used. If None is provided, a new
            `dask.distributed.Client` instance is constructed.

        Return
        ------
        Optimization object that can be given to `gpcam.GPOptimizer.update_hyperparameters()` to update the prior GP : object instance
        """

        opt_obj = super().train_async(
        hyperparameter_bounds = hyperparameter_bounds ,
        init_hyperparameters = init_hyperparameters,
        max_iter = max_iter,
        local_optimizer = local_optimizer,
        global_optimizer = global_optimizer,
        constraints = constraints,
        dask_client = dask_client
        )
        return opt_obj

    ##############################################################
    def stop_async_train(self, opt_obj):
        """
        Function to stop an asynchronous training. This leaves the dask.distributed.Client alive.

        Parameters
        ----------
        opt_obj : object instance
            Object instance created by gpcam.GPOptimizer.train_gp_async()
        """
        super().stop_training(opt_obj)

    def kill_async_train(self, opt_obj):
        """
        Function to kill an asynchronous training. This shuts down the associated dask.distributed.Client.

        Parameters
        ----------
        opt_obj : object instance
            Object instance created by gpcam.GPOptimizer.train_gp_async()
        """
        super().kill_training(opt_obj)

    ##############################################################
    def update_hyperparameters(self, opt_obj):
        """
        Function to update the Gaussian Process hyperparameters if an asynchronous training is running.

        Parameters
        ----------
        opt_obj : object instance
            Object instance created by gpcam.gp_optimizer.train_gp_async()

        Return
        ------
            hyperparameters : np.ndarray
        """

        hps = super().update_hyperparameters(opt_obj)
        return hps

    ##############################################################
    def ask(self,
            bounds,
            position=None, 
            n=1,
            acquisition_function="variance",
            method="global",
            pop_size=20,
            max_iter=20,
            tol=1e-6,
            constraints = (),
            x0=None,
            vectorized = True,
            args = {},
            info = False,
            dask_client=None):

        """
        Given that the acquisition device is at "position", this function ask()s for
        "n" new optimal points within certain "bounds" and using the optimization setup: "method",
        "acquisition_function_pop_size", "max_iter", "tol", "constraints", and "x0".

        Parameters
        ----------
        bounds : np.ndarray
            A numpy array of floats of shape D x 2 describing the
            search range.
        position : np.ndarray, optional
            Current position in the input space. If a cost function is 
            provided this position will be taken into account
            to guarantee a cost-efficient new suggestion. The default is None.
        n  : int, optional
            The algorithm will try to return n suggestions for
            new measurements. This is either done by method = 'hgdl', or otherwise
            by maximizing the collective information gain (default).
        acquisition_function : Callable, optional
            The acquisition function accepts as input a numpy array
            of size V x D (such that V is the number of input
            points, and D is the parameter space dimensionality) and
            a `GPOptimizer` object. The return value is 1-D array
            of length V providing 'scores' for each position,
            such that the highest scored point will be measured next.
            Built-in functions can be used by one of the following keys: 
            `'shannon_ig'`, `'shannon_ig_vec'`,
            `'ucb'`, `'maximum'`,
            `'minimum'`, `'covariance'`, `'variance'`, `'expected_improvement'`, 
            `'PI'` (probability of improvement), and `'gradient'`. 
            If None, the default function `'variance'`, meaning
            `fvgp.GP.posterior_covariance` with variance_only = True will be used.
            The acquisition function can be a callable of the form my_func(x,gpcam.GPOptimizer)
            which will be maximized (!!!), so make sure desirable new measurement points
            will be located at maxima.
            Explanations of the acquisition functions:
            shannon_if: mutual information, shannon info gain, predicted entropy change,
            shannon_ig_vec: mutual info per-point but vectorized for speed
            ucb: upper confidence bound, posterior mean + 3. std
            maximum: finds the maximum of the current posterior mean
            minimum: finds the maximum of the current posterior mean
            covariances: sqrt(|posterior covariance|)
            variance: the posterior variance
            gradient: puts focus on high-gradient regions
        method : str, optional
            A string defining the method used to find the maximum
            of the acquisition function. Choose from `global`,
            `local`, `hgdl`. HGDL is an in-house hybrid optimizer
            that is comfortable on HPC hardware.
            The default is `global`.
        pop_size : int, optional
            An integer defining the number of individuals if `global` 
            is chosen as method. The default is 20. For
            `hgdl` this will be overwritten
            by the 'dask_client` definition.
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
        constraints : tuple of object instances, optional
            scipy constraints instances, depending on the used optimizer.
        args : dict, optional
            Provides arguments for certain acquisition
            functions, such as, "target_probability". In this case it should be
            defined as {"a": some lower bound, "b":some upper bound}, 
            example: "args = {"a": 1.0,"b": 3.0}".
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed
            `acquisition_function` optimization. If None is provided,
            a new `dask.distributed.Client` instance is constructed for hgdl.

        Return
        ------
        dictionary : {'x': np.array(maxima), "f(x)" : np.array(func_evals), "opt_obj" : opt_obj}
            Found maxima of the acquisition function, the associated function values and optimization object
            that, only in case of `method` = `hgdl` can be queried for solutions.
        """

        logger.info("ask() initiated with hyperparameters: {}", self.hyperparameters)
        logger.info("optimization method: {}", method)
        logger.info("bounds:\n{}", bounds)
        logger.info("acq func: {}", acquisition_function)

        if n > 1 and method != "hgdl":
            method = "global"
            new_optimization_bounds = np.row_stack([bounds for i in range(n)])
            bounds = new_optimization_bounds
            acquisition_function = "shannon_ig"
            vectorized = False
        if acquisition_function == "shannon_ig" or \
           acquisition_function == "covariance":
               vectorized = False
        if method != "global": vectorized = False

        print(acquisition_function)
        print(bounds)


        maxima, func_evals, opt_obj = sm.find_acquisition_function_maxima(
            self,
            acquisition_function,
            position, n, bounds,
            optimization_method=method,
            optimization_pop_size=pop_size,
            optimization_max_iter=max_iter,
            optimization_tol=tol,
            cost_function=self.cost_function,
            cost_function_parameters=self.cost_function_parameters,
            optimization_x0=x0,
            constraints = constraints,
            vectorized = vectorized,
            args = args,
            info = info,
            dask_client=dask_client)
        if n > 1: return {'x': maxima.reshape(n,self.input_space_dim), "f(x)": np.array(func_evals), "opt_obj": opt_obj}
        return {'x': np.array(maxima), "f(x)": np.array(func_evals), "opt_obj": opt_obj}

    ##############################################################
    def init_cost(self, cost_function, cost_function_parameters=None, cost_update_function=None):
        ####DEPRECIATED
        ####DEPRECIATED
        ####DEPRECIATED
        ####DEPRECIATED
        ####DEPRECIATED
        ####DEPRECIATED
        warnings.warn("Call 'init_cost' is depreciated in will be removed in the next version. The costs are now initialized in the GPOptimizer init call.")

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
            updating the parameters are both user-defined and this object has to be in accordance with those definitions.
        Return
        ------
            No return, the cost function parameters will automatically be updated.
        """

        if self.cost_function_parameters is None: warnings.warn("No cost_function_parameters specified. Cost update failed.")
        if callable(self.cost_update_function): self.cost_function_parameters = self.cost_update_function(measurement_costs, self.cost_function_parameters)
        else: warnings.warn("No cost_update_function available. Cost update failed.")

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
class fvGPOptimizer(fvGP):
    """
    This class is an optimization wrapper around the fvgp package for multi-task (scalar-valued) Gaussian Processes.
    Gaussian Processes can be initialized, trained, and conditioned; also
    the posterior can be evaluated and used via acquisition functions,
    and plugged into optimizers to find its maxima. This class inherits many methods from
    the fvgp.GP class. Check fvgp.readthedocs.io for a full list of capabilities.

    V ... number of input points
    D ... input space dimensionality
    N ... arbitrary integers (N1, N2,...)

    V ... number of input points
    Di... input space dimensionality
    Do... output space dimensionality
    No... number of outputs
    N ... arbitrary integers (N1, N2,...)


    The main logic of fvGP is that any multi-task GP is just a single-task GP
    over a Cartesian product space of input and output space, as long as the kernel
    is flexible enough, so prepare to work on your kernel. This is the best 
    way to give the user optimal control and power. At various instances, for instances
    prior-mean function, noise function, and kernel function definitions, you will
    see that the input ``x'' is defined over this combined space. 
    For example, if your input space is a Euclidean 2d space and your output
    is labelled [[0],[1]], the input to the mean, kernel, and noise function might be
    x =
    [[0.2, 0.3,0],[0.9,0.6,0],
     [0.2, 0.3,1],[0.9,0.6,1]]
    This has to be understood and taken into account when customizing fvGP for multi-task
    use.

    Parameters
    ----------
    input_space_dim : int
        Dimensionality of the input space (D).
    output_space_dim : int
        Integer specifying the number of dimensions of the output space. Most often 1. This is not the number of outputs/tasks.
        For instance, a spectrum as output at each input is itslef a function over a 1d space but has many outputs.
    output_number : int
        Number of output values.
    x_data : np.ndarray
        The input point positions. Shape (V x D), where D is the `input_space_dim`.
    y_data : np.ndarray
        The values of the data points. Shape (V,No).
    init_hyperparameters : np.ndarray, optional
        Vector of hyperparameters used by the GP initially.
        This class provides methods to train hyperparameters.
        The default is an array that specifies the right number of
        initial hyperparameters for the default kernel, which is
        a deek kernel with two layers of width
        fvgp.fvGP.gp_deep_kernel_layer_width. If you specify 
        another kernel, please provide
        init_hyperparameters.
    output_positions : np.ndarray, optional
        A 3-D numpy array of shape (U x output_number x output_dim), so that for each measurement position, the outputs
        are clearly defined by their positions in the output space. The default is np.array([[0],[1],[2],[3],...,[output_number - 1]]) for each
        point in the input space. The default is only permissible if output_dim is 1.
    noise_variances : np.ndarray, optional
        An numpy array defining the uncertainties/noise in the data
        `y_data` in form of a point-wise variance. Shape y_data.shape.
        Note: if no noise_variances are provided here, the gp_noise_function
        callable will be used; if the callable is not provided, the noise variances
        will be set to `abs(np.mean(y_data) / 100.0`. If
        noise covariances are required, also make use of the gp_noise_function.
    compute_device : str, optional
        One of "cpu" or "gpu", determines how linear system solves are run. The default is "cpu".
        For "gpu", pytoch has to be installed manually.
        If gp2Scale is enabled but no kernel is provided, the choice of the compute_device
        becomes much more important. In that case, the default kernel will be computed on
        the cpu or the gpu which will significantly change the compute time depending on the compute
        architecture.
    gp_kernel_function : Callable, optional
        A symmetric positive semi-definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters, obj).
        The input x1 is a N1 x Di+Do array of positions, x2 is a N2 x Di+Do
        array of positions, the hyperparameters argument
        is a 1d array of length N depending on how many hyperpapameters are initialized, and
        obj is an `fvgp.GP` instance. The default is a deep kernel with 2 hidden layers and
        a width of fvgp.fvGP.gp_deep_kernel_layer_width.
    gp_deep_kernel_layer_width : int, optional
        If no kernel is provided, fvGP will use a deep kernel of depth 2 and width gp_deep_kernel_layer_width.
        If a user defined kernel is provided this parameter is irrelevant. The default is 5.
    gp_kernel_function_grad : Callable, optional
        A function that calculates the derivative of the ``gp_kernel_function'' with respect to the hyperparameters.
        If provided, it will be used for local training (optimization) and can speed up the calculations.
        It accepts as input x1 (a N1 x Di+Do array of positions),
        x2 (a N2 x Di+Do array of positions),
        hyperparameters, and a
        `fvgp.GP` instance. The default is a finite difference calculation.
        If 'ram_economy' is True, the function's input is x1, x2, direction (int), hyperparameters (numpy array), and a
        `fvgp.GP` instance, and the output
        is a numpy array of shape (len(hps) x N).
        If 'ram economy' is False,the function's input is x1, x2, hyperparameters, and a
        `fvgp.GP` instance. The output is
        a numpy array of shape (len(hyperparameters) x N1 x N2). See 'ram_economy'.
    gp_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x Di+Do), hyperparameters
        and a `fvgp.GP` instance. The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used.
    gp_mean_function_grad : Callable, optional
        A function that evaluates the gradient of the ``gp_mean_function'' at a set of input positions with respect to the hyperparameters.
        It accepts as input an array of positions (of size N1 x Di+Do), hyperparameters
        and a `fvgp.GP` instance. The return value is a 2d array of shape (len(hyperparameters) x N1). If None is provided, either
        zeros are returned since the default mean function does not depend on hyperparametes, or a finite-difference approximation
        is used if ``gp_mean_function'' is provided.
    gp_noise_function : Callable optional
        The noise function is a callable f(x,hyperparameters,obj) that returns a
        positive symmetric definite matrix of shape(len(x),len(x)).
        The input x is a numpy array of shape (N x Di+Do). The hyperparameter array is the same
        that is communicated to mean and kernel functions. The obj is a fvgp.fvGP instance.
    gp_noise_function_grad : Callable, optional
        A function that evaluates the gradient of the ``gp_noise_function'' at an input position with respect to the hyperparameters.
        It accepts as input an array of positions (of size N x Di+Do), hyperparameters (a 1d array of length D+1 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 3-D array of shape (len(hyperparameters) x N x N). If None is provided, either
        zeros are returned since the default noise function does not dpeend on hyperparametes. If ``gp_noise_function'' is provided but no gradient function,
        a finite-difference approximation will be used. 
        The same rules regarding ram economoy as for the kernel definition apply here.
    normalize_y : bool, optional
        If True, the data values ``y_data'' will be normalized to max(y_data) = 1, min(y_data) = 0. The default is False.
        Variances will be updated accordingly.
    sparse_mode : bool, optional
        When sparse_mode is enabled, the algorithm will use a user-defined kernel function or, if that's not provided, an anisotropic Wendland kernel
        and check for sparsity in the prior covariance. If sparsity is present, sparse operations will be used to speed up computations.
        Caution: the covariace is still stored at first in a dense format. For more extreme scaling, check out the gp2Scale option.
    gp2Scale: bool, optional
        Turns on gp2Scale. This will distribute the covariance computations across multiple workers. This is an advaced feature for HPC GPs up to 10
        million datapoints. If gp2Scale is used, the default kernel is an anisotropic Wemsland kernel which is compactly supported. The noise function will have
        to return a scipy.sparse matrix instead of a numpy array. There are a few more things to consider (read on); this is an advanced option.
        If no kernel is provided, the compute_device option should be revisited. The kernel will use the specified device to compute covariances.
        The default is False.
    gp2Scale_dask_client : dask.distributed.Client, optional
        A dask client for gp2Scale to distribute covariance computations over. Has to contain at least 3 workers.
        On HPC architecture, this client is provided by the jobscript. Please have a look at the examples.
        A local client is used as default.
    gp2Scale_batch_size : int, optional
        Matrix batch size for distributed computing in gp2Scale. The default is 10000.
    store_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance matrix after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster.
        For larger problems (>2000 data points), the use of inversion should be avoided due to computational instability and costs. The default is
        True. Note, the training will always use Cholesky or LU decomposition instead of the inverse for stability reasons. Storing the inverse is
        a good option when the dataset is not too large and the posterior covariance is heavily used.
        If sparse_mode or gp2Scale is used, store_inv will be set to False.
    ram_economy : bool, optional
        Only of interest if the gradient and/or Hessian of the marginal log_likelihood is/are used for the training.
        If True, components of the derivative of the marginal log-likelihood are calculated subsequently, leading to a slow-down
        but much less RAM usage. If the derivative of the kernel (or noise function) with respect to the hyperparameters (gp_kernel_function_grad) is
        going to be provided, it has to be tailored: for ram_economy=True it should be of the form f(x1[, x2], direction, hyperparameters, obj)
        and return a 2d numpy array of shape len(x1) x len(x2).
        If ram_economy=False, the function should be of the form f(x1[, x2,] hyperparameters, obj) and return a numpy array of shape
        H x len(x1) x len(x2), where H is the number of hyperparameters. CAUTION: This array will be stored and is very large.
    args : any, optional
        args will be a class attribute and therefore available to kernel, noise and prior mean functions.
    info : bool, optional
        Provides a way how to see the progress of gp2Scale, Default is False
    cost_function : Callable, optional
        A function encoding the cost of motion through the input 
        space and the cost of a measurement. Its inputs
        are an `origin` (np.ndarray of size V x D), `x` 
        (np.ndarray of size V x D), and the value of `cost_func_params`;
        `origin` is the starting position, and `x` is the 
        destination position. The return value is a 1-D array of
        length V describing the costs as floats. The 'score' from 
        acquisition_function is divided by this
        returned cost to determine the next measurement point. 
        The default in no-op.
    cost_function_parameters : object, optional
        This object is transmitted to the cost function; 
        it can be of any type. The default is None.
    cost_update_function : Callable, optional
        If provided this function will be used when 
        `gpcam.gp_optimizer.GPOptimizer.update_cost_function` is called.
        The function `cost_update_function` accepts as 
        input costs (a list of cost values usually determined by
        `instrument_func`) and a parameter
        object. The default is a no-op.


    Attributes
    ----------
    x_data : np.ndarray
        Datapoint positions
    y_data : np.ndarray
        Datapoint values
    noise_variances : np.ndarray
        Datapoint observation variances
    input_dim : int
        Dimensionality of the input space
    input_space_bounds : np.ndarray
        Bounds of the input space
    gp_initialized : bool
        A check whether the object instance has an initialized Gaussian Process.
    hyperparameters : np.ndarray
        Only available after the training is executed.
    """


    def __init__(
        self,
        input_space_dim,
        output_space_dim,
        output_number,
        x_data,
        y_data,
        init_hyperparameters = None,
        output_positions = None,
        noise_variances = None,
        compute_device = "cpu",
        gp_kernel_function = None,
        gp_deep_kernel_layer_width = 5,
        gp_kernel_function_grad = None,
        gp_noise_function = None,
        gp_noise_function_grad = None,
        gp_mean_function = None,
        gp_mean_function_grad = None,
        sparse_mode = False,
        gp2Scale = False,
        gp2Scale_dask_client = None,
        gp2Scale_batch_size = 10000,
        normalize_y = False,
        store_inv = True,
        ram_economy = False,
        args = None,
        info = False,
        cost_function = None,
        cost_function_parameters=None,
        cost_update_function=None,
        ):

        self.iput_dim = input_space_dimension
        self.oput_dim = output_space_dimension
        self.output_number = output_number
        x_data = np.empty((1, self.iput_dim))
        y_data = np.empty((1, self.output_number))
        variances = np.empty((1, self.output_number))
        output_positions = np.empty((1, self.output_number, self.oput_dim))
        self.input_space_bounds = np.array(input_space_bounds)
        self.gp_initialized = True

        super().__init__(
                input_space_dim,
                output_space_dim,
                output_number,
                x_data,
                y_data,
                init_hyperparameters = init_hyperparameters,
                output_positions = output_positions,
                noise_variances = noise_variances,
                compute_device = compute_device,
                gp_kernel_function = gp_kernel_function,
                gp_kernel_function_grad = gp_kernel_function_grad,
                gp_noise_function = gp_noise_function,
                gp_noise_function_grad = gp_noise_function_grad,
                gp_mean_function = gp_mean_function,
                gp_mean_function_grad = gp_mean_function_grad,
                sparse_mode = sparse_mode,
                gp2Scale = gp2Scale,
                gp2Scale_dask_client = gp2Scale_dask_client,
                gp2Scale_batch_size = gp2Scale_batch_size,
                normalize_y = normalize_y,
                store_inv = store_inv,
                ram_economy = ram_economy,
                args = args,
                info = info)

        self.cost_function = cost_function
        self.cost_function_parameters = cost_function_parameters
        self.cost_update_function = cost_update_function


    ############################################################################
    def get_data(self):
        """
        Function that provides a way to access the class attributes.

        Return
        ------
        dictionary of class attributes : dict
        """

        return {
            "input dim": self.input_space_dim,
            "x data": self.x_data,
            "y data": self.y_data,
            "measurement variances": self.measurmenet_noise,
            "hyperparameters": self.hyperparameters,
            "cost function parameters": self.cost_function_parameters,
            "cost function": self.cost_function}

    ############################################################################
    def evaluate_acquisition_function(self, x, acquisition_function="variance", origin=None):
        """
        Function to evaluate the acquisition function.

        Parameters
        ----------
        x : np.ndarray
            Point positions at which the acquisition function is evaluated. This is a point in the input space.
        acquisition_function : Callable, optional
            Acquisition function to execute. Callable with inputs (x,gpcam.gp_optimizer.GPOptimizer),
            where x is a V x D array of input x_data. The return value is a 1-D array of length V.
            The default is `variance`.
        origin : np.ndarray, optional
            If a cost function is provided this 1-D numpy array of length D is used as the origin of motion.

        Return
        ------
        The acquisition function evaluations at all points `x` : np.ndarray
        """
        if self.cost_function and origin is None: print("Warning: For the cost function to be active, an origin has to be provided.")
        if self.gp_initialized is False:
            raise Exception(
                "Initialize GP before evaluating the acquisition function. "
                "See help(gp_init)."
            )
        x = np.array(x)
        cost_function = self.cost_function
        try:
            res = sm.evaluate_acquisition_function(
                x, self, acquisition_function, origin = origin, number_of_maxima_sought = None, cost_function = cost_function, cost_function_parameters = self.cost_function_parameters, args = None)
            return -res
        except Exception as ex:
            raise Exception("Evaluating the acquisition function was not successful.", ex)
            logger.error(ex)
            logger.error("Evaluating the acquisition function was not successful.")

    ############################################################################
    def tell(self, x, y, noise_variances=None, output_positions=None):
        """
        This function can tell() the gp_optimizer class
        the data that was collected. The data will instantly be used to update the gp_daa

        Parameters
        ----------
        x : np.ndarray
            Point positions (of shape U x D) to be communicated to the Gaussian Process.
        y : np.ndarray
            Point values (of shape U x 1 or U) to be communicated to the Gaussian Process.
        variances : np.ndarray, optional
            Point value variances (of shape U x 1 or U) to be communicated 
            to the Gaussian Process.
            If not provided, the GP will 1% of the y values as variances.
        output_positions : np.ndarray, optional
            A 3-D numpy array of shape (U x output_number x output_dim),
            so that for each measurement position, the outputs
            are clearly defined by their positions in the output space.
            The default is np.array([[0],[1],[2],[3],...,[output_number - 1]]) for each
            point in the input space. The default is only permissible if output_dim is 1.
        """
        super().update_gp_data(x,y,noise_variances = noise_variances, output_positions = output_positions)
    ##############################################################
    def train_gp(self,
        hyperparameter_bounds = None,
        init_hyperparameters = None,
        method = "global",
        pop_size = 20,
        tolerance = 0.0001,
        max_iter = 120,
        local_optimizer = "L-BFGS-B",
        global_optimizer = "genetic",
        constraints = (),
        dask_client = None):
        """
        This function finds the maximum of the log marginal likelihood and therefore trains the GP (synchronously).
        This can be done on a remote cluster/computer by specifying the method to be be 'hgdl' and
        providing a dask client. However, in that case fvgp.GP.train_aync() is preferred.
        The GP prior will automatically be updated with the new hyperparameters after the training.

        Parameters
        ----------
        hyperparameter_bounds : np.ndarray, optional
            A numpy array of shape (D x 2), defining the bounds for the optimization.
            The default is an array of bounds of the length of the initial hyperparameters
            with all bounds defined practically as [0.00001, inf].
            The initial hyperparameters are either defined by the user at initialization, or in this function call,
            or are defined as np.ones((input_space_dim + 1)).
            This choice is only recommended in very basic scenarios and
            can lead to suboptimal results. It is better to provide
            hyperparameter bounds.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is reusing the initial hyperparameters given at initialization
        method : str or Callable, optional
            The method used to train the hyperparameters. The options are `'global'`, `'local'`, `'hgdl'`, `'mcmc'`, and a callable.
            The callable gets an gp.GP instance and has to return a 1d np.array of hyperparameters.
            The default is `'global'` (scipy's differential evolution).
            If method = "mcmc",
            the attribute fvgp.GP.mcmc_info is updated and contains convergence and distribution information.
        pop_size : int, optional
            A number of individuals used for any optimizer with a global component. Default = 20.
        tolerance : float, optional
            Used as termination criterion for local optimizers. Default = 0.0001.
        max_iter : int, optional
            Maximum number of iterations for global and local optimizers. Default = 120.
        local_optimizer : str, optional
            Defining the local optimizer. Default = "L-BFGS-B", most scipy.opimize.minimize functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to method = hgdl. Default = `genetic`
        constraints : tuple of object instances, optional
            Equality and inequality constraints for the optimization. If the optimizer is ``hgdl'' see ``hgdl.readthedocs.io''.
            If the optimizer is a scipy optimizer, see the scipy documentation.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if HGDL is used. If None is provided, a new
            `dask.distributed.Client` instance is constructed.
        """
        super().train(
        hyperparameter_bounds = hyperparameter_bounds,
        init_hyperparameters = init_hyperparameters,
        method = method,
        pop_size = pop_size,
        tolerance = tolerance,
        max_iter = max_iter,
        local_optimizer = local_optimizer,
        global_optimizer = global_optimizer,
        constraints = constraints,
        dask_client = dask_client)

        return self.hyperparameters
    ##############################################################
    def train_gp_async(self,
        hyperparameter_bounds = None,
        init_hyperparameters = None,
        max_iter = 10000,
        local_optimizer = "L-BFGS-B",
        global_optimizer = "genetic",
        constraints = (),
        dask_client = None
        ):

        """
        This function asynchronously finds the maximum of the log marginal likelihood and therefore trains the GP.
        This can be done on a remote cluster/computer by
        providing a dask client. This function submits the training and returns
        an object which can be given to `fvgp.GP.update_hyperparameters()`, 
        which will automatically update the GP prior with the new hyperparameters.

        Parameters
        ----------
        hyperparameter_bounds : np.ndarray, optional
            A numpy array of shape (D x 2), defining the bounds for the optimization. The default is an array of bounds for the default kernel D = input_space_dim + 1
            with all bounds defined practically as [0.00001, inf]. This choice is only recommended in very basic scenarios.
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is reusing the initial hyperparameters given at initialization
        max_iter : int, optional
            Maximum number of epochs for HGDL. Default = 10000.
        local_optimizer : str, optional
            Defining the local optimizer. Default = "L-BFGS-B", most scipy.opimize.minimize functions are permissible.
        global_optimizer : str, optional
            Defining the global optimizer. Only applicable to method = hgdl. Default = `genetic`
        constraints : tuple of hgdl.NonLinearConstraint instances, optional
            Equality and inequality constraints for the optimization. See ``hgdl.readthedocs.io''
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training if HGDL is used. If None is provided, a new
            `dask.distributed.Client` instance is constructed.

        Return
        ------
        Optimization object that can be given to `fvgp.GP.update_hyperparameters()` to update the prior GP : object instance
        """

        opt_obj = super().train_async(
        hyperparameter_bounds = hyperparameter_bounds ,
        init_hyperparameters = init_hyperparameters,
        max_iter = max_iter,
        local_optimizer = local_optimizer,
        global_optimizer = global_optimizer,
        constraints = constraints,
        dask_client = dask_client
        )
        return opt_obj

    ##############################################################
    def stop_async_train(self, opt_obj):
        """
        Function to stop an asynchronous training. This leaves the dask.distributed.Client alive.

        Parameters
        ----------
        opt_obj : object instance
            Object instance created by gpcam.GPOptimizer.train_gp_async()
        """
        super().stop_training(opt_obj)

    def kill_async_train(self, opt_obj):
        """
        Function to kill an asynchronous training. This shuts down the associated dask.distributed.Client.

        Parameters
        ----------
        opt_obj : object instance
            Object instance created by gpcam.GPOptimizer.train_gp_async()
        """
        super().kill_training(opt_obj)

    ##############################################################
    def update_hyperparameters(self, opt_obj):
        """
        Function to update the Gaussian Process hyperparameters if an asynchronous training is running.

        Parameters
        ----------
        opt_obj : object instance
            Object instance created by gpcam.gp_optimizer.train_gp_async()

        Return
        ------
            hyperparameters : np.ndarray
        """

        hps = super().update_hyperparameters(self, opt_obj)
        return hps


    def ask(self, 
            acquisition_function,  ###this is now mandatory but will be replaces with some standard ones
            bounds,
            position=None, 
            n=1,
            method="global",
            pop_size=20,
            max_iter=20,
            tol=1e-6,
            constraints = (),
            x0=None,
            vectorized = True,
            args = {},
            dask_client=None):

        """
        Given that the acquisition device is at "position", the function ask()s for
        "n" new optimal points within certain "bounds" and using the optimization setup:
        "acquisition_function_pop_size", "max_iter" and "tol"

        Parameters
        ----------
        position : np.ndarray, optional
            Current position in the input space. If a cost function is 
            provided this position will be taken into account
            to guarantee a cost-efficient new suggestion. The default is None.
        n  : int, optional
            The algorithm will try to return this many suggestions for 
            new measurements. This may be limited by how many
            optima the algorithm may find. If greater than 1, then 
            the `acq_func` optimization method is automatically
            set to use HGDL. The default is 1.
        acquisition_function : Callable, optional
            The acquisition function accepts as input a numpy array 
            of size V x D (such that V is the number of input
            points, and D is the parameter space dimensionality) and 
            a `GPOptimizer` object. The return value is 1-D array
            of length V providing 'scores' for each position, 
            such that the highest scored point will be measured next.
            Built-in functions can be used by one of the following keys: 
            `'shannon_ig'`, `'shannon_ig_vec'`,
            `'ucb'`, `'maximum'`,
            `'minimum'`, `'covariance'`, `'variance'`, and `'gradient'`. 
            If None, the default function is the `'variance'`, meaning
            `fvgp.gp.GP.posterior_covariance` with variance_only = True.
            Explanations of the acquisition functions:
            shannon_ig: mutual information, shannon info gain, ientropy change
            shannon_ig_vec: mutual info per-point but vecotrized for speed
            ucb: upper confidence bound, posterior mean + 3. std
            maximum: finds the maximum of the current posterior mean
            minimum: finds the maximum of the current posterior mean
            covariances: sqrt(|posterior covariance|)
            variance: the posterior variance
            gradient: puts focus on high-gradient regions
        bounds : np.ndarray, optional
            A numpy array of floats of shape D x 2 describing the 
            search range. The default is the entire input space.
        method : str, optional
            A string defining the method used to find the maximum 
            of the acquisition function. Choose from `global`,
            `local`, `hgdl`.
            The default is `global`.
        pop_size : int, optional
            An integer defining the number of individuals if `global` 
            is chosen as method. The default is 20. For
            `hgdl` this will be overwritten
            by the 'dask_client` definition.
        max_iter : int, optional
            This number defined the number of iterations 
            before the optimizer is terminated. The default is 20.
        tol : float, optional
            Termination criterion for the local optimizer. 
            The default is 1e-6.
        x0 : np.ndarray, optional
            A set of points as numpy array of shape V x D, 
            used as starting location(s) for the local and hgdl
            optimization
            algorithm. The default is None.
        vectorized : bool, optional
            If your acquisiiton function vecotrized to return the 
            solution to an array of inquiries as an array, 
            this optionmakes the optimization faster if method = 'global'
            is used. The default is True but will be set to 
            False if method is not global.
        constraints : tuple of object instances, optional
            Either a tuple of hgdl.constraints.NonLinearConstraint 
            or scipy constraints instances, depending on the used optimizer.
        args : dict, optional
            Provides arguments for certain acquisition 
            functions, such as, "target_probability". In this case it should be
            defined as {"a": some lower bound, "b":some upper bound}, 
            example: "args = {"a": 1.0,"b": 3.0}".
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed 
            `acquisition_func` computation. If None is provided,
            a new `dask.distributed.Client` instance is constructed.

        Return
        ------
        dictionary : {'x': np.array(maxima), "f(x)" : np.array(func_evals), "opt_obj" : opt_obj}
            Found maxima of the acquisition function, the associated function values and optimization object
            that, only in case of `method` = `hgdl` can be queried for solutions.
        """

        logger.info("ask() initiated with hyperparameters: {}", self.hyperparameters)
        logger.info("optimization method: {}", method)
        logger.info("bounds:\n{}", bounds)
        logger.info("acq func: {}", acquisition_function)

        if n > 1 and method != "hgdl":
            method = "global"
            new_optimization_bounds = np.row_stack([bounds for i in range(n)])
            bounds = new_optimization_bounds
            acquisition_function = "shannon_ig"
            vectorized = False
        if acquisition_function == "shannon_ig" or \
           acquisition_function == "covariance":
               vectorized = False

        if method != "global": vectorized = False

        maxima, func_evals, opt_obj = sm.find_acquisition_function_maxima(
            self,
            acquisition_function,
            position, n, bounds,
            optimization_method=method,
            optimization_pop_size=pop_size,
            optimization_max_iter=max_iter,
            optimization_tol=tol,
            cost_function=self.cost_function,
            cost_function_parameters=self.cost_function_parameters,
            optimization_x0=x0,
            constraints = constraints,
            vectorized = vectorized,
            args = args,
            dask_client=dask_client)
        if n > 1: return {'x': maxima.reshape(n,self.input_dim), "f(x)": np.array(func_evals), "opt_obj": opt_obj}
        return {'x': np.array(maxima), "f(x)": np.array(func_evals), "opt_obj": opt_obj}

    ##############################################################
    def init_fvgp(
            self,
            init_hyperparameters,
            compute_device="cpu",
            gp_kernel_function=None,
            gp_mean_function=None,
            use_inv=False,
            ram_economy=True
    ):
        ####DEPRECIATED
        ####DEPRECIATED
        ####DEPRECIATED
        ####DEPRECIATED
        ####DEPRECIATED
        ####DEPRECIATED
        ####DEPRECIATED
        warnings.warn("Call 'init_fvgp' is depreciated'. The initialization happens at object initialization.")

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
            updating the parameters are both user-defined and this object has to be in accordance with those definitions.
        Return
        ------
            No return, the cost function parameters will automatically be updated.
        """

        if self.cost_function_parameters is None: warnings.warn("No cost_function_parameters specified. Cost update failed.")
        if callable(self.cost_update_function): self.cost_function_parameters = self.cost_update_function(measurement_costs, self.cost_function_parameters)
        else: warnings.warn("No cost_update_function available. Cost update failed.")

