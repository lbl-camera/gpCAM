#!/usr/bin/env python

import numpy as np
from loguru import logger

from fvgp.fvgp import fvGP
from fvgp.gp import GP
from gpcam import surrogate_model as sm


class GPOptimizer(GP):
    """
    This class is an optimization wrapper around the fvgp package for single-task (scalar-valued) Gaussian Processes.
    Gaussian Processes can be initialized, trained, and conditioned; also
    the posterior can be evaluated and plugged into optimizers to find
    its maxima.

    Parameters
    ---------
    input_space_dimension : int
        Integer specifying the number of dimensions of the input space.
    input_space_bounds : np.ndarray
        A numpy array of floats of shape D x 2 describing the input space range

    Attributes
    ----------
    points : np.ndarray
        Datapoint positions
    values : np.ndarray
        Datapoint values
    variances : np.ndarray
        Datapoint observation variances
    input_dim : int
        Dimensionality of the input space
    input_space_bounds : np.ndarray
        Bounds of the input space
    gp_initialized : bool
        A check whether the object instance has an initialized Gaussian Process.
    hyperparameters : np.ndarray
        Only available after training is executed.
    """

    def __init__(
            self,
            input_space_dimension,
            input_space_bounds,
    ):
        self.iput_dim = input_space_dimension
        self.points = np.empty((0, self.iput_dim))
        self.values = np.empty((0))
        self.variances = np.empty((0))
        self.input_space_bounds = np.array(input_space_bounds)
        self.gp_initialized = False
        self.cost_function_parameters = None
        self.cost_function = None

    def get_data(self):
        """
        Function that provides a way to access the class attributes.

        Return
        ------
        dictionary of class attributes : dict
        """

        if self.gp_initialized:
            hps = self.hyperparameters
        else:
            hps = None
        return {
            "input dim": self.iput_dim,
            "x": self.points,
            "y": self.values,
            "measurement variances": self.variances,
            "hyperparameters": hps,
            "cost function parameters": self.cost_function_parameters,
            "cost function": self.cost_function,
        }

    def evaluate_acquisition_function(self, x, acquisition_function="variance", origin=None):
        """
        Function to evaluate the acquisition function.

        Parameters
        ----------
        x : np.ndarray
            Point positions at which the acquisition function is evaluated.
        acquisition_function : Callable, optional
            Acquisiiton functio to execute. Callable with inputs (x,gpcam.gp_optimizer.GPOptimizer),
            where x is a V x D array of input points. The return value is a 1-D array of length V.
            The default is `variance`.
        origin : np.ndarray, optional
            If a cost function is provided this 1-D numpy array of length D is used as the origin of motion.

        Return
        ------
        The acquisition function evaluations at all points `x` : np.ndarray
            
        """

        if self.gp_initialized is False:
            raise Exception(
                "Initialize GP before evaluating the acquisition function. "
                "See help(gp_init)."
            )
        x = np.array(x)
        cost_function = self.cost_function
        try:
            return sm.evaluate_acquisition_function(
                x, self, acquisition_function, origin, cost_function,
                self.cost_function_parameters)
        except Exception as ex:
            logger.error(ex)
            logger.error("Evaluating the acquisition function was not successful.")

    def tell(self, x, y, variances=None):
        """
        This function can tell() the gp_optimizer class
        the data that was collected. The data will instantly be used to update the gp data
        if a GP was previously initialized.

        Parameters
        ----------
        x : np.ndarray
            Point positions (of shape U x D) to be communicated to the Gaussian Process.
        y : np.ndarray
            Point values (of shape U x 1 or U) to be communicated to the Gaussian Process.
        variances : np.ndarray, optional
            Point value variances (of shape U x 1 or U) to be communicated to the Gaussian Process.
            If not provided, the GP will 1% of the y values as variances.
        """
        self.points = x
        self.values = y
        self.variances = variances

        if self.gp_initialized:
            self._update_gp()

    ##############################################################
    def init_gp(
            self,
            init_hyperparameters,
            compute_device="cpu",
            gp_kernel_function=None,
            gp_mean_function=None,
            use_inv=False,
            ram_economy=True
    ):
        """
        Function to initialize the GP.

        Parameters
        ----------
        init_hyperparameters : np.ndarray
            Initial hyperparameters as 1-D numpy array.
        compute_device : str, optional
            One of "cpu" or "gpu", determines how linear system solves are run. The default is "cpu".
        gp_kernel_function : Callable, optional
            A function that calculates the covariance between datapoints. It accepts as input x1 (a V x D array of
            positions),
            x2 (a U x D array of positions), hyperparameters (a 1-D array of length D+1 for the default kernel), and a
            `gpcam.gp_optimizer.GPOptimizer` instance. The default is a stationary anisotropic kernel
            (`fvgp.gp.GP.default_kernel`).
        gp_mean_function : Callable, optional
            A function that evaluates the prior mean at an input position. It accepts as input a
            `gpcam.gp_optimizer.GPOptimizer` instance, an array of positions (of size V x D), and hyperparameters (a 1-D
            array of length D+1 for the default kernel). The return value is a 1-D array of length V. If None is
            provided,
            `fvgp.gp.GP.default_mean_function` is used.
        use_inv : bool, optional
            If True, the algorithm calculates and stores the inverse of the covariance matrix after each training or update of the dataset,
            which makes computing the posterior covariance faster.
            For larger problems (>2000 data points), the use of inversion should be avoided due to computational instability. The default is
            False. Note, the training will always use a linear solve instead of the inverse for stability reasons.
        ram_economy : bool, optional
            Offers a ram-efficient way to compute marginal-log-likelihood derivatives for training.
        """

        if self.gp_initialized is False:
            GP.__init__(
                self,
                self.iput_dim,
                self.points,
                self.values,
                init_hyperparameters,
                variances=self.variances,
                compute_device=compute_device,
                gp_kernel_function=gp_kernel_function,
                gp_mean_function=gp_mean_function,
                use_inv=use_inv,
                normalize_y=False,
                ram_economy=ram_economy
            )
            self.gp_initialized = True
        else:
            print("No initialization. GP already initialized")

    ##############################################################
    def _update_gp(self):
        """
        This function updates the data in the GP, tell(...) will call this function automatically if
        a GP is intialized
        """
        self.update_gp_data(
            self.points,
            self.values,
            variances=self.variances)

    ##############################################################
    def train_gp_async(self,
                       hyperparameter_bounds,
                       max_iter=10000,
                       dask_client=None,
                       constraints = (),
                       local_method="L-BFGS-B",
                       global_method="genetic"):
        """
        Function to train a Gaussian Process asynchronously on distributed HPC compute architecture using the `HGDL`
        software package.

        Parameters
        ----------
        hyperparameters_bounds : np.ndarray
            Bounds for the optimization of the hyperparameters of shape (V x 2)
        max_iter : int, optional
            Number of iterations before the optimization algorithm is terminated. Since the algorithm works
            asynchronously, this
            number can be high. The default is 10000
        constraints: tuple of object instances, optional
            Either a tuple of hgdl.constraints.NonLinearConstraint or scipy constraints instances, depending on the used optimizer.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training. If None is provided, a local
            `dask.distributed.Client` instance is constructed.
        local_method : str, optional
            Controls the local optimizer running in `HGDL`. Many scipy.minimize optimizers can be used,
            in addition, "dNewton". `L-BFGS-B` is the default.
        global_method : str, optional
            Global optimization step running in `HGDL`. Choose from `genetic` or 'random'.
            The default is `genetic`

        Returns
        -------
            This function return an optimization object (opt_obj) that can be used to stop(opt_obj) or kill(opt_obj)
            the optimization.
            Call `gpcam.gp_optimizer.GPOptimizer.update_hyperparameters(opt_obj)` to update the
            current Gaussian Process with the new hyperparameters. This allows to start several optimization procedures
            and selectively use or stop them.
        """

        if self.gp_initialized is False:
            raise Exception("No GP to be trained. Please call init_gp(...) before training.")
        opt_obj = self.train_async(
            hyperparameter_bounds,
            init_hyperparameters=self.hyperparameters,
            max_iter=max_iter,
            constraints = constraints,
            dask_client=dask_client,
            local_optimizer=local_method,
            global_optimizer=global_method
        )
        return opt_obj

    ##############################################################
    def train_gp(self,
                 hyperparameter_bounds,
                 max_iter=120,
                 method="global",
                 pop_size=20,
                 tolerance=1e-6,
                 constraints = ()
                 ):
        """
        Function to train a Gaussian Process.

        Parameters
        ----------
        hyperparameters_bounds : np.ndarray
            Bounds for the optimization of the hyperparameters of shape (V x 2)
        max_iter : int, optional
            Number of iterations before the optimization algorithm is terminated. The default is 120
        method : str or callable, optional
            Optimization method. Choose from `'local'` or `'global'`. The default is `global`.
            The argument also accepts a callable that accepts as input a `fvgp.gp.GP` 
            instance and returns a new vector of hyperparameters.
        pop_size : int, optional
            The number of individuals used if `global` is chosen as method.
        tolerance : float, optional
            Tolerance to be used to define a termination criterion for the optimizer.
        constraints: tuple of object instances, optional
            Either a tuple of hgdl.constraints.NonLinearConstraint or scipy constraints instances, depending on the used optimizer.


        Return
        ------
        hyperparameters : np.ndarray
            This is just informative, the Gaussian Process is automatically updated.
        """

        if self.gp_initialized is False:
            raise Exception("No GP to be trained. Please call init_gp(...) before training.")
        self.train(
            hyperparameter_bounds,
            init_hyperparameters=self.hyperparameters,
            method=method,
            pop_size=pop_size,
            constraints = constraints,
            tolerance=tolerance,
            max_iter=max_iter
        )
        return self.hyperparameters

    ##############################################################
    def stop_async_train(self, opt_obj):
        """
        Function to stop an asynchronous training. This leaves the dask.distributed.Client alive.

        Parameters
        ----------
        opt_obj : object instance
            Object instance created by gpcam.gp_optimizer.GPOptimizer.train_gp_async()
        """
        try:
            self.stop_training(opt_obj)
        except:
            pass

    def kill_async_train(self, opt_obj):
        """
        Function to kill an asynchronous training. This shuts down the associated dask.distributed.Client.

        Parameters
        ----------
        opt_obj : object instance
            Object instance created by gpcam.gp_optimizer.GPOptimizer.train_gp_async()
        """
        try:
            self.kill_training(opt_obj)
        except Exception as ex:
            raise RuntimeError("kill not sucessful in GPOptimizer") from ex

    ##############################################################
    def update_hyperparameters(self, opt_obj):
        """
        Function to update the Gaussian Process hyperparameters is an asynchronous training is running.

        Parameters
        ----------
        opt_obj : object instance
            Object instance created by gpcam.gp_optimizer.GPOptimizer.train_gp_async()

        Return
        ------
            hyperparameters : np.ndarray
        """

        hps = GP.update_hyperparameters(self, opt_obj)
        return hps

    ##############################################################
    def ask(self, position=None, n=1,
            acquisition_function="variance",
            bounds=None,
            method="global",
            pop_size=20,
            max_iter=20,
            tol=1e-6,
            constraints = (),
            x0=None,
            dask_client=None):

        """
        Given that the acquisition device is at "position", the function ask()s for
        "n" new optimal points within certain "bounds" and using the optimization setup:
        "acquisition_function_pop_size", "max_iter" and "tol"

        Parameters
        ----------
        position : np.ndarray, optional
            Current position in the input space. If a cost function is provided this position will be taken into account
            to guarantee a cost-efficient new suggestion. The default is None.
        n  : int, optional
            The algorithm will try to return this many suggestions for new measurements. This may be limited by how many
            optima the algorithm may find. If greater than 1, then the `acq_func` optimization method is automatically
            set to use HGDL. The default is 1.
        acquisition_function : Callable, optional
            The acquisition function accepts as input a numpy array of size V x D (such that V is the number of input
            points, and D is the parameter space dimensionality) and a `GPOptimizer` object. The return value is 1-D
            array
            of length V providing 'scores' for each position, such that the highest scored point will be measured next.
            Built-in functions can be used by one of the following keys: `'shannon_ig'`, `'UCB'`, `'maximum'`,
            `'minimum'`,
            `'covariance'`, and `'variance'`. If None, the default function is the `'variance'`, meaning
            `fvgp.gp.GP.posterior_covariance` with variance_only = True.
        bounds : np.ndarray, optional
            A numpy array of floats of shape D x 2 describing the search range. The default is the entire input space.
        method: str, optional
            A string defining the method used to find the maximum of the acquisition function. Choose from `global`,
            `local`, `hgdl`.
            The default is `global`.
        pop_size: int, optional
            An integer defining the number of individuals if `global` is chosen as method. The default is 20. For
            `hgdl` this will be overwritten
            by the 'dask_client` definition.
        max_iter: int, optional
            This number defined the number of iterations before the optimizer is terminated. The default is 20.
        tol: float, optional
            Termination criterion for the local optimizer. The default is 1e-6.
        x0: np.ndarray, optional
            A set of points as numpy array of shape V x D, used as starting location(s) for the local and hgdl
            optimization
            algorithm. The default is None.
        constraints: tuple of object instances, optional
            Either a tuple of hgdl.constraints.NonLinearConstraint or scipy constraints instances, depending on the used optimizer.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed `acquisition_func` computation. If None is provided,
            a new
            `dask.distributed.Client` instance is constructed.

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
        if n > 1: method = "hgdl"
        if bounds is None: bounds = self.input_space_bounds
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
            dask_client=dask_client)
        return {'x': np.array(maxima), "f(x)": np.array(func_evals), "opt_obj": opt_obj}

    ##############################################################
    def init_cost(self, cost_function, cost_function_parameters=None, cost_update_function=None):
        """
        This function initializes the cost function and its parameters.
        If used, the acquisition function will be augmented by the costs which leads to different suggestions.

        Parameters
        ----------
        cost_function : Callable
            A function encoding the cost of motion through the input space and the cost of a measurement. Its inputs
            are an
            `origin` (np.ndarray of size V x D), `x` (np.ndarray of size V x D), and the value of `cost_func_params`;
            `origin` is the starting position, and `x` is the destination position. The return value is a 1-D array of
            length V describing the costs as floats. The 'score' from acquisition_function is divided by this
            returned cost to determine
            the next measurement point.
        cost_function_parameters : object, optional
            This object is transmitted to the cost function; it can be of any type. The default is None.
        cost_update_function : Callable, optional
            If provided this function will be used when `gpcam.gp_optimizer.GPOptimizer.update_cost_function` is called.
            The function `cost_update_function` accepts as input costs (a list of cost values usually determined by
            `instrument_func`) and a parameter
            object. The default is a no-op.
        Return
        ------
            No return, cost function will automatically be used by GPOptimizer.ask()
        """

        self.cost_function = cost_function
        self.cost_function_parameters = cost_function_parameters
        self.cost_update_function = cost_update_function
        return self.cost_function

    ##############################################################
    def update_cost_function(self, measurement_costs):
        """
        This function updates the parameters for the user-defined cost function
        It essentially calls the user-given cost_update_function which
        should return the new parameters how they are used by the
        cost function.
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

        if self.cost_function_parameters is None: raise Exception(
            "No cost function parameters specified. Please call init_cost() first.")
        self.cost_function_parameters = \
            self.cost_update_function(measurement_costs, self.cost_function_parameters)
        #print("cost parameters changed to: ", self.cost_function_parameters)


######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
class fvGPOptimizer(fvGP, GPOptimizer):
    """
    This class is an optimization wrapper around the fvgp package for multi-task (multi-variate) Gaussian Processes.
    Gaussian Processes can be initialized, trained, and conditioned; also
    the posterior can be evaluated and plugged into optimizers to find
    its maxima.

    Parameters
    ---------
    input_space_dimension : int
        Integer specifying the number of dimensions of the input space.
    output_space_dimension : int
        Integer specifying the number of dimensions of the output space. Most often 1.
    output_number : int
        Number of output values.
    input_space_bounds : np.ndarray
        A numpy array of floats of shape D x 2 describing the input space range

    Attributes
    ----------
    points : np.ndarray
        Datapoint positions
    values : np.ndarray
        Datapoint values
    variances : np.ndarray
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
            input_space_dimension,
            output_space_dimension,
            output_number,
            input_space_bounds,
    ):

        self.iput_dim = input_space_dimension
        self.oput_dim = output_space_dimension
        self.output_number = output_number
        self.points = np.empty((0, self.iput_dim))
        self.values = np.empty((0, self.output_number))
        self.variances = np.empty((0, self.output_number))
        self.value_positions = np.empty((0, self.output_number, self.oput_dim))
        self.input_space_bounds = np.array(input_space_bounds)
        self.gp_initialized = False
        self.cost_function_parameters = None
        self.cost_function = None

        GPOptimizer.__init__(self,
                             input_space_dimension,
                             input_space_bounds
                             )

    def get_data_fvGP(self):
        """
        Function that provides a way to access the class attributes.

        Return
        ------
        dictionary of class attributes : dict
        """

        res = self.get_data
        res["output number"] = self.output_number
        res["output dim"] = self.oput_dim
        res["measurement value positions"] = self.value_positions
        return res

    def tell(self, x, y, variances=None, value_positions=None):
        """
        This function can tell() the gp_optimizer class
        the data that was collected. The data will instantly be used to update the gp_data
        if a GP was previously initialized.

        Parameters
        ----------
        x : np.ndarray
            Point positions (of shape U x D) to be communicated to the Gaussian Process.
        y : np.ndarray
            Point values (of shape U x 1 or U) to be communicated to the Gaussian Process.
        variances : np.ndarray, optional
            Point value variances (of shape U x 1 or U) to be communicated to the Gaussian Process.
            If not provided, the GP will 1% of the y values as variances.
        value_positions : np.ndarray, optional
            A 3-D numpy array of shape (U x output_number x output_dim), so that for each measurement position, the outputs
            are clearly defined by their positions in the output space. The default is np.array([[0],[1],[2],[3],...,[output_number - 1]]) for each
            point in the input space. The default is only permissible if output_dim is 1.
        """
        self.points = x
        self.values = y
        self.variances = variances
        self.value_positions = value_positions

        if self.gp_initialized is True: self.update_fvgp()

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
        """
        Function to initialize the multi-task GP.

        Parameters
        ----------
        init_hyperparameters : np.ndarray
            Initial hyperparameters as 1-D numpy array.
        compute_device : str, optional
            One of "cpu" or "gpu", determines how linear system solves are run. The default is "cpu".
        gp_kernel_function : Callable, optional
            A function that calculates the covariance between datapoints. It accepts as input x1 (a V x D array of
            positions),
            x2 (a U x D array of positions), hyperparameters (a 1-D array of length D+1 for the default kernel), and a
            `gpcam.gp_optimizer.GPOptimizer` instance. The default is a stationary anisotropic kernel
            (`fvgp.gp.GP.default_kernel`).
        gp_mean_function : Callable, optional
            A function that evaluates the prior mean at an input position. It accepts as input a
            `gpcam.gp_optimizer.GPOptimizer` instance, an array of positions (of size V x D), and hyperparameters (a 1-D
            array of length D+1 for the default kernel). The return value is a 1-D array of length V. If None is
            provided,
            `fvgp.gp.GP.default_mean_function` is used.
        use_inv : bool, optional
            If True, the algorithm calculates and stores the inverse of the covariance matrix after each training or update of the dataset,
            which makes computing the posterior covariance faster.
            For larger problems (>2000 data points), the use of inversion should be avoided due to computational instability. The default is
            False. Note, the training will always use a linear solve instead of the inverse for stability reasons.
        ram_economy : bool, optional
            Offers a ram-efficient way to compute marginal-log-likelihood derivatives for training.
        """

        if self.gp_initialized is False:
            fvGP.__init__(
                self,
                self.iput_dim,
                self.oput_dim,
                self.output_number,
                self.points,
                self.values,
                init_hyperparameters,
                value_positions=self.value_positions,
                variances=self.variances,
                compute_device=compute_device,
                gp_kernel_function=gp_kernel_function,
                gp_mean_function=gp_mean_function,
                use_inv=use_inv,
                ram_economy=ram_economy
            )
            self.gp_initialized = True
        else:
            raise RuntimeError("No initialization. fvGP already initialized")

    ##############################################################
    def update_fvgp(self):
        """
        This function updates the data in the fvGP, tell(...) will call this function automatically if
        GP is already intialized
        """
        self.update_fvgp_data(
            self.points,
            self.values,
            value_positions=self.value_positions,
            variances=self.variances)


