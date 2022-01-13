#!/usr/bin/env python

import numpy as np
from fvgp.fvgp import fvGP
from gpcam import surrogate_model as sm
from fvgp.gp import GP



        """
        Function to start the autonomous-data-acquisition loop.
        
        Parameters
        ----------
        N : int, optional
            Run for N iterations. The default is `1e15`.
        breaking_error : float, optional
            Run until breaking_error is achieved (or at max N). The default is `1e-15`.
        retrain_globally_at : Iterable[int], optional
            Retrains the hyperparameters at the given number of measurements using global optimization. The deafult is
            `[100,400,1000]`.
        retrain_locally_at : Iterable[int], optional
            Retrains the hyperparameters at the given number of measurements using local gradient-based optimization.
            The default is `[20,40,60,80,100,200,400,1000]`.
        retrain_async_at : Iterable[int], optional
            Retrains the hyperparameters at the given number of measurements using the HGDL algorithm. This training is
            asynchronous and can be run in a distributed fashion using `training_dask_client`. The default is `[1000,2000,5000,10000]`.
        retrain_callable_at : Iterable[int], optional
            Retrains the hyperparameters at the given number of measurements using a callable provided by `training_opt_callable`.
        update_cost_func_at : Iterable[int], optional
            Calls the `update_cost_func` at the given number of measurements.
        acq_func_opt_setting : Callable, optional
            A callable that accepts as input the iteration index and returns either `'local'` or `'global'`. This
            switches between local gradient-based and global optimization for the acquisition function.
        training_opt_callable : Callable, optional
            A callable that accepts as input a `fvgp.gp.GP` instance and returns a new vector of hyperparameters.
        training_opt_max_iter : int, optional
            The maximum number of iterations for any training.
        training_opt_pop_size : int, optional
            The population size used for any training with a global component (HGDL or standard global optimizers).
        training_opt_tol : float, optional
            The optimization tolerance for all training optimization.
        acq_func_opt_max_iter : int, optional
            The maximum number of iterations for the `acq_func` optimization.
        acq_func_opt_pop_size : int, optional
            The population size used for any `acq_func` optimization with a global component (HGDL or standard global
            optimizers).
        acq_func_opt_tol : float, optional
            The optimization tolerance for all `acq_func` optimization.
        acq_func_opt_tol_adjust : float, optional
            The `acq_func` optimization tolerance is adjusted at every iteration as a fraction of this value.
        number_of_suggested_measurements : int, optional
            The algorithm will try to return this many suggestions for new measurements. This may be limited by how many
            optima the algorithm may find. If greater than 1, then the `acq_func` optimization method is automatically
            set to use HGDL. The default is 1.
        """


class GPOptimizer(GP):
    """
    This class is an optimization wrapper around the fvgp package.
    Gaussian Processes can be intialized, trained and conditioned here; also
    the posterior can be evaluated and plugged into optimizers to find
    a its maxima.

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
        Only available after a training is executed.
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
        self.consider_costs = False

    def get_data(self):
        """
        Function that provides a way to access the class attributes.

        Return
        ------
        dict
            Dictionary containing the input dim, output dim, output number,
            x & y data, measurement variances, measurement value positions,
            hyperparameters, cost function parameters and consider costs
            class attributes. Note that if tell() has not been called, many
            of these returned values will be `None`.
        """

        if self.gp_initialized: hps = self.hyperparameters
        else: hps = None
        return {
            "input dim": self.iput_dim,
            "x": self.points,
            "y": self.values,
            "measurement variances": self.variances,
            "hyperparameters": hps,
            "cost function parameters": self.cost_function_parameters,
            "consider costs": self.consider_costs,
            }

    def evaluate_acquisition_function(self,
        x, acquisition_function="variance", cost_function=None,
        origin=None):
        """
        Function to evaluate the acquisition function.

        Parameters
        ----------
        x : np.ndarray
            Point positions at which the acquisition function is evaluated
        acquisition_function : Callable, optional
            Acquisiiton functio to execute. Callable with inputs (x,gpcam.gp_optimizer.GPOptimizer),
            where x is a V x D array of input points. The return value is a 1-D array of length V.
            The default is `variance`.
        cost_function : Callable, optional
            Function to specify the cost of motion through the input space. If provided the acquisition function evaluation
            will be divided by the cost. Its inputs are an
            `origin` (np.ndarray of size V x D), `x` (np.ndarray of size V x D), and the value of `cost_func_params`;
            `origin` is the starting position, and `x` is the destination position. The return value is a 1-D array of
            length V describing the costs as floats.
        origin : np.ndarray, optional
            If a cost function is provided this 1-D numpy array of length D is used as the origin of motion.

        Returns
        -------
            np.ndarray
            The acquisition function evaluations at all points `x`.
        """

        if self.gp_initialized is False:
            raise Exception(
                "Initialize GP before evaluating the acquisition function. "
                "See help(gp_init)."
            )
        x = np.array(x)
        try:
            return sm.evaluate_acquisition_function(
                x, self, acquisition_function, origin, cost_function,
                self.cost_function_parameters)
        except Exception as a:
            print("Evaluating the acquisition function was not successful.")
            print("Error Message:")
            print(str(a))

    def tell(self, x, y, variances=None):
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
        """
        self.points = x
        self.values = y
        self.variances = variances
        print("New data communicated via tell()")

        if self.gp_initialized is True: self._update_gp()

##############################################################
    def init_gp(
        self,
        init_hyperparameters,
        compute_device="cpu",
        gp_kernel_function=None,
        gp_mean_function=None,
        sparse=False, use_inv = False,
        ram_economy = True
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
            A function that calculates the covariance between datapoints. It accepts as input x1 (a V x D array of positions),
            x2 (a U x D array of positions), hyperparameters (a 1-D array of length D+1 for the default kernel), and a
            `gpcam.gp_optimizer.GPOptimizer` instance. The default is a stationary anisotropic kernel
            (`fvgp.gp.GP.default_kernel`).
        gp_mean_function : Callable, optional
            A function that evaluates the prior mean at an input position. It accepts as input a
            `gpcam.gp_optimizer.GPOptimizer` instance, an array of positions (of size V x D), and hyperparameters (a 1-D
            array of length D+1 for the default kernel). The return value is a 1-D array of length V. If None is provided,
            `fvgp.gp.GP.default_mean_function` is used.
        sparse : bool, optional
            If True, the algorithm check for sparsity of the covariance matrix and exploits it. The default is False.
        use_inv : bool, optional
            If True, the algorithm retains the inverse of the covariance matrix, which makes computing the posterior faster.
            For larger problems, this use of inversion should be avoided due to computational stability. The default is `False`.
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
            variances = self.variances,
            compute_device = compute_device,
            gp_kernel_function = gp_kernel_function,
            gp_mean_function = gp_mean_function,
            sparse = sparse, use_inv = use_inv,
            normalize_y = False,
            ram_economy = ram_economy
            )
            self.gp_initialized = True
            print("GP successfully initiated")
        else: print("GP already initialized")

##############################################################
    def _update_gp(self):
        """
        This function updates the data in the GP, tell(...) will call this function automatically if
        GP is intialized
        """
        self.update_gp_data(
            self.points,
            self.values,
            variances = self.variances)
        print("GP data updated")

##############################################################
    def train_gp_async(self,
            hyperparameter_bounds,
            max_iter = 10000,
            dask_client = None,
            local_method = "L-BFGS-B",
            global_method = "genetic"):
        """
        Function to train a Gaussian Process asynchronously on distributed HPC compute architecture using the `HGDL` software package.

        Parameters
        ----------
        hyperparameters_bounds : np.ndarray
            Bounds for the optimization of the hyperparameters of shape (V x 2)
        max_iter : int, optional
            Number of iterations before the optimization algorithm is terminated. Since the algorithms works asynchronously, this
            number can be high. The default is 10000
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed training. If None is provided, a local
            `dask.distributed.Client` instance is constructed.
        local_method : str, optional
            Controls the local optimizer running in `HGDL`. Many scipy.minimize optimizers can be used,
            in addition "dNewton". `L-BFGS-B` is the default.
        global_method : str, optional
            Gloabl optimization step running in `HGDL`. Choose from `genetic` or 'random'.
            The default is `genetic`

        Returns
        -------
            This function return an optimization object (opt_obj) that can be used to stop(opt_obj) or kill(opt_obj) the optimization.
            Call `gpcam.gp_optimizer.GPOptimizer.update_hyperparameters(opt_obj)` to update the
            current Gaussian Process with the new hyperparameters. This allows to start several optimization procedures
            and selectively use or stop them.
        """

        print("GPOptimizer async training was called with dask_client: ", dask_client)
        if self.gp_initialized is False:
            raise Exception("No GP to be trained. Please call init_gp(...) before training.")
        opt_obj = self.train_async(
                hyperparameter_bounds,
                init_hyperparameters = self.hyperparameters,
                max_iter = max_iter,
                dask_client = dask_client,
                local_optimizer = local_method,
                global_optimizer = global_method
                )
        print("The GPOptimizer has created an optimization object.")
        return opt_obj

##############################################################
    def train_gp(self,
            hyperparameter_bounds,
            max_iter = 120
            method = "global",
            pop_size = 20,
            optimization_dict = None,
            tolerance = 1e-6,
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
            Optimization method. Choose from `local`, `global` or `hgdl`. The default is `global`.
            The argument also accepts a callable that accepts as input a `fvgp.gp.GP` 
            instance and returns a new vector of hyperparameters.
        pop_size : int, optional
            The number of individuals used if `global` is chosen as method.
        optimization_dict : dict, optional
            A dictionary that will be passed to the callable that is specified as `method`.
        tolerance : float, optional
            Tolerance to be used to define a termination criterion for the optimizer.

        Returns
        -------
        hyperparameters : np.ndarray
            This is just informative, the Gaussian Process is automatically updated.
        """

        if self.gp_initialized is False:
            raise Exception("No GP to be trained. Please call init_gp(...) before training.")
        self.train(
                hyperparameter_bounds,
                init_hyperparameters = self.hyperparameters,
                method = method,
                optimization_dict = optimization_dict,
                pop_size = pop_size,
                tolerance = tolerance,
                max_iter = max_iter
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
        try: self.stop_training(opt_obj)
        except: pass

    def kill_async_train(self, opt_obj):
        """
        Function to kill an asynchronous training. This shuts down the associated dask.distributed.Client.

        Parameters
        ----------
        opt_obj : object instance
            Object instance created by gpcam.gp_optimizer.GPOptimizer.train_gp_async()
        """
        try: self.kill_training(opt_obj)
        except Exception as e: print("kill not sucessful in GPOptimizer due to: ",str(e))

##############################################################
    def update_hyperparameters(self, opt_obj):
        """
        Function to update the Gaussian Process hyperparameters is an asynchronous training is running.

        Parameters
        ----------
        opt_obj : object instance
            Object instance created by gpcam.gp_optimizer.GPOptimizer.train_gp_async()

        Returns
        -------
            hyperparameters : np.ndarray
        """

        hps = GP.update_hyperparameters(self, opt_obj)
        print("The GPOptimizer updated the Hyperperameters: ", self.hyperparameters)
        return hps

##############################################################
    def ask(self, position = None, n = 1,
            acquisition_function = "variance",
            cost_function = None,
            bounds = None,
            method = "global",
            pop_size = 20,
            max_iter = 20,
            tol = 10e-6,
            x0 = None,
            dask_client = False):

        """
        Given that the acquisition device is at "position", the function ask() s for
        "n" new optimal points within certain "bounds" and using the optimization setup:
        "acquisition_function_pop_size", "max_iter" and "tol"

        Parameters
        ----------
        position : np.ndarray, optional
            Current position in th einput space. If a cost function is provided this position will be taken into account
            to guarantee an cost-efficient new suggestion. The default is None.
        n  : int, optional
            The algorithm will try to return this many suggestions for new measurements. This may be limited by how many
            optima the algorithm may find. If greater than 1, then the `acq_func` optimization method is automatically
            set to use HGDL. The default is 1.
        acquisition_function : Callable, optional
            The acquisition function accepts as input a numpy array of size V x D (such that V is the number of input
            points, and D is the parameter space dimensionality) and a `GPOptimizer` object. The return value is 1-D array
            of length V providing 'scores' for each position, such that the highest scored point will be measured next.
            Built-in functions can be used by one of the following keys: `'shannon_ig'`, `'UCB'`, `'maximum'`, `'minimum'`,
            `'covariance'`, and `'variance'`. If None, the default function is the `'variance'`, meaning
            `fvgp.gp.GP.posterior_covariance` with variance_only = True.
        cost_function : Callable, optional
            A function encoding the cost of motion through the input space and the cost of a measurement. Its inputs are an
            `origin` (np.ndarray of size V x D), `x` (np.ndarray of size V x D), and the value of `cost_func_params`;
            `origin` is the starting position, and `x` is the destination position. The return value is a 1-D array of
            length V describing the costs as floats. The 'score' from acq_func is divided by this returned cost to determine
            the next measurement point. If None, the default is a uniform cost of 1.
        bounds : np.ndarray, optional
            A numpy array of floats of shape D x 2 describing the search range. The default is the entire input space.
        method:
        pop_size:
        max_iter:
        tol:
        x0:
        dask_client:

        Returns
        -------
            {'x': np.array(maxima), "f(x)" : np.array(func_evals), "opt_obj" : opt_obj}
        """



        """
        Parameters:
        -----------

        Optional Parameters:
        --------------------
            position (numpy array):            last measured point, default = None
            n (int):                           how many new measurements are requested, default = 1
            acquisition_function:              default = None, means that the class acquisition function will be used
            cost_function:                     default = None, otherwise cost objective received from init_cost, or callable
            bounds (2d list/None):             default = None
            method:                            default = "global", "global"/"hgdl"
            pop_size (int):                    default = 20
            max_iter (int):                    default = 20
            tol (float):                       default = 10e-6
            x0:                                default = None, starting positions for optimizer
            dask_client:                                    default = False
        """
        print("ask() initiated with hyperparameters:",self.hyperparameters)
        print("optimization method: ", method)
        print("bounds: ",bounds)
        print("acq func: ",acquisition_function)
        if bounds is None: bounds = self.input_space_bounds
        maxima,func_evals,opt_obj = sm.find_acquisition_function_maxima(
                self,
                acquisition_function,
                position,n, bounds,
                optimization_method = method,
                optimization_pop_size = pop_size,
                optimization_max_iter = max_iter,
                optimization_tol = tol,
                cost_function = cost_function,
                cost_function_parameters = self.cost_function_parameters,
                optimization_x0 = x0,
                dask_client = dask_client)
        return {'x':np.array(maxima), "f(x)" : np.array(func_evals), "opt_obj" : opt_obj}

##############################################################
    def init_cost(self,cost_function,cost_function_parameters,cost_update_function = None):
        """
        This function initializes the costs. If used, the acquisition function will be augmented by the costs
        which leads to different suggestions

        Parameters:
        -----------
            cost_function: callable
            cost_function_parameters: arbitrary, are passed to the user defined cost function

        Optional Parameters:
        --------------------
            cost_update_function: a function that updates the cost_function_parameters, default = None
            cost_function_optimization_bounds: optimization bounds for the update, default = None

        Return:
        -------
            cost function that can be injected into ask()
        """

        self.cost_function = cost_function
        self.cost_function_parameters = cost_function_parameters
        self.cost_update_function = cost_update_function
        self.consider_costs = True
        print("Costs successfully initialized")
        return self.cost_function

##############################################################
    def update_cost_function(self,measurement_costs):
        """
        This function updates the parameters for the cost function
        It essentially calls the user-given cost_update_function which
        should return the new parameters how they are used by the user defined
        cost function
        Parameters:
        -----------
            measurement_costs:    an arbitrary structure that describes 
                                  the costs when moving in the parameter space
            cost_update_function: a user-defined function 
                                  def name(measurement_costs,
                                  cost_function_optimization_bounds,cost_function_parameters)
                                  which returns the new parameters
            cost_function_optimization_bounds: see above
        Optional Parameters:
        --------------------
            cost_function_parameters, default = None
        """

        print("Performing cost function update...")
        if self.cost_function_parameters is None: raise Exception("No cost function parameters specified. Please call init_cost() first.")
        self.cost_function_parameters = \
        self.cost_update_function(measurement_costs, self.cost_function_parameters)
        print("cost parameters changed to: ", self.cost_function_parameters)
######################################################################################
######################################################################################
######################################################################################
class fvGPOptimizer(fvGP, GPOptimizer):
    """
    fvGPOptimizer class: Given data, this class can determine which
    data should be collected next.
    Initialize and then use tell() to communicate data.
    Use init_gp() to initlaize a GP.
    Use ask() to ask for the optimal next point

    symbols:
        N: Number of points in the data set
        n: number of return values
        dim1: number of dimension of the input space
        dim2: number of dimension of the output space

    Attributes:
        input_space_dimension (int):         dim1
        output_space_dimension (int):        dim2
        output_number (int):                 n
        input_space_bounds (2d array):         bounds of the index set


    Example:
        obj = fvGPOptimizer(3,1,2,[[0,10],[0,10],[0,10]])
        obj.tell(x,y)
        obj.init_gp(...)
        obj.train_gp(...) #can be "train_gp_async()"
        obj.init_cost(...)
        obj.update_cost_function(...)
        prediction = obj.gp.posterior_mean(x0)
    ------------------------------------------------------------
    """

    def __init__(
        self,
        input_space_dimension,
        output_space_dimension,
        output_number,
        input_space_bounds,
    ):
        """
        GPOptimizer constructor
        type help(gp_optimizer) for help
        """

        self.iput_dim = input_space_dimension
        self.oput_dim = output_space_dimension
        self.output_number = output_number
        self.points = np.empty((0, self.iput_dim))
        self.values = np.empty((0, self.output_number))
        self.variances = np.empty((0, self.output_number))
        self.value_positions = np.empty((0, self.output_number, self.oput_dim))
        self.input_space_bounds = np.array(input_space_bounds)
        #self.hyperparameters = None
        self.gp_initialized = False
        self.cost_function_parameters = None
        self.cost_function = None
        self.consider_costs = False
        GPOptimizer.__init__(self,
                input_space_dimension,
                input_space_bounds
                )

    def get_data_fvGP(self):
        """
        Provides a way to access the current class variables.

        Returns
        -------
        dict
            Dictionary containing the input dim, output dim, output number,
            x & y data, measurement variances, measurement value positions,
            hyperparameters, cost function parameters and consider costs
            class attributes. Note that if tell() has not been called, many
            of these returned values will be None.
        """

        res = self.get_data
        res["output number"] = self.output_number
        res["output dim"] = self.oput_dim
        res["measurement value positions"] =  self.value_positions
        return res


    def tell(self, x, y, variances=None, value_positions=None):
        """
        This function can tell() the gp_optimizer class
        the data that was collected. The data will instantly be use to update the gp_data
        if a GP was previously initialized

        Parameters:
        -----------
            x (2d numpy array):                A 2d array of all the points in the data
            values (2d numpy array):           A 2d array of all the values measured at the associated points

        Optional Parameters:
        --------------------
            variances (2d numpy array):         A 2d array of all the variances of the measured points
            value_positions (3d numpy array):   A 3d numpy array that stores a 
                                                2d array of locations for each each data point
                                                    e.g. 
                                                    * 2 data points with 2 ouputs in 1d:
                                                      value_posiitons = np.array([
                                                      [[0],[1]]
                                                      [[0],[1]]
                                                      ])
                                                    * 2 data points with 3 ouputs in 2d:
                                                      value_positions = np.array([
                                                      [[0,1],[2,3],[4,5]]
                                                      [[0,2],[4,2],[7,8]]
                                                      ])

        Returns:
        --------
            no returns
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
        sparse=False, use_inv = False,
        ram_economy = True
    ):
        """
        Function to initialize the GP if it has not already been initialized
        Parameters:
        -----------
            init_hyperparameters:   1d numpy array containing the initial guesses for the hyperparemeters
        Optional Parameters:
        --------------------
            compute_device, default = cpu, others = "gpu"
            gp_kernel_function, default = fvGP default
            gp_mean_function, default = fvGP default, i.e. average of data
            sparse, default = False
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
            value_positions = self.value_positions,
            variances = self.variances,
            compute_device = compute_device,
            gp_kernel_function = gp_kernel_function,
            gp_mean_function = gp_mean_function,
            sparse = sparse, use_inv = use_inv,
            ram_economy = ram_economy
            )
            self.gp_initialized = True
        else: print("fvGP already initialized")

##############################################################
    def update_fvgp(self):
        """
        This function updates the data in the fvGP, tell(...) will call this function automatically if
        GP is already intialized
        Paramaters:
        -----------
            no input parameters
        """
        self.update_fvgp_data(
            self.points,
            self.values,
            value_positions = self.value_positions,
            variances = self.variances)

