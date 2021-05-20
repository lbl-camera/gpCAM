#!/usr/bin/env python

import numpy as np
from fvgp.fvgp import fvGP
from gpcam import surrogate_model as sm
from fvgp.gp import GP

class GPOptimizer(GP):
    """
    GPOptimizer class: Given data, this class can determine which
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
        index_set_bounds (2d array):         bounds of the index set


    Example:
        obj = GPOptimizer(3,[[0,10],[0,10],[0,10]])
        obj.tell(x,y)
        obj.init_gp(...)
        obj.train_gp(...) #can be "train_gp_async()"
        co = obj.init_cost(...)
        ask(cost_function = co)
        obj.update_cost_function(...)
        prediction = obj.posterior_mean(x0)
    ------------------------------------------------------------
    """

    def __init__(
        self,
        input_space_dimension,
        index_set_bounds,
    ):
        """
        GPOptimizer constructor
        type help(gp_optimizer) for help
        """
        self.iput_dim = input_space_dimension
        self.points = np.empty((0, self.iput_dim))
        self.values = np.empty((0))
        self.variances = np.empty((0))
        self.index_set_bounds = np.array(index_set_bounds)
        self.hyperparameters = None
        self.gp_initialized = False
        self.cost_function_parameters = None
        self.cost_function = None
        self.consider_costs = False

    def get_data(self):
        """
        Provides a way to access the current class varibles.

        Returns
        -------
        dict
            Dictionary containing the input dim, output dim, output number,
            x & y data, measurement variances, measurement value positions,
            hyperparameters, cost function parameters and consider costs
            class attributes. Note that if tell() has not been called, many
            of these returned values will be None.
        """

        return {
            "input dim": self.iput_dim,
            "x": self.points,
            "y": self.values,
            "measurement variances": self.variances,
            "hyperparameters": self.hyperparameters,
            "cost function parameters": self.cost_function_parameters,
            "consider costs": self.consider_costs,
        }

    def evaluate_acquisition_function(self,
        x, acquisition_function="covariance", cost_function=None,
        origin=None):
        """
        Evaluates the acquisition function.

        Parameters:
        -----------
        x: 1d numpy array.

        Optional Parameters:
        --------------------
        acquisition_function : default = "covariance",
                               "covariance","shannon_ig" ,..., or callable, use the same you use
                               in ask(). (The default is "covariance").
        origin:                default = None, only important for cost considerations

        Returns
        -------
        float or numpy array
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
        the data that was collected. The data will instantly be use to update the gp_data
        if a GP was previously initialized

        Parameters:
        -----------
            x (2d numpy array):                A 2d array of all the points in the data
            values (2d numpy array):           A 2d array of all the values measured at the associated points

        Optional Parameters:
        --------------------
            variances (2d numpy array):         A 2d array of all the variances of the measured points

        Returns:
        --------
            no returns
        """
        self.points = x
        self.values = y
        self.variances = variances

        if self.gp_initialized is True:
            self.update_gp()

##############################################################
    def init_gp(
        self,
        init_hyperparameters,
        compute_device="cpu",
        gp_kernel_function=None,
        gp_mean_function=None,
        sparse=False
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
            sparse = sparse,
            normalize_y = False
            )
            self.gp_initialized = True
            self.hyperparameters = np.array(init_hyperparameters)
            print("GP successfully initiated")

##############################################################
    def update_gp(self):
        """
        This function updates the data in the GP, tell(...) will call this function automatically if
        GP is intialized
        Paramaters:
        -----------
            no input parameters
        """
        self.update_gp_data(
            self.points,
            self.values,
            variances = self.variances)

##############################################################
    def train_gp_async(self, hyperparameter_bounds,
            pop_size = 20,
            tolerance = 1e-6,
            max_iter = 10000,
            dask_client = None):
        """
        Function to start fvGP asynchronous training.
        Parameters:
        -----------
            hyperparameter_bounds:                  2d np.array of bounds for the hyperparameters
        Optional Parameters:
        --------------------
            pop_size:       number of walkers in the optimization, default = 20
            tolerance:      tolerance for termination, default = 1e-6
            max_iter:       maximum number of iterations, default = 10000
            dask_client:                            a DASK client, see dask package docs for explanation
        Return:
            Nothing, call update_hyperparameters() for the result
        """
        if self.gp_initialized is False:
            raise Exception("No GP to be trained. Please call init_gp(...) before training.")
        self.train_async(
                hyperparameter_bounds,
                init_hyperparameters = self.hyperparameters,
                pop_size = pop_size,
                tolerance = tolerance,
                max_iter = max_iter,
                dask_client = dask_client
                )

##############################################################
    def train_gp(self,hyperparameter_bounds,
            method = "global",pop_size = 20,
            optimization_dict = None,tolerance = 1e-6,
            max_iter = 120):
        """
        Function to perform fvGP training.
        Parameters:
        -----------
            hyperparameter_bounds:                  2d np.array of bounds for the hyperparameters
        Optional Parameters:
        --------------------
            method:         "hgdl"/"global"/"local", default = "global"
            pop_size:       number of walkers in the optimization, default = 20
            optimization_dict:                      default = None
            tolerance:      tolerance for termination, default = 1e-6
            max_iter:       maximum number of iterations, default = 120
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
    def stop_async_train(self):
        """
        function to stop vfGP async training
        Parameters:
        -----------
            no input parameters
        """
        try: self.stop_training()
        except: pass

##############################################################
    def update_hyperparameters(self):
        self.hyperparameters = GP.update_hyperparameters(self)
        print("GPOptimizer updated the Hyperperameters: ", self.hyperparameters)
        return self.hyperparameters

##############################################################
    def ask(self, position = None, n = 1,
            acquisition_function = "covariance",
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
        print("aks() initiated with hyperparameters:",self.hyperparameters)
        print("optimization method: ", method)
        print("bounds: ",bounds)
        if bounds is None: bounds = self.index_set_bounds
        maxima,func_evals = sm.find_acquisition_function_maxima(
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
        return {'x':np.array(maxima), "f(x)" : np.array(func_evals)}

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
