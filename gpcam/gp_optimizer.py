import numpy as np
from fvgp.fvgp import FVGP
from . import surrogate_model as sm
import time


class GPOptimizer():
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
        output_space_dimension (int):        dim2
        output_number (int):                 n
        index_set_bounds (2d array):         bounds of the index set


    Example:
        obj = GPOptimizer(3,1,2,[[0,10],[0,10],[0,10]])
        obj.tell(x,y)
        obj.init_gp(...)
        obj.train_gp(...) #can be "async_train_gp()"
        obj.init_cost(...)
        obj.update_cost_function(...)
        prediction = obj.gp.posterior_mean(x0)
    ------------------------------------------------------------
    """
##############################################################
    def __init__(
        self,
        input_space_dimension,
        output_space_dimension,
        output_number,
        index_set_bounds,
    ):
        """
        GPOptimizer constructor
        type help(gp_optimizer) for help
        """
        self.iput_dim = input_space_dimension
        self.oput_dim = output_space_dimension
        self.output_number = output_number
        self.points = np.empty((0,self.iput_dim))
        self.values = np.empty((0,self.output_number))
        self.variances = np.empty((0,self.output_number))
        self.value_positions = np.empty((0,self.output_number,self.oput_dim))
        self.index_set_bounds = np.array(index_set_bounds)
        self.hyperparameters = None
        self.gp_initialized = False
        self.cost_function_parameters = None
        self.cost_function = None
        self.consider_costs = False


##############################################################
    def get_data(self):
        """
        this provides a way to see the current class varibles
        the return is a dictionary of class variables
        """
        try:
            res = {"input dim": self.iput_dim,
                "output dim": self.oput_dim,
                "output number": self.output_number,
                "x": self.points,
                "y": self.values,
                "measurement variances":self.variances,
                "measurement value positions":self.value_positions,
                "hyperparameters": self.hyperparameters,
                "cost function parameters": self.cost_function_parameters,
                "consider costs": self.consider_costs,
                }
        except:
            print("Not all data is assigned yet, call tell(...) before asking for the data.")
            res = {}
        return res

##############################################################
    def evaluate_objective_function(self, x, objective_function = "covariance", origin = None):
        """
        function that evaluates the objective function
        input:
            x: 1d numpy array
            objective_function: "covariance","shannon_ig",..., or callable, use the same you use in ask()
            origin = None
        returns:
            scalar (float) or array
        """
        if self.gp_initialized is False: raise Exception("Initialize GP before evaluating the objective function. see help(gp_init)")
        x = np.array(x)
        try:
            return sm.evaluate_objective_function(x, self.gp, objective_function,
                origin, self.cost_function, self.cost_function_parameters)
        except Exception as a:
            print("Evaluating the objective function was not successful.")
            print("Error Message:")
            print(str(a))

##############################################################
    def tell(self, x, y,
            variances = None,
            value_positions = None,
            append = False,
            ):
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
                                                      value_posiitons = np.array([
                                                      [[0,1],[2,3],[4,5]]
                                                      [[0,2],[4,2],[7,8]]
                                                      ])
            append:                             default = False, True/False, append data or rewrite it

        Returns:
        --------
            no returns
        """
        ######create the current data
        if len(x) != len(y): raise Exception("Length of x and y has to be the same!")
        if append is True and variances is not None and value_positions is not None:
            if len(x) != len(value_positions): raise Exception("Length of value positions is not correct!")
            if y.shape != variance.shape: raise Exception("Shape of variance array not correct!")
            self.points = np.vstack([self.points,x])
            self.values = np.vstack([self.values,y])
            self.variances = np.vstack([self.variances,variances])
            self.value_positions = np.vstack([self.value_positions,value_positions])
        else:
            self.points = x
            self.values = y
            self.variances = variances
            self.value_positions = value_positions
        if self.gp_initialized is True: self.update_gp()

##############################################################
    def init_gp(self,init_hyperparameters, compute_device = "cpu",gp_kernel_function = None,
            gp_mean_function = None, sparse = False):
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
            self.gp = FVGP(
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
            sparse = sparse,
            )
            self.gp_initialized = True
            self.hyperparameters = np.array(init_hyperparameters)

##############################################################
    def update_gp(self):
        """
        This function updates the data in the GP, tell(...) will call this function automatically if
        GP is intialized
        Paramaters:
        -----------
            no input parameters
        """
        self.gp.update_gp_data(
            self.points,
            self.values,
            value_positions = self.value_positions,
            variances = self.variances)

##############################################################
    def async_train_gp(self, hyperparameter_bounds,
            likelihood_optimization_pop_size = 20,
            likelihood_optimization_tolerance = 1e-6,
            likelihood_optimization_max_iter = 10000,
            dask_client = True):
        """
        Function to start fvGP asynchronous training.
        Parameters:
        -----------
            hyperparameter_bounds:                  2d np.array of bounds for the hyperparameters
        Optional Parameters:
        --------------------
            likelihood_optimization_pop_size:       number of walkers in the optimization, default = 20
            likelihood_optimization_tolerance:      tolerance for termination, default = 1e-6
            likelihood_optimization_max_iter:       maximum number of iterations, default = 10000
            dask_client:                            a DASK client, see dask package docs for explanation
        """
        if self.gp_initialized is False:
            raise Exception("No GP to be trained. Please call init_gp(...) before training.")
        self.gp.train(
                hyperparameter_bounds,
                init_hyperparameters = self.hyperparameters,
                optimization_method = "hgdl",
                optimization_pop_size = likelihood_optimization_pop_size,
                optimization_tolerance = likelihood_optimization_tolerance,
                optimization_max_iter = likelihood_optimization_max_iter,
                dask_client = dask_client
                )
        self.hyperparameters = np.array(self.gp.hyperparameters)
        return self.hyperparameters

##############################################################
    def train_gp(self,hyperparameter_bounds,
            likelihood_optimization_method = "global",likelihood_optimization_pop_size = 20,
            likelihood_optimization_tolerance = 1e-6,likelihood_optimization_max_iter = 120):
        """
        Function to perform fvGP training.
        Parameters:
        -----------
            hyperparameter_bounds:                  2d np.array of bounds for the hyperparameters
        Optional Parameters:
        --------------------
            likelihood_optimization_method:         "hgdl"/"global"/"local", default = "global"
            likelihood_optimization_pop_size:       number of walkers in the optimization, default = 20
            likelihood_optimization_tolerance:      tolerance for termination, default = 1e-6
            likelihood_optimization_max_iter:       maximum number of iterations, default = 120
        """

        if self.gp_initialized is False:
            raise Exception("No GP to be trained. Please call init_gp(...) before training.")
        self.gp.train(
                hyperparameter_bounds,
                init_hyperparameters = self.hyperparameters,
                optimization_method = likelihood_optimization_method,
                optimization_pop_size = likelihood_optimization_pop_size,
                optimization_tolerance = likelihood_optimization_tolerance,
                optimization_max_iter = likelihood_optimization_max_iter,
                dask_client = False
                )
        self.hyperparameters = np.array(self.gp.hyperparameters)
        return self.hyperparameters

##############################################################
    def stop_async_train(self):
        """
        function to stop vfGP async training
        Parameters:
        -----------
            no input parameters
        """
        self.gp.stop_training()

##############################################################
    def update_hyperparameters(self):
        self.gp.update_hyperparameters()
        self.hyperparameters = np.array(self.gp.hyperparameters)
        return self.hyperparameters

##############################################################
    def ask(self, position = None, n = 1,
            objective_function = "covariance",
            optimization_bounds = None,
            optimization_method = "global",
            optimization_pop_size = 20,
            optimization_max_iter = 20,
            optimization_tol = 10e-6,
            dask_client = False):
        """
        Given that the acquisition device is at "position", the function ask() s for
        "n" new optimal points within certain "bounds" and using the optimization setup:
        "objective_function_pop_size", "max_iter" and "tol"
        Parameters:
        -----------

        Optional Parameters:
        --------------------
            position (numpy array):            last measured point, default = None
            n (int):                           how many new measurements are requested, default = 1
            objective_function:                default = None, means that the class objective function will be used
            optimization_bounds (2d list/None):             default = None
            optimization_method:                            default = "global", "global"/"hgdl"
            optimization_pop_size (int):                    default = 20
            optimization_max_iter (int):                    default = 20
            optimization_tol (float):                       default = 10e-6
            dask_client:                                    default = False
        """
        print("aks() initiated with hyperparameters:",self.hyperparameters)
        print("optimization method: ", optimization_method)
        print("bounds: ",optimization_bounds)
        if optimization_bounds is None: optimization_bounds = self.index_set_bounds
        maxima,func_evals = sm.find_objective_function_maxima(
                self.gp,
                objective_function,
                position,n, optimization_bounds,
                optimization_method = optimization_method,
                optimization_pop_size = optimization_pop_size,
                optimization_max_iter = optimization_max_iter,
                optimization_tol = optimization_tol,
                cost_function = self.cost_function,
                cost_function_parameters = self.cost_function_parameters,
                dask_client = dask_client)
        return {'x':np.array(maxima), "f(x)" : np.array(func_evals)}

##############################################################
    def init_cost(self,cost_function,cost_function_parameters,
            cost_update_function = None, cost_function_optimization_bounds = None):
        """
        This function initializes the costs. If used, the objective function will be augmented by the costs
        which leads to different suggestions

        Parameters:
        -----------
            cost_function: callable
            cost_function_parameters: arbitrary, are passed to the user defined cost function

        Optional Parameters:
        --------------------
            cost_update_function: a function that updates the cost_fucntion_parameters, default = None
            cost_function_optimization_bounds: optimization bounds for the update, default = None
        Return:
        -------
            no returns
        """

        self.cost_function = cost_function
        self.cost_function_parameters = cost_function_parameters
        self.cost_function_optimization_bounds = cost_function_optimization_bounds
        self.cost_update_function = cost_update_function
        self.consider_costs = True
        print("Costs successfully initialized")

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
                                  cost_fucntion_optimization_bounds,cost_function_parameters)
                                  which returns the new parameters
            cost_function_optimization_bounds: see above
        Optional Parameters:
        --------------------
            cost_function_parameters, default = None
        """

        print("Performing cost function update...")
        if self.cost_function_parameters is None: raise Exception("No cost function parameters specified. Please call init_cost() first.")
        self.cost_function_parameters = \
        self.cost_update_function(measurement_costs,
        self.cost_function_optimization_bounds,
        self.cost_function_parameters)
        print("cost parameters changed to: ", self.cost_function_parameters)
######################################################################################
######################################################################################
######################################################################################
