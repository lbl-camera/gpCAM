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
        index_set_bounds (2d list):          bounds of the index set
        hyperparameter_bounds (2d list):     list or 2d numpy array of bounds


    Example:
        obj = GPOptimizer(3,1,2,[[0,10],[0,10],[0,10]])
        obj.tell(x,y)
        obj.init_gp(...)
        obj.train_gp(...) #can be an async_train_gp()
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
        self.index_set_bounds = index_set_bounds
        self.hyperparameters = None
        self.gp_initialized = False
        self.cost_function_parameters = None


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
                }
        except:
            print("Not all data is assigned yet, call tell(...) before asking for the data.")
            res = {}
        return res

##############################################################
    def evaluate_objective_function(self, x, objective_function,origin = None,
            cost_function = None,
            cost_function_parameters = None):
        """
        function that evaluates the objective function
        input:
            x
            objective_function
            origin = None
            cost_function = None
            cost_function_parameters = None (the class variable will be used)
        returns:
            scalar (float)
        """
        if gp_initialized is False: raise Exception("Initialize GP before evaluating the objective function")
        if cost_function_parameters is None and self.cost_function_parameters is not None:
            cost_function_parameters = self.cost_function_parameters
        try:
            return sm.evaluate_objective_function(x, self.gp, objective_function,
                origin, cost_function, cost_function_parameters)
        except:
            print("Evaluating the objective function was not successful.")

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
                                            2d array of locations for each each point
            append:                             default = False, True/False, append data or rewrite it

        Returns:
        --------
            no returns
        """
        ######create the current data
        if append is True and variances is not None and value_positions is not None:
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
    def update_cost_function(self,
            measurement_costs,
            cost_update_function,
            cost_function_optimization_bounds,
            cost_function_parameters = None
            ):
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
        if cost_function_parameters is None: cost_function_parameters = self.cost_function_parameters
        self.cost_function_parameters = \
        cost_update_function(measurement_costs,
        cost_function_optimization_bounds,
        cost_function_parameters)
        print("cost parameters changed to: ", self.cost_function_parameters)

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
            self.hyperparameters = init_hyperparameters

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
            likelihood_optimization_pop_size,
            likelihood_optimization_tolerance,
            likelihood_optimization_max_iter,
            dask_client):
        """
        Function to start fvGP asynchronous training.
        Parameters:
        -----------
            hyperparameter_bounds:                  2d np.array of bounds for the hyperparameters
            likelihood_optimization_pop_size:       number of walkers in the optimization
            likelihood_optimization_tolerance:      tolerance for termination
            likelihood_optimization_max_iter:       maximum number of iterations
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
        self.hyperparameters = self.gp.hyperparameters
        return self.hyperparameters

##############################################################
    def train_gp(self,hyperparameter_bounds,
            likelihood_optimization_method,likelihood_optimization_pop_size,
            likelihood_optimization_tolerance,likelihood_optimization_max_iter):
        """
        Function to perform fvGP training.
        Parameters:
        -----------
            hyperparameter_bounds:                  2d np.array of bounds for the hyperparameters
            likelihood_optimization_method:         "hgdl"/"global"/"local"
            likelihood_optimization_pop_size:       number of walkers in the optimization
            likelihood_optimization_tolerance:      tolerance for termination
            likelihood_optimization_max_iter:       maximum number of iterations
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
        self.hyperparameters = self.gp.hyperparameters
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
        self.hyperparameters = self.gp.hyperparameters
        return self.hyperparameters

##############################################################
    def ask(self, position = None, n = 1,
            objective_function = "covariance",
            cost_function = None,
            cost_function_parameters = None,
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

        Optional Parameters:
        --------------------
            position (numpy array):            last measured point, default = None
            n (int):                           how many new measurements are requested, default = 1
            objective_function:                default = None, means that the class objective function will be used
            cost_function:                     default = None, i.e. no costs are used
            cost_function_parameters:          defaulr = None, i.e. the class variable is used
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
        if cost_function_parameters is None: cost_function_parameters = self.cost_function_parameters
        maxima, func_evals = sm.find_objective_function_maxima(self.gp,objective_function,
                position,n, optimization_bounds,
                optimization_method = optimization_method,
                optimization_pop_size = optimization_pop_size,
                optimization_max_iter = optimization_max_iter,
                optimization_tol = optimization_tol,
                cost_function = cost_function,
                cost_function_parameters = cost_function_parameters,
                dask_client = dask_client)
        return {'x':np.array(maxima), "f(x)" : np.array(func_evals)}

##############################################################
    def simulate(self, points, cost_function = None, cost_function_parameters = None, origin = None):
        """
        this function simulates a measurement:
        Parameters:
        -----------
        points (2d numpy array):           A 2d array of all the points we want to simulate

        Optional Parameters:
        --------------------
            origin (numpy array): default = None

        returns:
        --------
            return values, variances, value_positions, costs
        """
        if cost_function_parameters is None: cost_function_parameters = self.cost_function_parameters
        a = self.gp.posterior_mean(np.asarray(points))["f(x)"]
        b = self.gp.posterior_covariance(np.asarray(points))["v(x)"]
        variances = []
        values = []
        value_positions = []
        costs = []
        for i in range(len(points)):
            values.append(a[i])
            variances.append(b[i])
            if cost_function is not None and cost_function_parameters is not None:
                costs.append({"origin":origin, "point": np.array(points[i]),"cost": \
                    cost_function(origin,points[i],cost_function_parameters)})
            else:
                costs.append(None)
            value_positions.append(self.value_positions[-1])
        return values, variances, value_positions, costs
