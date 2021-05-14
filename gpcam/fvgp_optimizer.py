#!/usr/bin/env python

import numpy as np
from fvgp.fvgp import fvGP
from gpcam import surrogate_model as sm
from fvgp.gp import GP
from gpcam.gp_optimizer import GPOptimizer

class fvGPOptimizer(fvGP, GPOptimizer):
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
        index_set_bounds,
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
        self.index_set_bounds = np.array(index_set_bounds)
        self.hyperparameters = None
        self.gp_initialized = False
        self.cost_function_parameters = None
        self.cost_function = None
        self.consider_costs = False
        GPOptimizer.__init__(self,
                input_space_dimension,
                index_set_bounds
                )

    def get_data_fvGP(self):
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
                                                      value_posiitons = np.array([
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

        if self.gp_initialized is True:
            self.update_fvgp()

##############################################################
    def init_fvgp(
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
            sparse = sparse,
            )
            self.gp_initialized = True
            self.hyperparameters = np.array(init_hyperparameters)

##############################################################
    def update_fvgp(self):
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
            value_positions = self.value_positions,
            variances = self.variances)

