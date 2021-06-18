###########################################
###Configuration File######################
###for gpcam steering of experiments#######
###########################################
import numpy as np
import dask.distributed
###############################
###General#####################
###############################
parameters = np.array([[-5,5],[-10,10]])


from instrument_function import synthetic_function
from acquisition_function import exploration, upper_confidence_bounds
from mean_function import example_mean
from cost_function import l1_cost
from cost_function import update_l1_cost_function
from kernel_function import kernel_l2_single_task,kernel_l2_multi_task, symmetric_kernel2,non_stat_kernel_2d,periodic_kernel_2d
from run_every_iteration import write_vtk_file
from plotting_function import plot_function

gp ={
        "kernel function": None,
        #"kernel function": kernel_l2_multi_task,
        "hyperparameters": [1.0,1.0,1.0],
        "hyperparameter bounds": [[1.0,100.0],[0.10,100.0],[0.10,100.0]],
        "number of returns": 1,
        "dimensionality of return": 1,
        "acquisition function optimization tolerance": 0.001,
        "adjust optimization tolerance": [True,0.1],
        "run function in every iteration": None,
        "instrument function": synthetic_function,
        #"acquisition function": "covariance",
        "acquisition function": "shannon_ig",
        "mean function": None,
        "cost function": None,
        "cost update function": None,
        "cost function parameters": {"offset": 10,"slope":[2.0,2.0]},
        "plot function": plot_function
}
append_data_after_send = True
breaking_error = 1e-12
########################################
###Variance Optimization################
########################################
acquisition_function_optimization_method = "global"
acquisition_function_optimization_population_size = 10
acquisition_function_optimization_max_iter = 20
number_of_suggested_measurements = 1  ###only important for "hgdl" in acquisition_function_optimization_method

initial_likelihood_optimization_method = "global"
global_likelihood_optimization_at = [10,20,30,40]
local_likelihood_optimization_at = [15,35,1000]
hgdl_likelihood_optimization_at = [17,32]
likelihood_optimization_population_size = 20
likelihood_optimization_tolerance = 0.001
likelihood_optimization_max_iter = 120
########################################
###Computation Parameters###############
########################################
compute_device = "cpu"
sparse = False
training_dask_client = None #dask.distributed.Client()  #None/False/client
prediction_dask_client = None  #None/False/client
###############################
###DATA ACQUISITION############
###############################
initial_dataset_size = 20
max_number_of_measurements = 100

#####################################################################
###############The END###############################################
#####################################################################
