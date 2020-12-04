###########################################
###Configuration File######################
###for gpcam steering of experiments#######
###########################################
import numpy as np
import dask.distributed
###############################
###General#####################
###############################
parameters = {
    "x1": {
        "element interval": [-5,5],  ####either an interval, several intervals or discrete points
    },
    "x2": {
        "element interval": [-5,5],
    },
}


from data_acquisition_functions import synthetic_function, send_data_as_files
#from objective_function_definition import exploitation,shape_finding, gradient_mode
from mean_functions import example_mean
from cost_function_definition import l1_cost
from cost_function_definition import update_l1_cost_function
from kernel_definition import kernel_l2_single_task,kernel_l2_multi_task, symmetric_kernel2,non_stat_kernel_2d,periodic_kernel_2d
from run_in_every_iteration import write_vtk_file
from plotting_functions import plot_2d_function

gaussian_processes = {
    "model_1": {
        "kernel function": None,
        #"kernel function": kernel_l2_multi_task,
        "hyperparameters": [1.0,1.0,1.0],
        "hyperparameter bounds": [[1.0,100.0],[0.10,100.0],[0.10,100.0]],
        "number of returns": 1,
        "dimensionality of return": 1,
        "objective function optimization tolerance": 0.001,
        "adjust optimization threshold": [True,0.1],
        "run function in every iteration": None,
        #"data acquisition function": send_data_as_files,
        "data acquisition function": synthetic_function,
        "objective function": "covariance",
        #"objective function": "shannon_ig",
        "mean function": None,
        "cost function": None,
        "cost update function": None,
        "cost function parameters": {"offset": 10,"slope":[2.0,2.0]},
        "cost function optimization bounds": [[0.0,10.0],[0.0,10.0],[0.0,10.0]],
        "cost optimization chance" : 0.1,
        "plot function": plot_2d_function
    },
    ##definition of more gps here if desired
}

breaking_error = 1e-12
automatic_signal_variance_range_determination = True
########################################
###Variance Optimization################
########################################
objective_function_optimization_method = "global"
chance_for_local_objective_function_optimization = 0.5 #\in [0,1], omly relevant of method is global
objective_function_optimization_population_size = 10
objective_function_optimization_max_iter = 20
number_of_suggested_measurements = 1  ###only important for "hgdl" in objective_function_optimization_method

initial_likelihood_optimization_method = "global"
global_likelihood_optimization_at = [200]
local_likelihood_optimization_at = [100,400,1000]
hgdl_likelihood_optimization_at = []
likelihood_optimization_population_size = 20
likelihood_optimization_tolerance = 0.001
likelihood_optimization_max_iter = 120
########################################
###Computation Parameters###############
########################################
compute_device = "cpu"
sparse = False
compute_inverse = False
training_dask_client = False #dask.distributed.Client()  #None/False/client
prediction_dask_client = False  #None/False/client
###############################
###DATA ACQUISITION############
###############################
initial_data_set_size = 20
max_number_of_measurements = 100

animation = {
        'model': 'model',
        'parameter 1': 'x1',
        'bounds 1': [-5.0,5.0],
        'parameter 2': 'x2',
        'bounds 2': [-5.0,5.0]
        }




#####################################################################
###############The END###############################################
#####################################################################














#########################################
####example for time series experiments:#
#########################################
"""
time_series = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
parameters = {}
for i in time_series:
    parameters["temperature_"+str(i)] = {
        "element interval": [1.0,500.0],  ####either an interval, several intervals or discrete points
hp = np.array([[10,1,1,1,1,1,1,1,1,1,1,1,1]])
b = np.array([[
        [0.0001,1000],
        [0.0001,1000],
        [0.0001,1000],
        [0.0001,1000],
        [0.0001,1000],
        [0.0001,1000],
        [0.0001,1000],
        [0.0001,1000],
        [0.0001,1000],
        [0.0001,1000],
        [0.0001,1000],
        [0.0001,1000],
        [1,2]
        ]])
"""

