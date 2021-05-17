#!/usr/bin/env python
import time
from time import strftime
import numpy as np
from gpcam import global_config as conf

from . import misc as smc
from .data import Data
from .gp_optimizer import GPOptimizer
import dask.distributed



def main(data_path = None, hyperparameter_path = None):
    """
    The main loop

    Parameters
    ----------
    you have the option to specify paths by calling "python Run_Visualization.py path_to_data path_to_hyperparameters"
    If the command line options are gives but set to "None" the data will be read from the file specified in the configuration file
    "Config_Visualization.py" and the hyperparameters will be recomputed. If you want to specify one path to data or hyperparameters, set the other one 
    to "None". If both file paths are specified in the configuration file, you can call Run_Visualization without command line parameters
    """
    #########################################
    ######Prepare first set of Experiments###
    #########################################
    print("################################################")
    print("#gpCAM####powered by CAMERA @ LBNL##############")
    print("################################################")
    print("#########Version: 6.0###########################")
    print("################################################")
    print("")
    start_time = time.time()
    start_date_time = strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    print("Date and time:       ", start_date_time)
    print("################################################")
    #########Initialize a set of random measurements####
    #########this will be the initial experiment data###
    print("################################################")
    print("Visualizing data set...")
    print("################################################")
    ###############################################
    ###Begin GP initialization loop################
    ###############################################
    gp_optimizers = {}
    data = {}
    number_of_measurements = {}
    GlobalUpdateKernelNumber = {}
    LocalUpdateKernelNumber = {}
    error = {}
    measurement_costs = {}

    print("Data set will be read from",data_path,".")
    try: 
        d = np.load(data_path, allow_pickle = True)
    except:
        print("Something went wrong when I tried to read the file ",data_path)
    print("")

    gp_idx = d[0]["function name"]
    data[gp_idx] = Data(gp_idx,1.0,conf,d)
    print("Length of data to be plotted: ", len(d))
    if hyperparameter_path is None:
        print("You have chosen to recompute the hyperparameters.")
        hps =  conf.gaussian_processes[gp_idx]["hyperparameters"]
        training = "global"
    elif hyperparameter_path is not None:
        print("You have chosen to specify your path for the hyperparameters as command line option.")
        print("Hyper parameters will be read from",hyperparameter_path,".")
        hps = list(np.load(hyperparameter_path, allow_pickle = True))
        training = None
    else:
        print("Not sure where to get the hyperparameters from")
    print("hyperparameters: ", hps)

    error[gp_idx] = np.inf
    gp_optimizers[gp_idx] = GPOptimizer(
        len(conf.parameters),
        conf.gaussian_processes[gp_idx]["dimensionality of return"],
        conf.gaussian_processes[gp_idx]["number of returns"],
        data[gp_idx].variance_optimization_bounds,
        )
    gp_optimizers[gp_idx].tell(
        data[gp_idx].points,
        data[gp_idx].values,
        variances = data[gp_idx].variances,
        value_positions = data[gp_idx].value_positions,
        append = False
            )
    gp_optimizers[gp_idx].init_gp(hps,compute_device = conf.compute_device,
        gp_kernel_function = conf.gaussian_processes[gp_idx]["kernel function"],
        gp_mean_function = conf.gaussian_processes[gp_idx]["mean function"],
        sparse = conf.sparse
            )
    if training is not None: 
        if training_dask_client is not False:
            gp_optimizers[gp_idx].async_train(
                conf.gaussian_processes[gp_idx]["hyperparameter bounds"],
                conf.likelihood_optimization_population_size,
                conf.likelihood_optimization_tolerance,
                conf.likelihood_optimization_max_iter,
                training_dask_client
                )
        else: gp_optimizers[gp_idx].train(
                conf.gaussian_processes[gp_idx]["hyperparameter bounds"],
                training, conf.likelihood_optimization_population_size,
                conf.likelihood_optimization_tolerance,
                conf.likelihood_optimization_max_iter
                )
    if conf.gaussian_processes[gp_idx]["cost function"] is not None and\
        conf.gaussian_processes[gp_idx]["cost function parameters"] is not None:
            gp_optimizers[gp_idx].init_cost(
            conf.gaussian_processes[gp_idx]["cost function"],
            conf.gaussian_processes[gp_idx]["cost function parameters"],
            conf.gaussian_processes[gp_idx]["cost update function"],
            conf.gaussian_processes[gp_idx]["cost function optimization bounds"],
                    )

    if conf.gaussian_processes[gp_idx]["cost update function"] is not None and\
        conf.gaussian_processes[gp_idx]["cost function optimization bounds"] is not None:
        gp_optimizers[gp_idx].update_cost_function(
            data[gp_idx].measurement_costs)

    if training == True:
        print("Hyper parameters saved in ../data/historic_data/hyperparameters_from_last_visualization_"+start_date_time)
        np.save('../data/historic_data/hyperparameters_from_visualization'+'_'+ start_date_time, gp_optimizers[gp_idx].gp.gp_kernel)

    conf.gaussian_processes[gp_idx]["plot function"](gp_optimizers[gp_idx])

    print("########################################################")
    print("#######Visualization Concluded##########################")
    print("########################################################")


if __name__ == "__main__":
    main()
