#/usr/bin/env python
import time
from time import strftime
import numpy as np
from gpcam import global_config as conf

from . import misc as smc
from .data import Data
from .gp_optimizer import GPOptimizer
import dask.distributed


def main(init_data_files = None, init_hyperparameter_files = None):
    """
    The main loop

    Parameters
    ----------
    path_new_experiment_command : Path
        Full path to where to look to read data

    path_new_experiment_result : Path
        Full path to where to write new commands
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
    smc.delete_files()
    print("################################################")
    print("Initializing data set...")
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
    opt_tol = {}
    if conf.training_dask_client is True: training_dask_client = dask.distributed.Client()
    else: training_dask_client = conf.training_dask_client
    if conf.prediction_dask_client is True: prediction_dask_client = dask.distributed.Client()
    else: prediction_dask_client = conf.prediction_dask_client
    if training_dask_client is True and  prediction_dask_client is True: raise ValueError("Can't use same client for training and prediction")

    #################################################
    ####initialization of each Gaussian process######
    #################################################
    for gp_idx in conf.gaussian_processes.keys():
        function = conf.gaussian_processes[gp_idx]["data acquisition function"]
        print("Initialize the Gaussian Process for model: ", gp_idx)
        #####initializing data
        if init_data_files is not None:
            print("You have chosen to start with previously-collected data")
            if isinstance(init_data_files, str) and gp_idx in init_fata_files:
                print("Data set will be read from", init_data_files,".")
                d = list(np.load(init_data_file, allow_pickle = True))
                data[gp_idx] = Data(gp_idx,function,conf,d)
            elif isinstance(init_data_files, list):
                for entry in init_data_files:
                    print("read data from: ",entry)
                    if gp_idx in entry:
                        print("Data set for ",gp_idx,"will be read from", entry,".")
                        d = list(np.load(entry, allow_pickle = True))
                        data[gp_idx] = Data(gp_idx,function,conf,d)
            else:
                data[gp_idx] = Data(gp_idx,function,conf)
        else:
            data[gp_idx] = Data(gp_idx,function,conf)

        #####initializing hyperparameters
        if init_hyperparameter_files is not None:
            if isinstance(init_hyperparameter_files, str) and gp_idx in init_hyperparameter_files:
                print("Hyper parameters will be read form ", init_hyperparameter_files,".")
                hps = list(np.load(init_hyperparameter_files, allow_pickle = True))
                print("hyperparameters:", hps)
                training = None
            elif isinstance(init_hyperparameter_files, list):
                for entry in init_hyperparameter_files:
                    if gp_idx in entry:
                        print("Hyper parameters for ", gp_idx," will be read form ", entry,".")
                        hps = list(np.load(entry, allow_pickle = True))
                        print("hyperparameters:", hps)
                training = None
            else:
                hps = conf.gaussian_processes[gp_idx]["hyperparameters"]
                training = conf.likelihood_optimization_method
        else:
            hps =  conf.gaussian_processes[gp_idx]["hyperparameters"]
            training = conf.initial_likelihood_optimization_method
        #########################################
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
            if training_dask_client is not False and training == "hgdl":
                gp_optimizers[gp_idx].async_train_gp(
                    conf.gaussian_processes[gp_idx]["hyperparameter bounds"],
                    conf.likelihood_optimization_population_size,
                    conf.likelihood_optimization_tolerance,
                    conf.likelihood_optimization_max_iter,
                    training_dask_client
                    )
            else: gp_optimizers[gp_idx].train_gp(
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

        # save the found hyperparameters for fast restart and faster plotting
        np.save('../data/historic_data/Data_'+ start_date_time+"_" + gp_idx, data[gp_idx].data_set)
        np.save('../data/historic_data/hyperparameters_'+ start_date_time+"_" + gp_idx, gp_optimizers[gp_idx].gp.hyperparameters)
        np.save('../data/current_data/Data_' + gp_idx, data[gp_idx].data_set)
        np.save('../data/current_data/hyperparameters_' + gp_idx, gp_optimizers[gp_idx].gp.hyperparameters)

        number_of_measurements[gp_idx] = len(data[gp_idx].points)
        current_position = data[gp_idx].points[np.argmax(data[gp_idx].times)]
        opt_tol[gp_idx] = conf.gaussian_processes[gp_idx]["objective function optimization tolerance"]


    if conf.automatic_signal_variance_range_determination is True:
        a = smc.determine_signal_variance_range(data[gp_idx].values)
        print("automatic signal variance range determination activated")
        print("new signal variance range: ", a)
        conf.gaussian_processes[gp_idx]["hyperparameter bounds"][0] = a

    print("#############################################################")
    print("Initialization concluded, start of autonomous data collection")
    print("#############################################################")
    ###################################################################
    #######MAIN LOOP STARTS HERE########################################
    ####################################################################
    next_measurement_points = {}
    func_evals = {}
    post_var = {}
    simulated_next_values = {}
    simulated_next_variances = {}
    simulated_next_value_positions = {}
    simulated_next_costs = {}
    number_of_suggested_measurements = conf.number_of_suggested_measurements
    error_array = []
    while max(error.values()) > conf.breaking_error:
        print("")
        print("")
        print("")
        for gp_idx in conf.gaussian_processes.keys():
            number_of_measurements[gp_idx] = len(data[gp_idx].points)
            print("==========================================================")
            print("computing gp: ",gp_idx)
            print("Total Run Time: ", time.time() - start_time, "     seconds")
            print("number of measurements performed: ", number_of_measurements[gp_idx])
            print("==========================================================")
            #########################################
            ###ask for new points:###################
            #########################################
            if conf.objective_function_optimization_method == "hgdl":
                print("Asking for ",number_of_suggested_measurements," new point(s) and using hgdl")
            elif conf.objective_function_optimization_method == "global" and \
                 np.random.rand() < conf.chance_for_local_objective_function_optimization:
                ofom = "local"
                print("Next objective function optimization is local")
            else:
                ofom = conf.objective_function_optimization_method
                print("Next objective function optimization is ", conf.objective_function_optimization_method)
            ask_res = gp_optimizers[gp_idx].ask(position = current_position,
                    n = number_of_suggested_measurements,
                    objective_function = conf.gaussian_processes[gp_idx]["objective function"],
                    optimization_bounds = None,
                    optimization_method = ofom,
                    optimization_pop_size = conf.objective_function_optimization_population_size,
                    optimization_max_iter = conf.objective_function_optimization_max_iter, 
                    optimization_tol = opt_tol[gp_idx],
                    dask_client = prediction_dask_client)
            next_measurement_points[gp_idx] = ask_res["x"]
            func_evals[gp_idx] = ask_res["f(x)"]
            post_var[gp_idx] = gp_optimizers[gp_idx].gp.posterior_covariance(next_measurement_points[gp_idx])

            if conf.gaussian_processes[gp_idx]["adjust optimization threshold"][0] == True:
                opt_tol[gp_idx] = abs(func_evals[gp_idx][0] * conf.gaussian_processes[gp_idx]\
                ["adjust optimization threshold"][1])
                print("variance optimization tolerance of ",gp_idx," changed to: ",opt_tol[gp_idx])
            error[gp_idx] = abs(post_var[gp_idx]["v(x)"][0])
            #########################################
            ###simulate new points:##################
            #########################################
            print("Next points to be requested for ",gp_idx,": ")
            print(next_measurement_points[gp_idx])
            print("===============================")

            #simulated_next_values[gp_idx],\
            #simulated_next_variances[gp_idx],\
            #simulated_next_value_positions[gp_idx],\
            #simulated_next_costs[gp_idx]=\
            #gp_optimizers[gp_idx].simulate(next_measurement_points[gp_idx],
            #        cost_function = conf.gaussian_processes[gp_idx]["cost function"],
            #        origin = current_position)
            #########################################
            ###update data###########################
            #########################################
            print("Gathering data and performing measurement...")
            data[gp_idx].update_data(next_measurement_points[gp_idx],
                #simulated_next_values[gp_idx],
                #simulated_next_variances[gp_idx],
                #simulated_next_value_positions[gp_idx],
                #simulated_next_costs[gp_idx],
                error[gp_idx],
                gp_optimizers[gp_idx].hyperparameters)

            #########################################
            ###############preparing to tell()#######
            #########################################
            print("Communicating new data to the GP...")
            gp_optimizers[gp_idx].tell(
                    data[gp_idx].points,
                    data[gp_idx].values,
                    variances = data[gp_idx].variances,
                    value_positions = data[gp_idx].value_positions,
                    append = False)
            print("Training...")
            if number_of_measurements[gp_idx] in conf.global_likelihood_optimization_at:
                print("Fresh optimization from scratch via global optimization")
                gp_optimizers[gp_idx].stop_async_train()
                gp_optimizers[gp_idx].train_gp(
                conf.gaussian_processes[gp_idx]["hyperparameter bounds"],
                "global", conf.likelihood_optimization_population_size,
                conf.likelihood_optimization_tolerance,
                conf.likelihood_optimization_max_iter
                )
            if number_of_measurements[gp_idx] in conf.hgdl_likelihood_optimization_at:
                hyperparameter_update_mode = "hgdl"
                print("Fresh optimization from scratch via hgdl optimization")
                gp_optimizers[gp_idx].stop_async_train()
                if training_dask_client is not False:
                    print("Dask client for training specified; therefore, I will start")
                    print("an asynchronous hgdl training session")
                    gp_optimizers[gp_idx].async_train_gp(
                    conf.gaussian_processes[gp_idx]["hyperparameter bounds"],
                    hyperparameter_update_mode, conf.likelihood_optimization_population_size,
                    conf.likelihood_optimization_tolerance,
                    conf.likelihood_optimization_max_iter,
                    training_dask_client
                    )
                else:
                    print("Dask client for training not specified; therefore, I will start")
                    print("a synchronous hgdl training")
                    gp_optimizers[gp_idx].train_gp(
                    conf.gaussian_processes[gp_idx]["hyperparameter bounds"],
                    hyperparameter_update_mode, conf.likelihood_optimization_population_size,
                    conf.likelihood_optimization_tolerance,
                    conf.likelihood_optimization_max_iter
                    )
            elif number_of_measurements[gp_idx] in conf.local_likelihood_optimization_at:
                print("Local training initiated")
                gp_optimizers[gp_idx].stop_async_train()
                gp_optimizers[gp_idx].train_gp(
                    conf.gaussian_processes[gp_idx]["hyperparameter bounds"],
                    "local", conf.likelihood_optimization_population_size,
                    conf.likelihood_optimization_tolerance,
                    conf.likelihood_optimization_max_iter
                    )
            else:
                print("No training was performed in this iteration.")
                gp_optimizers[gp_idx].update_hyperparameters()

            if np.random.random() < conf.gaussian_processes[gp_idx]["cost optimization chance"]\
                    and conf.gaussian_processes[gp_idx]["cost function"] is not None \
                    and conf.gaussian_processes[gp_idx]["cost update function"] is not None:
                gp_optimizers[gp_idx].update_cost_function(
                data[gp_idx].measurement_costs,
                conf.gaussian_processes[gp_idx]["cost update function"],
                conf.gaussian_processes[gp_idx]["cost function optimization bounds"])
            print("===============================")
            ########################################
            ###save current data####################
            ########################################
            if conf.gaussian_processes[gp_idx]["run function in every iteration"] is not None:
                conf.gaussian_processes[gp_idx]["run function in every iteration"](gp_optimizers[gp_idx])
            np.save('../data/historic_data/Data_'+ start_date_time+"_" + gp_idx, data[gp_idx].data_set)
            np.save('../data/historic_data/hyperparameters_'+ start_date_time+"_" + gp_idx, gp_optimizers[gp_idx].gp.hyperparameters)
            np.save('../data/current_data/Data_' + gp_idx, data[gp_idx].data_set)
            np.save('../data/current_data/hyperparameters_' + gp_idx, gp_optimizers[gp_idx].gp.hyperparameters)

            current_position = data[gp_idx].points[np.argmax(data[gp_idx].times)]
        if any(i >= conf.max_number_of_measurements for i in number_of_measurements.values()):
            print("The maximum number of measurements has been reached")
            break
        print("found posterior variances:", error)
        error_array.append(error[gp_idx])

    print("====================================================")
    print("The autonomous experiment was concluded successfully")
    print("====================================================")
    np.save("../data/current_data/error_array",error_array)

if __name__ == "__main__":
    main()

