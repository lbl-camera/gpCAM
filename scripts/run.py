#/usr/bin/env python
import numpy as np
from gpcam.autonomous_experimenter import AutonomousExperimenterGP
import Config as c



def run(init_data_file = None):
    #initialize
    my_ae = AutonomousExperimenterGP(
            c.parameters,
            c.gp["instrument function"],c.gp["hyperparameters"], c.gp["hyperparameter bounds"],
            init_dataset_size = c.initial_dataset_size, dataset = init_data_file,
            acq_func = c.gp["acquisition function"], cost_func = c.gp["cost function"],
            cost_update_func = c.gp["cost update function"], cost_func_params = c.gp["cost function parameters"],
            kernel_func = c.gp["kernel function"], prior_mean_func = c.gp["mean function"],
            run_every_iteration = c.gp["run function in every iteration"],
            append_data_after_send = c.append_data_after_send, compute_device = c.compute_device,
            sparse = c.sparse,
            training_dask_client = c.training_dask_client,
            acq_func_opt_dask_client = c.prediction_dask_client)
    #train
    my_ae.train()
    #start the autonomous loop
    my_ae.go(N = c.max_number_of_measurements, breaking_error = c.breaking_error,
             retrain_globally_at = c.global_likelihood_optimization_at,
             retrain_locally_at = c.local_likelihood_optimization_at,
             retrain_async_at = c.hgdl_likelihood_optimization_at,
            retrain_callable_at = [],
            acq_func_opt_setting = lambda number: "global" if number % 2 == 0 else "local",
            training_opt_callable = None,
            training_opt_max_iter = c.likelihood_optimization_max_iter,
            training_opt_pop_size = c.likelihood_optimization_population_size,
            training_opt_tol      = c.likelihood_optimization_tolerance,
            acq_func_opt_max_iter = c.acquisition_function_optimization_max_iter,
            acq_func_opt_pop_size = c.acquisition_function_optimization_population_size,
            acq_func_opt_tol      = c.gp["acquisition function optimization tolerance"],
            acq_func_opt_tol_adjust = c.gp["adjust optimization tolerance"],
            number_of_suggested_measurements = c.number_of_suggested_measurements,
            )

