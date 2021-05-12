#/usr/bin/env python
import time
from time import strftime
import numpy as np
from gpcam import global_config as conf

from . import misc as smc
from .data import Data
from .gp_optimizer import GPOptimizer
import dask.distributed




class AutonomousExperimenterGP():
    def __init__(parameter_bounds,
            init_data_size = 10,
            instrument_func,
            hyperparameters,
            hyperparameter_bounds,
            acq_func = None,
            cost_func = None,
            cost_update_func = None,
            cost_func_params = {},
            kernel_func = None,
            prior_mean_func = None,
            run_every_iteration = None,
            x = None, y = None, v = None,
            append_data = False
            compute_device = "cpu",
            sparse = False
            ):
        self.dim = len(parameter_bounds)
        self.instrument_func = instrument_func
        self.init_data_size = init_data_size
        self.hyperparameters = hyperparameters
        self.hyperparameter_bounds = hyperparameter_bounds
        self.cost_func = cost_func
        self.cost_update_func = cost_update_func
        self.cost_func_params = cost_func_params
        self.kernel_func = kernel_func
        self.prior_mean_func = prior_mean_func
        self.run_every_iteration = run_every_iteration
        self.compute_device = compute_device
        self.append = append_data
        self.sparse = sparse
        self.async_train = False
        if x is None or y is None:
            self.x = self._create_random_data()
            self.y, self.v = self.instrument_func(self.x)
        else:
            self.x = x
            self.y = y
            self.v = v
        self.gp_optimizer = GPOptimizer(self.dim,parameter_bounds)
        gp_optimizer.tell(self.x,
            self.y,variances = self.v,
            append = self.data)
        gp_optimizer.init_gp(self.hyperparameters,compute_device = self.compute_device,
            gp_kernel_function = self.kernel_func,
            gp_mean_function = self.prior_mean_func,
            sparse = self.sparse)
        self.data = gpData(self.dim, self.instrument_func)
    print("##################################################################################")
    print("Initialization successfully concluded")
    print("now train(...) or train_async(...), and then go(...)")
    print("##################################################################################")
    ###################################################################################
    def train(self,pop_size,tol,max_iter, method = "global", dask_client = None):
        gp_optimizer.train_gp(
        self.hyperparameter_bounds,method,pop_size,tol,max_iter,dask_client
        )
    def train_async(self,pop_size,tol,max_iter, dask_client = None):
        self.gp_optimizer.train_gp_async(
        self.hyperparameter_bounds,pop_size,tol,max_iter,dask_client
        )
        self.async_train = True
    def kill_training(self)
        self.gp_optimizer.stop_async_train()
        self.async_train = False
    def update_hps(self):
        self.gp_optimizer.update_hyperparameters()

    ###################################################################################
    def go(self, N = 1e15, breaking_error = 1e-50,
            retrain_globally_at = [100,400,1000],
            retrain_locally_at = [20,40,60,80,100,200,400,1000],
            retrain_async_at = [1000,2000,5000,10000],
            acq_func_opt_settings = ,
            training_opt = "global",
            acq_func_opt = "global",
            training_opt_max_iter = 20,
            training_opt_pop_size = 10,
            training_opt_tol      = 1e-6,
            acq_func_opt_max_iter = 20,
            acq_func_opt_pop_size = 20,
            acq_func_opt_tol      = 1e-6,
            acq_func_opt_tol_adjust = [True,0.1],
            number_of_suggested_measurements = 1,
            sparse = False,
            training_dask_client = None,
            acq_func_opt_dask_client = None
            ):
        """
        function to start the autonomous-data-acquisition loop
        optional parameters:
        -----------
            * N = 1e15 ... run N iterations
            * breaking_error = 1e-15 ... run until breaking_error is achieved
            * retrain_globally_at = [100,400,1000]
            * retrain_locally_at = [20,40,60,80,100,200,400,1000]
            * retrain_async_at = [1000,2000,5000,10000]
            * search_settings = when global, local, hgdl, other
            * training_opt = "global", local, hgdl, callable
            * acq_func_opt = "global", local, hgdl, callable
            * training_opt_max_iter = 20
            * training_opt_pop_size = 10
            * training_opt_tol      = 1e-6
            * acq_func_opt_max_iter = 20
            * acq_func_opt_pop_size = 20
            * acq_func_opt_tol      = 1e-6
            * acq_func_opt_tol_adjust = [True, 0.1]
            * number_of_suggested_measurements = 1
            * compute_device = cpu
            * sparse = False
            * training_dask_client = None
            * acq_func_opt_dask_client = None
        """
        start_time = time.time()
        start_date_time = strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
        print("Date and time:       ", start_date_time)
        
        for i in range(N):
            n_measurements = len(self.x)
            print("====================")
            print("====================")
            print("iteration: ",i)
            print("Run Time: ", time.time() - self.start_time, "     seconds")
            print("Number of measurements: ", n_measurements)
            print("====================")
            #ask() for new suggestions
            current_position = x[-1]
            res = gp_optimizer.ask(
                    position = current_position,
                    n = number_of_suggested_measurements,
                    acquisition_function = conf.gaussian_processes[gp_idx]["acquisition function"],
                    bounds = None,
                    method = ofom,
                    pop_size = conf.acquisition_function_optimization_population_size,
                    max_iter = conf.acquisition_function_optimization_max_iter, 
                    tol = opt_tol[gp_idx],
                    dask_client = prediction_dask_client)
            #########################
            next_measurement_points = res["x"]
            func_evals = res["f(x)"]
            post_var = gp_optimizer.posterior_covariance(next_measurement_points[gp_idx])
            error = np.max(post_var[gp_idx]["v(x)"])
            if acq_func_opt_tol_adjust[0]: 
                acq_func_opt_tol = abs(func_evals[0]) * acq_func_opt_tol_adjust[1]
                print("variance optimization tolerance of changed to: ", acq_func_opt_tol)
            print("Next points to be requested: ")
            print(next_measurement_points)
            #update and tell() new data
            self.x,self.y.self.v = self.data.update_data(next_measurement_points,
                                   post_var,gp_optimizer.hyperparameters)
            ###########################
            #train()
            if len(self.x) in retrain_async_at: self.train_async()
            elif: len(self.x) in retrain_globally:
                print("Fresh optimization from scratch via global optimization")
                gp_optimizers.stop_async_train()
                self.train(pop_size,training_opt_tol,max_iter, method = "global", dask_client = None):
                gp_optimizer.train_gp(
                self.hyperparameter_bounds,method,pop_size,tol,max_iter,dask_client
                )

                self.train(method = "global")

            elif: len(self.x) in retarin_locally: self.train(method = "local")
            else: print("No training in this round")
            ###########################
            #cost_update()
            self.cost_func_parameters = self.cost_update_func()

            if error < breaking_error: break
        print("====================================================")
        print("The autonomous experiment was concluded successfully")
        print("====================================================")
    ###################################################################################
    def _create_random_x(self):
        return np.linalg.uniform(low = self.parameter_bounds[:,0],
                                 high =self.parameter_bounds[:,1],
                                 size = self.dim)
    def _create_random_points(self):
        x = np.empty((self.init_data_size,self.dim))
        for i in range(len(self.init_data_size)):
            x[i,:] = data_create_random_x()
        return x
    ###################################################################################


#class AutonomousExperimenterfvGP():
#class AutonomousExperimenterEnsembleGP():
