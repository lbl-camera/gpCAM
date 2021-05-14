#/usr/bin/env python
import time
from time import strftime
import numpy as np
from gpcam.gp_data import gpData
from gpcam.gp_optimizer import GPOptimizer

#todo: costs are not included, autonomous experimenter for fvgp, data class for fvgp, script that uses the config and runs the  autonomous experimenter


class AutonomousExperimenterGP():
    """
    class AutonomousExperimenterGP:
    executes the autonomous loop for a single-task GP
    use class AutonomousExperimenterfvGP for multi-task experiments
    Parameters:
    -----------
        * parameter_bounds
        * instrument_func
        * hyperparameters
        * hyperparameter_bounds
    Optional Parameters:
    --------------------
        * init_dataset_size = None: int or None, None means you have to provide intial data
        * acq_func = "covariance": acquisition function to be maximized in search of new measurements
        * cost_func = None
        * cost_update_func = None
        * cost_func_parameters = {}
        * kernel_func = None
        * prior_mean_func = None
        * run_every_iteration = None
        * x = None, y = None, v = None: inital data can be supplied here
        * append_data = False: Append data or communiate entire dataset
        * compute_device = "cpu"
        * sparse = False
    """
    def __init__(self,
            parameter_bounds,
            instrument_func,
            hyperparameters,
            hyperparameter_bounds,
            init_dataset_size = None,
            acq_func = "covariance",
            cost_func = None,
            cost_update_func = None,
            cost_func_params = {},
            kernel_func = None,
            prior_mean_func = None,
            run_every_iteration = None,
            x = None, y = None, v = None, dataset = None,
            append_data = False,
            compute_device = "cpu",
            sparse = False,
            training_dask_client = None,
            acq_func_opt_dask_client = None
            ):
        self.parameter_bounds = parameter_bounds
        self.dim = len(parameter_bounds)
        self.instrument_func = instrument_func
        self.hyperparameters = hyperparameters
        self.hyperparameter_bounds = hyperparameter_bounds
        self.acq_func = acq_func
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
        self.training_dask_client = training_dask_client
        self.acq_func_opt_dask_client = acq_func_opt_dask_client
        #getting the data ready
        if init_dataset_size is None and x is None:
            raise Exception("Either provide length of initial data or an inital dataset")
        self.data = gpData(self.dim, self.parameter_bounds,self.instrument_func,init_dataset_size,self.append)
        if (x is None or y is None) and dataset is None:
            self.data.create_random_init_dataset()
        elif (x is None or y is None) and dataset is not None:
            self.data.comm_init_dataset(list(np.load(dataset, allow_pickle = True)))
            self.hyperparameters = self.data.dataset[-1]["hyperparameters"]
        else:
            self.data.comm_init_data(self.data.translate2data(x,y,v))
        self.x = self.data.x
        self.y = self.data.y
        self.v = self.data.v
        self.init_dataset_size = len(self.x)
        ######################
        self.gp_optimizer = GPOptimizer(self.dim,parameter_bounds)
        self.gp_optimizer.tell(self.x, self.y,variances = self.v)
        self.gp_optimizer.init_gp(self.hyperparameters,compute_device = self.compute_device,
            gp_kernel_function = self.kernel_func,
            gp_mean_function = self.prior_mean_func,
            sparse = self.sparse)
        print("##################################################################################")
        print("Initialization successfully concluded")
        print("now train(...) or train_async(...), and then go(...)")
        print("##################################################################################")
    ###################################################################################
    def train(self,pop_size = 10,tol = 1e-6, max_iter = 20, method = "global"):
        self.gp_optimizer.train_gp(
        self.hyperparameter_bounds,
        method = method, pop_size = pop_size,
        tolerance = tol, max_iter = max_iter)
    def train_async(self,pop_size = 10,tol = 1e-6, max_iter = 20, dask_client = None):
        self.gp_optimizer.train_gp_async(
        self.hyperparameter_bounds,pop_size = pop_size,
        tolerance = tol,max_iter = max_iter,
        dask_client = dask_client
        )
        self.async_train = True
    def kill_training(self):
        self.gp_optimizer.stop_async_train()
        self.async_train = False

    def update_hps(self):
        self.gp_optimizer.update_hyperparameters()

    ###################################################################################
    def go(self, N = 1e15, breaking_error = 1e-50,
            retrain_globally_at = [100,400,1000],
            retrain_locally_at = [20,40,60,80,100,200,400,1000],
            retrain_async_at = [1000,2000,5000,10000],
            retrain_callable_at = [],
            acq_func_opt_setting = lambda number: "global" if number % 2 == 0 else "local",
            training_opt_callable = None,
            training_opt_max_iter = 20,
            training_opt_pop_size = 10,
            training_opt_tol      = 1e-6,
            acq_func_opt_max_iter = 20,
            acq_func_opt_pop_size = 20,
            acq_func_opt_tol      = 1e-6,
            acq_func_opt_tol_adjust = [True,0.1],
            number_of_suggested_measurements = 1,
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
            * retrain_callable = []: if this is not an empty list, "training_opt_callable" has to be provided
            * search_setting = lambda function to decide when global, local, hgdl, other ask()
            * training_opt_callable = None, callable
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

        for i in range(len(self.x),int(N)):
            n_measurements = len(self.x)
            print("")
            print("")
            print("")
            print("====================")
            print("====================")
            print("iteration: ",i)
            print("Run Time: ", time.time() - start_time, "     seconds")
            print("Number of measurements: ", n_measurements)
            print("====================")
            #ask() for new suggestions
            current_position = self.x[-1]
            res = self.gp_optimizer.ask(
                    position = current_position,
                    n = number_of_suggested_measurements,
                    acquisition_function = self.acq_func,
                    bounds = None,
                    method = acq_func_opt_setting(i),
                    pop_size = acq_func_opt_pop_size,
                    max_iter = acq_func_opt_max_iter,
                    tol = acq_func_opt_tol,
                    dask_client = self.acq_func_opt_dask_client)
            #########################
            next_measurement_points = res["x"]
            func_evals = res["f(x)"]
            post_var = self.gp_optimizer.posterior_covariance(next_measurement_points)
            error = np.max(post_var["v(x)"])
            if acq_func_opt_tol_adjust[0]:
                acq_func_opt_tol = abs(func_evals[0]) * acq_func_opt_tol_adjust[1]
                print("variance optimization tolerance of changed to: ", acq_func_opt_tol)
            print("Next points to be requested: ")
            print(next_measurement_points)
            #update and tell() new data
            self.x,self.y, self.v = self.data.add_data_points(next_measurement_points,
                                   post_var,self.gp_optimizer.hyperparameters)
            self.gp_optimizer.tell(self.x, self.y,variances = self.v)
            ###########################
            #train()
            if len(self.x) in retrain_async_at:
                self.kill_training()
                self.train_async(training_opt_pop_size = 10,training_opt_tol = 1e-6, 
                                 training_opt_max_iter = 20, dask_client = self.training_dask_client)
            elif len(self.x) in retrain_globally_at:
                self.kill_training()
                print("Fresh optimization from scratch via global optimization")
                self.train(training_opt_pop_size = 10,training_opt_tol = 1e-6,
                                 training_opt_max_iter = 20, method = "global")
            elif len(self.x) in retrain_locally_at:
                self.kill_training()
                print("Fresh optimization from scratch via global optimization")
                self.train(training_opt_pop_size = 10,training_opt_tol = 1e-6,
                                 training_opt_max_iter = 20, method = "local")
            elif len(self.x) in retrain_callable_at:
                self.kill_training()
                print("Fresh optimization from scratch via global optimization")
                self.train(training_opt_pop_size = 10,training_opt_tol = 1e-6,
                                 training_opt_max_iter = 20, method = training_opt)
            else:
                self.update_hps()
                print("No training in this round but I tried and update the hyperparameters")
            ###save some data
            try: np.save('Data_'+ start_date_time, self.data.dataset)
            except Exception as e: print("Data not saved due to ", str(e))
            ###########################
            #cost_update()
            #self.cost_func_parameters = self.cost_update_func()

            if error < breaking_error: break
        print("====================================================")
        print("The autonomous experiment was concluded successfully")
        print("====================================================")
    ###################################################################################
    ###################################################################################


#class AutonomousExperimenterfvGP():
#class AutonomousExperimenterEnsembleGP():
