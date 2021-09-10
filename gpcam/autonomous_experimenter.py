#/usr/bin/env python
import time
from time import strftime
import numpy as np
from gpcam.data import gpData
from gpcam.data import fvgpData
from gpcam.gp_optimizer import GPOptimizer
from gpcam.gp_optimizer import fvGPOptimizer
import dask
import dask.distributed as distributed

#todo: autonomous experimenter for fvgp, data class for fvgp


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
        * cost_func_params = {}
        * kernel_func = None
        * prior_mean_func = None
        * run_every_iteration = None
        * x = None, y = None, v = None: inital data can be supplied here
        * append_data_after_send = False: Append data or communiate entire dataset
        * compute_device = "cpu"
        * sparse = False
        * training_dask_client = None
        * acq_func_opt_dask_client = None
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
            append_data_after_send = False,
            compute_device = "cpu",
            sparse = False,
            training_dask_client = None,
            acq_func_opt_dask_client = None
            ):
        dim = len(parameter_bounds)
        self.instrument_func = instrument_func
        self.hyperparameter_bounds = hyperparameter_bounds
        self.acq_func = acq_func
        self.cost_func = cost_func
        self.cost_update_func = cost_update_func
        self.kernel_func = kernel_func
        self.prior_mean_func = prior_mean_func
        self.run_every_iteration = run_every_iteration
        self.append = append_data_after_send
        self.async_train_in_progress = False
        self.training_dask_client = training_dask_client
        if self.training_dask_client is None: self.training_dask_client = dask.distributed.Client()
        self.acq_func_opt_dask_client = acq_func_opt_dask_client
        if self.acq_func_opt_dask_client is None: self.acq_func_opt_dask_client = self.training_dask_client
        ################################
        #getting the data ready#########
        ################################
        if init_dataset_size is None and x is None and dataset is None:
            raise Exception("Either provide length of initial data or an inital dataset")
        self.data = gpData(dim, parameter_bounds)
        if  x is None and dataset is None:
            self.data.create_random_dataset(init_dataset_size)
            self.data.dataset = self.instrument_func(self.data.dataset)
        elif dataset is not None:
            self.data.inject_dataset(list(np.load(dataset, allow_pickle = True)))
            hyperparameters = self.data.dataset[-1]["hyperparameters"]
        elif x is not None and y is not None:
            self.data.dataset = self.data.inject_arrays(x,y=y,v=v)
        elif x is not None and y is None:
            self.data.dataset = self.instrument_func(self.data.inject_arrays(x,y=y,v=v))
        else: raise Exception("No viable option for data given!")
        if self.data.nan_in_dataset(): self.data.clean_data_NaN()
        self.x, self.y, self.v, self.t, self.c = self.data.extract_data()
        self.init_dataset_size = len(self.x)
        ######################
        ######################
        ######################
        self.gp_optimizer = GPOptimizer(dim,parameter_bounds)
        self.gp_optimizer.tell(self.x, self.y,variances = self.v)
        self.gp_optimizer.init_gp(hyperparameters, compute_device = compute_device,
            gp_kernel_function = self.kernel_func,
            gp_mean_function = self.prior_mean_func,
            sparse = sparse)
        #init costs
        self._init_costs(cost_func_params)
        print("##################################################################################")
        print("Autonomous Experimenter initialization successfully concluded")
        print("now train(...) or train_async(...), and then go(...)")
        print("##################################################################################")
    ###################################################################################
    def train(self,pop_size = 10, tol = 1e-6, max_iter = 20, method = "global"):
        self.gp_optimizer.train_gp(
        self.hyperparameter_bounds,
        method = method, pop_size = pop_size,
        tolerance = tol, max_iter = max_iter)

    def train_async(self, max_iter = 20, dask_client = None):
        if dask_client is None: dask_client = self.training_dask_client
        print("AutonomousExperimenter starts async training with dask client:")
        print(dask_client)
        self.opt_obj = self.gp_optimizer.train_gp_async(
        self.hyperparameter_bounds,max_iter = max_iter,
        dask_client = dask_client
        )
        print("The Autonomous Experimenter started an instance of the asynchronous training.")
        self.async_train_in_progress = True

    def kill_training(self):
        print("async training is being killed")
        if self.async_train_in_progress: self.gp_optimizer.stop_async_train(self.opt_obj)
        else: print("no training to be killed")
        self.async_train_in_progress = False

    def kill_client(self):
        try: self.gp_optimizer.kill_async_train(self.opt_obj)
        except: print("Tried to kill the client, but it appears there was none.")

    def update_hps(self):
        print("The Autonomous Experimenter is trying to update the hyperparameters.")
        if self.async_train_in_progress:
            self.gp_optimizer.update_hyperparameters(self.opt_obj)
            print("The Autonomus Experimenter updated the hyperparameters")
        else: print("The autonomous experimenter could not find an instance of asynchronous training. Therefore, no update.")
        print("hps: ", self.gp_optimizer.hyperparameters)

    def _init_costs(self,cost_func_params):
        self.gp_optimizer.init_cost(self.cost_func,cost_func_params,
                cost_update_function = self.cost_update_func)

    def tell(self, x,y,v, vp  = None):
        if vp is None: self.gp_optimizer.tell(x, y,variances = v)
        else: self.gp_optimizer.tell(x, y,variances = v, value_positions = vp)

    def extract_data(self):
        x,y,v,t,c = self.data.extract_data()
        return x,y,v,t,c, None
    ###################################################################################
    def go(self, N = 1e15, breaking_error = 1e-50,
            retrain_globally_at = [100,400,1000],
            retrain_locally_at = [20,40,60,80,100,200,400,1000],
            retrain_async_at = [1000,2000,5000,10000],
            retrain_callable_at = [],
            update_cost_func_at = [],
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
            * update_cost_func_at = []: list containing numbers when the cost function is updated
            * acq_func_opt_setting= lambda function to decide when global, local, hgdl, other ask()
            * training_opt_callable = None, callable
            * training_opt_max_iter = 20
            * training_opt_pop_size = 10
            * training_opt_tol      = 1e-6
            * acq_func_opt_max_iter = 20
            * acq_func_opt_pop_size = 20
            * acq_func_opt_tol      = 1e-6
            * acq_func_opt_tol_adjust = [True, 0.1]
            * number_of_suggested_measurements = 1
        """
        start_time = time.time()
        start_date_time = strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
        print("Date and time:       ", start_date_time)

        for i in range(self.init_dataset_size,int(N)):
            n_measurements = len(self.x)
            print("")
            print("")
            print("")
            print("==================================")
            print("==================================")
            print("iteration: ",i)
            print("Run Time: ", time.time() - start_time, "     seconds")
            print("Number of measurements: ", n_measurements)
            print("==================================")
            print("==================================")
            #ask() for new suggestions
            current_position = self.x[-1]
            print("hps: ",self.gp_optimizer.hyperparameters)
            local_method = acq_func_opt_setting(i)
            if number_of_suggested_measurements > 1: local_method = "hgdl"
            res = self.gp_optimizer.ask(
                    position = current_position,
                    n = number_of_suggested_measurements,
                    acquisition_function = self.acq_func,
                    cost_function = self.cost_func,
                    bounds = None,
                    method = local_method,
                    pop_size = acq_func_opt_pop_size,
                    max_iter = acq_func_opt_max_iter,
                    tol = acq_func_opt_tol,
                    dask_client = self.acq_func_opt_dask_client)
            #########################
            next_measurement_points = res["x"]
            func_evals = res["f(x)"]
            post_var = self.gp_optimizer.posterior_covariance(next_measurement_points)["v(x)"]
            error = np.max(np.sqrt(post_var[0]))
            if acq_func_opt_tol_adjust[0]:
                acq_func_opt_tol = abs(func_evals[0]) * acq_func_opt_tol_adjust[1]
                print("acquisition funciton optimization tolerance changed to: ", acq_func_opt_tol)
            print("Next points to be requested: ")
            print(next_measurement_points)
            #update and tell() new data
            info  = [{"hyperparameters" : self.gp_optimizer.hyperparameters,
                      "posterior std" : np.sqrt(post_var[j])} for j in range(len(next_measurement_points))]
            new_data = self.data.inject_arrays(next_measurement_points, info = info)
            print("Sending request to instrument ...")
            if self.append: self.data.dataset = self.data.dataset + self.instrument_func(new_data)
            else: self.data.dataset = self.instrument_func(self.data.dataset + new_data)
            print("Data received")
            print("Checking if data is clean ...")
            if self.data.nan_in_dataset(): self.data.clean_data_NaN()
            self.data.check_incoming_data()
            print("done")
            #update arrays and the gp_optimizer
            self.x,self.y, self.v, self.t, self.c,vp = self.extract_data()
            print("Communicating new data to GP")
            self.tell(self.x, self.y, self.v, vp)
            ###########################
            #train()
            print("++++++++++++++++++++++++++")
            print("|Training ...            |")
            print("++++++++++++++++++++++++++")
            if n_measurements in retrain_async_at:
                print("    Starting  a new asynchronous training after killing the current one.")
                self.kill_training()
                self.train_async(max_iter = 100000000,
                                 dask_client = self.training_dask_client)
            elif n_measurements in retrain_globally_at:
                self.kill_training()
                print("    Fresh optimization from scratch via global optimization")
                self.train(pop_size = 10,tol = 1e-6,max_iter = 20, method = "global")
            elif n_measurements in retrain_locally_at:
                self.kill_training()
                print("    Fresh optimization from scratch via global optimization")
                self.train(pop_size = 10,tol = 1e-6, max_iter = 20, method = "local")
            elif n_measurements in retrain_callable_at:
                self.kill_training()
                print("    Fresh optimization from scratch via user-defined optimization")
                self.train(pop_size = 10,tol = 1e-6,max_iter = 20, method = training_opt)
            else:
                print("    No training in this round but I am trying to update the hyperparameters")
                self.update_hps()
            print("++++++++++++++++++++++++++")
            print("|Training Done           |")
            print("++++++++++++++++++++++++++")

            ###save some data
            if self.run_every_iteration is not None: self.run_every_iteration(self)
            try: np.save('Data_'+ start_date_time, self.data.dataset)
            except Exception as e: print("Data not saved due to ", str(e))
            ###########################
            #cost update
            if i in update_cost_func_at: self.gp_optimizer.update_cost_function(self.c)

            if error < breaking_error: break
        self.kill_client()
        print("====================================================")
        print("The autonomous experiment was concluded successfully")
        print("====================================================")

###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
class AutonomousExperimenterFvGP(AutonomousExperimenterGP):
    def __init__(self,
            parameter_bounds,
            output_number,
            output_dim,
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
            x = None, y = None, v = None, vp = None, dataset = None,
            append_data_after_send = False,
            compute_device = "cpu",
            sparse = False,
            training_dask_client = None,
            acq_func_opt_dask_client = None
            ):
        dim = len(parameter_bounds)
        self.instrument_func = instrument_func
        self.hyperparameters = hyperparameters
        self.hyperparameter_bounds = hyperparameter_bounds
        self.acq_func = acq_func
        self.cost_func = cost_func
        self.cost_update_func = cost_update_func
        self.kernel_func = kernel_func
        self.prior_mean_func = prior_mean_func
        self.run_every_iteration = run_every_iteration
        self.append = append_data_after_send
        self.async_train_in_progress = False
        self.training_dask_client = training_dask_client
        if self.training_dask_client is None: self.training_dask_client = dask.distributed.Client()
        self.acq_func_opt_dask_client = acq_func_opt_dask_client
        if self.acq_func_opt_dask_client is None: self.acq_func_opt_dask_client = self.training_dask_client
        ################################
        #getting the data ready#########
        ################################
        if init_dataset_size is None and x is None and dataset is None:
            raise Exception("Either provide length of initial data or an inital dataset")
        self.data = fvgpData(dim, parameter_bounds,
                output_number = output_number, output_dim = output_dim)
        if x is None and dataset is None:
            self.data.create_random_dataset(init_dataset_size)
            self.data.dataset = self.instrument_func(self.data.dataset)
        elif dataset is not None:
            self.data.inject_dataset(list(np.load(dataset, allow_pickle = True)))
            self.hyperparameters = self.data.dataset[-1]["hyperparameters"]
        elif x is not None and y is not None:
            self.data.dataset = self.data.inject_arrays(x,y=y,v=v)
        elif x is not None and y is None:
            self.data.dataset = self.instrument_func(self.data.inject_arrays(x,y=y,v=v))
        else: raise Exception("No viable option for data given!")
        if self.data.nan_in_dataset(): self.data.clean_data_NaN()
        self.x, self.y, self.v, self.t, self.c, self.vp = self.data.extract_data()
        self.init_dataset_size = len(self.x)
        ######################
        ######################
        ######################
        self.gp_optimizer = fvGPOptimizer(dim,output_dim,output_number,parameter_bounds)
        self.gp_optimizer.tell(self.x, self.y,variances = self.v,value_positions = self.vp)
        self.gp_optimizer.init_fvgp(self.hyperparameters,compute_device = compute_device,
            gp_kernel_function = self.kernel_func,
            gp_mean_function = self.prior_mean_func,
            sparse = sparse)
        #init costs
        self._init_costs(cost_func_params)
        print("##################################################################################")
        print("Autonomous Experimenter fvGP initialization successfully concluded")
        print("now train(...) or train_async(...), and then go(...)")
        print("##################################################################################")

    def extract_data(self):
        x,y,v,t,c,vp = self.data.extract_data()
        return x,y,v,t,c,vp


#class AutonomousExperimenterEnsembleGP():
