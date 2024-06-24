# /usr/bin/env python
import inspect
import time
import dask
import sys
import numpy as np
from loguru import logger
from gpcam.data import fvgpData, gpData
from gpcam.gp_optimizer import GPOptimizer, fvGPOptimizer


#TODO
#   docstrings
#   

class AutonomousExperimenterGP:
    """
    Executes the autonomous loop for a single-task Gaussian process.
    Use class AutonomousExperimenterFvGP for multitask experiments.
    The AutonomousExperimenter is a convenience-driven functionality that does not allow
    as much customization as using the GPOptimizer directly. But it is a great option to
    start with.

    Parameters
    ----------
    input_space : np.ndarray
        A numpy array of floats of shape D x 2 describing the input space (bounds).
    hyperparameters : np.ndarray, optional
        Vector of hyperparameters used by the GP initially.
        This class provides methods to train hyperparameters.
        The default is a random draw from a uniform distribution
        within hyperparameter_bounds, with a shape appropriate
        for the default kernel (D + 1), which is an anisotropic Matern
        kernel with automatic relevance determination (ARD). If gp2Scale is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
    hyperparameter_bounds : np.ndarray, optional
        A 2d numpy array of shape (N x 2), where N is the number of needed hyperparameters.
        The default is None, in which case the hyperparameter_bounds are estimated from the domain size
        and the initial y_data. If the data changes significantly,
        the hyperparameters and the bounds should be changed/retrained. Initial hyperparameters and bounds
        can also be set in the train calls. The default only works for the default kernels.
    instrument_function : Callable, optional
         A function that takes data points (a list of dicts), and returns the same
         with the measurement data filled in. The function is
         expected to communicate with the instrument and perform measurements,
         populating fields of the data input. `y_data` and `noise variance` have to be filled in.
    init_dataset_size : int, optional
        If `x` and `y` are not provided and `dataset` is not provided,
        `init_dataset_size` must be provided. An initial
        dataset is constructed randomly with this length. The `instrument_function`
        is immediately called to measure values
        at these initial points.
    acquisition_function : Callable, optional
        The acquisition function accepts as input a numpy array
        of size V x D (such that V is the number of input
        points, and D is the parameter space dimensionality) and
        a `GPOptimizer` object. The return value is 1-D array
        of length V providing 'scores' for each position,
        such that the highest scored point will be measured next.
        Built-in functions can be used by one of the following keys:
        `ucb`,`lcb`,`maximum`,
        `minimum`, `variance`,`expected_improvement`,
        `relative information entropy`,`relative information entropy set`,
        `probability of improvement`, `gradient`,`total correlation`,`target probability`.
        If None, the default function `variance`, meaning
        `fvgp.GP.posterior_covariance` with variance_only = True will be used.
        The acquisition function can be a callable function of the form my_func(x,gpcam.GPOptimizer)
        which will be maximized (!!!), so make sure desirable new measurement points
        will be located at maxima.
        Explanations of the acquisition functions:
        variance: simply the posterior variance
        relative information entropy: the KL divergence of the prior over predictions and the posterior
        relative information entropy set: the KL divergence of the prior
        defined over predictions and the posterior point-by-point
        ucb: upper confidence bound, posterior mean + 3. std
        lcb: lower confidence bound, -(posterior mean - 3. std)
        maximum: finds the maximum of the current posterior mean
        minimum: finds the maximum of the current posterior mean
        gradient: puts focus on high-gradient regions
        probability of improvement: as the name would suggest
        expected improvement: as the name would suggest
        total correlation: extension of mutual information to more than 2 random variables
        target probability: probability of a target; needs a dictionary
        GPOptimizer.args = {'a': lower bound, 'b': upper bound} to be defined.
    cost_function : Callable, optional
        A function encoding the cost of motion through the input space and the
        cost of a measurements. Its inputs are an
        `origin` (np.ndarray of size V x D), `x` (np.ndarray of size V x D),
        and the value of `cost_func_params`;
        `origin` is the starting position, and `x` is the destination position. The return value is a 1-D array of
        length V describing the costs as floats. The 'score' from
        acquisition_function is divided by this returned cost to determine
        the next measurement point. If None, the default is a uniform cost of 1.
    cost_update_function : Callable, optional
        A function that updates the `cost_func_params` which are communicated to the `cost_function`.
        This function accepts as input
        costs (a list of cost values determined by `instrument_function`), bounds (a V x 2 numpy array) and a parameters
        object. The default is a no-op.
    cost_function_parameters : Any, optional
        An object that is communicated to the `cost_function` and `cost_update_function`. The default is `{}`.
    online : bool, optional
        The default is False. `online=True` will lead to calls to `gpOptimizer.tell(append=True)` which
        potentially saves a lot of time in the GP update.
        This, together with `calc_inv=True` leads to fast online performance.
    kernel_function : Callable, optional
        A symmetric positive semi-definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters, obj).
        The input x1 is a N1 x D array of positions, x2 is a N2 x D
        array of positions, the hyperparameters argument
        is a 1d array of length D+1 for the default kernel and of a different
        user-defined length for other kernels
        obj is an `fvgp.GP` instance. The default is a stationary anisotropic kernel
        (`fvgp.GP.default_kernel`) which performs automatic relevance determination (ARD).
        The output is a covariance matrix, an N1 x N2 numpy array.
    prior_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x D), hyperparameters (a 1d array of length D+1 for the default kernel)
        and a `fvgp.GP` instance. The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used.
    noise_function : Callable optional
        The noise function is a callable function f(x,hyperparameters,obj) that returns a
        positive symmetric definite matrix of shape(len(x),len(x)).
        The input x is a numpy array of shape (N x D). The hyperparameter array is the same
        that is communicated to mean and kernel functions. The obj is a fvgp.GP instance.
    run_every_iteration : Callable, optional
        A function that is run at every iteration. It accepts as input a
        `gpcam.AutonomousExperimenterGP` instance. The default is a no-op.
    x_data : np.ndarray, optional
        Initial data point positions.
    y_data : np.ndarray, optional
        Initial data point values.
    noise_variances : np.ndarray, optional
        Initial data point observation variances.
    dataset : string, optional
        A filename of a gpcam-generated file that is used to initialize a new instance.
    communicate_full_dataset : bool, optional
        If True, the full dataset will be communicated to the `instrument_function`
        on each iteration. If False, only the
        newly suggested data points will be communicated. The default is False.
    compute_device : str, optional
        One of "cpu" or "gpu", determines how linear system solves are run. The default is "cpu".
    calc_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance
        matrix after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster. Together with `online=True`
        and `communicate_full_dataset=False` this leads to very fast online execution.
        The default is True. Note, the training will always use Cholesky or LU decomposition instead of the
        inverse for stability reasons.
    training_dask_client : distributed.client.Client, optional
        A Dask Distributed Client instance for distributed training. If None is provided, a new
        `dask.distributed.Client` instance is constructed.
    acq_func_opt_dask_client : distributed.client.Client, optional
        A Dask Distributed Client instance for distributed `acquisition_function`
        computation. If None is provided, a new
        `dask.distributed.Client` instance is constructed.
    info : bool, optional
        Specifies if info should be displayed. Default = False.


    Attributes
    ----------
    x_data : np.ndarray
        Data point positions
    y_data : np.ndarray
        Data point values
    noise_variances : np.ndarray
        Data point observation variances
    data.dataset : list
        All data
    hyperparameter_bounds : np.ndarray
        A 2d array of floats of size J x 2, such that J is the length
        matching the length of `hyperparameters` defining
        the bounds for training.
    gp_optimizer : gpcam.GPOptimizer
        A GPOptimizer instance used for initializing a Gaussian process and performing optimization of the posterior.
    """

    def __init__(self,
                 input_space,
                 hyperparameters=None,
                 hyperparameter_bounds=None,
                 instrument_function=None,
                 init_dataset_size=None,
                 acquisition_function="variance",
                 cost_function=None,
                 cost_update_function=None,
                 cost_function_parameters=None,
                 online=False,
                 kernel_function=None,
                 prior_mean_function=None,
                 noise_function=None,
                 run_every_iteration=None,
                 x_data=None, y_data=None, noise_variances=None, dataset=None,
                 communicate_full_dataset=False,
                 compute_device="cpu",
                 calc_inv=True,
                 training_dask_client=None,
                 acq_func_opt_dask_client=None,
                 gp2Scale=False,
                 gp2Scale_dask_client=None,
                 gp2Scale_batch_size=10000,
                 ram_economy=True,
                 info=False,
                 args=None
                 ):
        if info:
            logger.enable('gpcam')
            logger.add(sys.stdout, filter="gpcam", level="INFO")
            logger.enable('fvgp')
            logger.add(sys.stdout, filter="fvgp", level="INFO")
            logger.enable('hgdl')
            logger.add(sys.stdout, filter="hgdl", level="INFO")
        else:
            logger.disable('gpcam')
            logger.disable('fvgp')
            logger.disable('hgdl')

        if cost_function_parameters is None:
            cost_function_parameters = {}

        dim = len(input_space)
        self.input_space = input_space
        self.instrument_function = instrument_function
        self.hyperparameter_bounds = hyperparameter_bounds
        self.hyperparameters = hyperparameters
        self.acquisition_function = acquisition_function
        self.run_every_iteration = run_every_iteration
        self.communicate_full_dataset = communicate_full_dataset
        self.async_train_in_progress = False
        self.training_dask_client = training_dask_client
        self.acq_func_opt_dask_client = acq_func_opt_dask_client
        self.args = args
        self.online = online
        self.costs = None
        self.acq_func_max_opt_obj = None
        self.multi_task = False
        self.vp = []
        ################################
        # getting the data ready#########
        ################################
        if init_dataset_size is None and x_data is None and dataset is None:
            raise Exception("Either provide length of initial data or an initial dataset")
        self.data = gpData(dim, input_space)
        if x_data is None and dataset is None:
            self.data.create_random_dataset(init_dataset_size)
            if instrument_function is None: raise Exception("You need to provide an instrument function.")
            self.data.update_dataset(self.instrument_function(self.data.dataset))
        elif dataset is not None:
            self.data.inject_dataset(list(np.load(dataset, allow_pickle=True)))
            hyperparameters = self.data.dataset[-1]["hyperparameters"]
        elif x_data is not None and y_data is not None and noise_variances is not None:
            self.data.update_dataset(self.data.arrays2data(x_data, y=y_data, v=noise_variances))
        elif x_data is not None and y_data is None:
            if instrument_function is None: raise Exception("You need to provide an instrument function.")
            self.data.update_dataset(self.instrument_function(self.data.arrays2data(x_data,
                                                                                    y=y_data, v=noise_variances)))
        else:
            raise Exception("No viable option for data given!")
        self.data.check_incoming_data()
        if self.data.nan_in_dataset(): self.data.clean_data_NaN()
        self.x_data, self.y_data, self.noise_variances, self.times, self.cost = self.data.extract_data()

        self.init_dataset_size = len(self.x_data)
        ######################
        ######################
        ######################
        self.opt_obj = None
        self.gp_optimizer = GPOptimizer(x_data=self.x_data,
                                        y_data=self.y_data,
                                        init_hyperparameters=hyperparameters,
                                        noise_variances=self.noise_variances,
                                        compute_device=compute_device,
                                        gp_kernel_function=kernel_function,
                                        gp_kernel_function_grad=None,
                                        gp_noise_function=noise_function,
                                        gp_noise_function_grad=None,
                                        gp_mean_function=prior_mean_function,
                                        gp_mean_function_grad=None,
                                        gp2Scale=gp2Scale,
                                        gp2Scale_dask_client=gp2Scale_dask_client,
                                        gp2Scale_batch_size=gp2Scale_batch_size,
                                        calc_inv=calc_inv,
                                        ram_economy=ram_economy,
                                        args=args,
                                        info=info,
                                        cost_function=cost_function,
                                        cost_function_parameters=cost_function_parameters,
                                        cost_update_function=cost_update_function)

        # init costs
        logger.info(inspect.cleandoc("""#
        ##################################################################################
        Autonomous Experimenter initialization successfully concluded
        now train(...) or train_async(...), and then go(...)
        ##################################################################################"""))

    ###################################################################################
    def train(self, init_hyperparameters=None, pop_size=10, tol=0.0001, max_iter=20, method="global", constraints=()):
        """
        This function finds the maximum of the log marginal likelihood and therefore trains the GP (synchronously).
        The GP prior will automatically be updated with the new hyperparameters after the training.

        Parameters
        ----------
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is a random draw from a uniform distribution within the bounds.
        pop_size : int, optional
            A number of individuals used for any optimizer with a global component. Default = 20.
        tol : float, optional
            Used as termination criterion for local optimizers. Default = 0.0001.
        max_iter : int, optional
            Maximum number of iterations for global and local optimizers. Default = 120.
        method : str, optional
            Method to be used for the training. Default is `global` which means
            a differential evolution algorithm is run with the specified parameters.
            The options are `global` or `local`, or `mcmc`.
        constraints : tuple of object instances, optional
            Equality and inequality constraints for the optimization.
            If the optimizer is `hgdl` see `hgdl.readthedocs.io`.
            If the optimizer is a scipy optimizer, see the scipy documentation.
        """

        self.gp_optimizer.train(
            hyperparameter_bounds=self.hyperparameter_bounds,
            init_hyperparameters=init_hyperparameters,
            method=method,
            pop_size=pop_size,
            tolerance=tol,
            max_iter=max_iter,
            constraints=constraints)

    def train_async(self, init_hyperparameters=None, max_iter=10000, local_method="L-BFGS-B", global_method="genetic",
                    constraints=()):
        """
        Function to train the Gaussian Process asynchronously using the HGDL optimizer. 
        The use is entirely optional; this function will be called
        as part of the go() command, if so specified. This call starts a highly parallelized optimization process,
        on an architecture specified by the dask.distributed.Client. The main purpose of this function is to
        allow for large-scale distributed training. 

        Parameters
        ----------
        init_hyperparameters : np.ndarray, optional
            Initial hyperparameters used as starting location for all optimizers with local component.
            The default is a random draw from a uniform distribution within the bounds.
        max_iter : int, optional
            Maximum number of iterations for the global method. Default = 10000
            It is important to remember here that the call is run asynchronously, so
            this number does not affect run time.
        local_method : str, optional
            Local method to be used inside HGDL. Many scipy.optimize.minimize methods
            can be used, or a user-defined callable. Please read the HGDL docs for more information.
            Default = `L-BFGS-B`.
        global_method : str, optional
            Local method to be used inside HGDL. Please read the HGDL docs for more information.
            Default = `genetic`.
        constraints : tuple of object instances, optional
            Equality and inequality constraints for the optimization.
            See `hgdl.readthedocs.io` for setting up constraints.
        """

        if self.training_dask_client is None: self.training_dask_client = dask.distributed.Client()

        logger.info("AutonomousExperimenter starts async training with dask client:")
        self.opt_obj = self.gp_optimizer.train_async(
            hyperparameter_bounds=self.hyperparameter_bounds,
            init_hyperparameters=init_hyperparameters,
            max_iter=max_iter,
            local_optimizer=local_method,
            global_optimizer=global_method,
            constraints=constraints,
            dask_client=self.training_dask_client
        )
        logger.info("The Autonomous Experimenter started an instance of the asynchronous training.")
        self.async_train_in_progress = True

    def kill_training(self):
        """
        Function to stop an asynchronous training. This leaves the dask.distributed.Client alive.
        """
        if self.async_train_in_progress:
            self.gp_optimizer.stop_training(self.opt_obj)
            self.async_train_in_progress = False
        else:
            logger.info("no training to be killed")

    def kill_all_clients(self):
        """
        Function to kill both dask.distributed.Client instances.
        Will be called automatically at the end of go().
        """
        try:
            self.training_dask_client.close()
            self.acq_func_opt_dask_client.close()
            self.async_train_in_progress = False
        except Exception as ex:
            logger.error(str(ex))
            logger.error("Killing of the clients failed. Please do so manually before initializing a new one.")

    def update_hps(self):
        """
        Function to update the hyperparameters if an asynchronous training is running.
        Will be called during go() as specified.
        """
        if self.async_train_in_progress:
            self.gp_optimizer.update_hyperparameters(self.opt_obj)
            logger.info("The Autonomous Experimenter updated the hyperparameters")
        else:
            logger.warning(
                "The autonomous experimenter could not find an instance of asynchronous training. Therefore no update.")
        logger.info("hps: {}", self.gp_optimizer.hyperparameters)

    def _tell(self, x, y, v, vp=None, append=True):
        if not self.multi_task: self.gp_optimizer.tell(x, y, noise_variances=v, append=append)
        else: self.gp_optimizer.tell(x, y, noise_variances=v, output_positions=vp, append=append)

    def _ask(self,
             bounds=None,
             position=None,
             x_out=None,
             n=1,
             acquisition_function="variance",
             method="global",
             pop_size=20,
             max_iter=20,
             tol=1e-6,
             constraints=(),
             dask_client=None):

        if not self.multi_task:
            res = self.gp_optimizer.ask(
                bounds,
                position=position,
                n=n,
                acquisition_function=acquisition_function,
                method=method,
                pop_size=pop_size,
                max_iter=max_iter,
                tol=tol,
                constraints=constraints,
                dask_client=dask_client)
        else:
            res = self.gp_optimizer.ask(
                bounds,
                x_out,
                position=position,
                n=n,
                acquisition_function=acquisition_function,
                method=method,
                pop_size=pop_size,
                max_iter=max_iter,
                tol=tol,
                constraints=constraints,
                dask_client=dask_client)
        return res

    def _extract_data(self):
        x, y, v, t, c = self.data.extract_data()
        return x, y, v, t, c, np.zeros(len(c))

    ###################################################################################
    def go(self, N=1e15, breaking_error=1e-50,
           retrain_globally_at=(20, 50, 100, 400),
           retrain_locally_at=(20, 40, 60, 80, 100, 200, 400, 1000),
           retrain_async_at=(),
           update_cost_func_at=(),
           acq_func_opt_setting=lambda number: "global",
           training_opt_max_iter=20,
           training_opt_pop_size=10,
           training_opt_tol=1e-6,
           acq_func_opt_max_iter=20,
           acq_func_opt_pop_size=20,
           acq_func_opt_tol=1e-6,
           acq_func_opt_tol_adjust=0.1,
           number_of_suggested_measurements=1,
           checkpoint_filename=None,
           constraints=(),
           break_condition_callable=lambda n: False
           ):
        """
        Function to start the autonomous-data-acquisition loop.

        Parameters
        ----------
        N : int, optional
            Run until N points are measured. The default is `1e15`.
        breaking_error : float, optional
            Run until breaking_error is achieved (or at max N). The default is `1e-50`.
        retrain_globally_at : Iterable [int], optional
            Retrains the hyperparameters at the given number of measurements using global optimization. The default is
            `[20,50,100,400,1000]`.
        retrain_locally_at : Iterable[int], optional
            Retrains the hyperparameters at the given number of measurements using local gradient-based optimization.
            The default is `[20,40,60,80,100,200,400,1000]`.
        retrain_async_at : Iterable[int], optional
            Retrains the hyperparameters at the given number of measurements using the HGDL algorithm. This training is
            asynchronous and can be run in a distributed fashion using `training_dask_client`. The default is `[]`.
        update_cost_func_at : Iterable[int], optional
            Calls the `update_cost_function` at the given number of measurements.
            Default = ()
        acq_func_opt_setting : Callable, optional
            A callable that accepts as input the iteration index and returns either `local`, `global`, `hgdl`.
            This switches between local gradient-based, global and hybrid optimization for the acquisition function.
            The default is `lambda number: "global" if number % 2 == 0 else "local"`.
        training_opt_max_iter : int, optional
            The maximum number of iterations for any training.
            The default value is 20.
        training_opt_pop_size : int, optional
            The population size used for any training with a global component (HGDL or standard global optimizers).
            The default value is 10.
        training_opt_tol : float, optional
            The optimization tolerance for all training optimization. The default is 1e-6.
        acq_func_opt_max_iter : int, optional
            The maximum number of iterations for the `acquisition_function` optimization. The default is 20.
        acq_func_opt_pop_size : int, optional
            The population size used for any `acquisition_function` optimization with a global
            component (HGDL or standard global
            optimizers). The default value is 20.
        acq_func_opt_tol : float, optional
            The optimization tolerance for all `acquisition_function` optimization.
            The default value is 1e-6
        acq_func_opt_tol_adjust : float, optional
            The `acquisition_function` optimization tolerance is adjusted at every iteration as a
            fraction of this value.
            The default value is 0.1 .
        number_of_suggested_measurements : int, optional
            The algorithm will try to return this many suggestions for new measurements. This may be limited by how many
            optima the algorithm may find. If greater than 1, then the `acquisition_function`
            optimization method is automatically
            set to use HGDL. The default is 1.
        checkpoint_filename : str, optional
            When provided, a checkpoint of all the accumulated data will be written to this file on each iteration.
        constraints : tuple, optional
            If provided, this subjects the acquisition function optimization to constraints.
            For the definition of the constraints, follow
            the structure your chosen optimizer requires.
        break_condition_callable : Callable, optional
            Autonomous loop will stop when this function returns True. The function takes as
            input a gpcam.AutonomousExperimenterGP instance.
        """
        # set up
        start_time = time.time()
        start_date_time = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
        logger.info("Starting the autonomous loop...")
        i = 0
        n_measurements = len(self.x_data)
        # start the loop
        while n_measurements < N:
            logger.info("----------------------------")
            logger.info(f"iteration {i}")
            logger.info(f"Run Time: {time.time() - start_time} seconds")
            logger.info(f"Number of measurements {n_measurements}")
            logger.info("current hps: {}", self.gp_optimizer.hyperparameters)

            ###################################################
            # ask() for new suggestions########################
            ###################################################
            current_position = self.x_data[-1]
            current_method = acq_func_opt_setting(self)
            if number_of_suggested_measurements > 1 and current_method != "hgdl": current_method = "global"
            if current_method == "hgdl" and self.acq_func_opt_dask_client is None:
                self.acq_func_opt_dask_client = dask.distributed.Client()
            if self.multi_task:
                x_out = self.data.dataset[-1]["output positions"]
            else:
                x_out = None

            res = self._ask(
                self.input_space,
                position=current_position,
                x_out=x_out,
                n=number_of_suggested_measurements,
                acquisition_function=self.acquisition_function,
                method=current_method,
                pop_size=acq_func_opt_pop_size,
                max_iter=acq_func_opt_max_iter,
                tol=acq_func_opt_tol,
                constraints=constraints,
                dask_client=self.acq_func_opt_dask_client)

            self.acq_func_max_opt_obj = res["opt_obj"]
            next_measurement_points = res["x"]
            func_evals = res["f(x)"]

            ###################################################
            # assess performance###############################
            ###################################################
            if self.multi_task:
                a = np.array(next_measurement_points)
                b = np.array(self.vp[-1])
                test_points = np.array([np.append(a[i], b[j]) for i in range(len(a)) for j in range(len(b))])
                post_var = self.gp_optimizer.posterior_covariance(test_points)["v(x)"]
            else:
                post_var = self.gp_optimizer.posterior_covariance(next_measurement_points)["v(x)"]
            error = np.max(np.sqrt(post_var))

            ###################################################
            # adjust tolerances if necessary###################
            ###################################################
            if acq_func_opt_tol_adjust:
                acq_func_opt_tol = abs(func_evals[0]) * acq_func_opt_tol_adjust
                logger.info("acquisition function optimization tolerance changed to: {}", acq_func_opt_tol)
            logger.info("Next points to be requested: ")
            logger.info(next_measurement_points)

            ###################################################
            # send suggestions to instrument and get results###
            ###################################################
            info = [{"hyperparameters": self.gp_optimizer.hyperparameters,
                     "posterior std": np.sqrt(post_var[j])} for j in range(len(next_measurement_points))]
            new_data = self.data.arrays2data(next_measurement_points, info=info)
            logger.info("Sending request to instrument ...")
            if self.communicate_full_dataset:
                if self.online: raise Exception("You specified online=True but you communicated the whole dataset."
                                                "This violates the security protocol.")
                self.data.update_dataset(self.instrument_function(self.data.dataset + new_data))
                len_of_new_data_received = len(self.data.dataset)
            else:
                new_data_received = self.instrument_function(new_data)
                len_of_new_data_received = len(new_data_received)
                self.data.update_dataset(self.data.dataset + new_data_received)
            logger.info("Data received")
            logger.info("Checking if data is clean ...")
            self.data.check_incoming_data()
            if self.data.nan_in_dataset(): self.data.clean_data_NaN()
            # update arrays and the gp_optimizer
            self.x_data, self.y_data, self.noise_variances, self.times, self.costs, self.vp = self._extract_data()
            logger.info("Communicating new data to the GP")

            ###################################################
            # tell() the GP about new data#####################
            ###################################################
            if self.online and i % 5 == 0 and error > 0.0:
                self._tell(self.x_data[-len_of_new_data_received:],
                           self.y_data[-len_of_new_data_received:],
                           self.noise_variances[-len_of_new_data_received:],
                           self.vp[-len_of_new_data_received:], append=True)
            else:
                self._tell(self.x_data,
                           self.y_data,
                           self.noise_variances,
                           self.vp, append=False)

            ###################################################
            # train() the GP###################################
            ###################################################
            if any(n in retrain_async_at for n in range(n_measurements, len(self.x_data))) and n_measurements < N:
                if self.training_dask_client is None: self.training_dask_client = dask.distributed.Client()
                logger.info("    Starting a new asynchronous training after killing the current one.")
                self.kill_training()
                self.train_async(max_iter=10000)
            elif any(n in retrain_globally_at for n in range(n_measurements, len(self.x_data))) and n_measurements < N:
                self.kill_training()
                logger.info("    Fresh optimization from scratch via global optimization")
                self.train(pop_size=training_opt_pop_size,
                           tol=training_opt_tol,
                           max_iter=training_opt_max_iter,
                           method="global")
            elif any(n in retrain_locally_at for n in range(n_measurements, len(self.x_data))) and n_measurements < N:
                self.kill_training()
                logger.info("    Fresh optimization from scratch via local optimization")
                self.train(pop_size=training_opt_pop_size,
                           tol=training_opt_tol,
                           max_iter=training_opt_max_iter,
                           method="local")
            else:
                logger.info("    No training in this round but I am trying to update the hyperparameters")
                self.update_hps()
            logger.info("    Training successfully concluded")

            ###################################################
            # run a user-defined callable######################
            ###################################################
            if self.run_every_iteration is not None: self.run_every_iteration(self)

            ###################################################
            # save some data###################################
            ###################################################
            if checkpoint_filename:
                try:
                    np.save(checkpoint_filename, self.data.dataset)
                except Exception as e:
                    raise RuntimeError("Data not saved") from e

            ###################################################
            # update costs#####################################
            ###################################################
            if i in update_cost_func_at: self.gp_optimizer.update_cost_function(self.costs)

            ###################################################
            # break check######################################
            ###################################################
            logger.info("Current error: {}", error)
            if breaking_error > error > 0.:
                logger.info("Breaking error has been exceeded. {}", error)
                break
            if break_condition_callable(self): break
            # update iteration numbers
            i += 1
            n_measurements = len(self.x_data)

        # clean up
        logger.debug("killing the client... and then we are done")
        self.kill_all_clients()

        logger.info(inspect.cleandoc("""#
            ====================================================
            The autonomous experiment was concluded successfully
            ===================================================="""))


###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
class AutonomousExperimenterFvGP(AutonomousExperimenterGP):
    """
    Executes the autonomous loop for a multitask Gaussian process.

    Parameters
    ----------
    input_space : np.ndarray
        A numpy array of floats of shape D x 2 describing the input space (bounds).
    output_number : int
        An integer defining how many outputs are created by each measurement.
    hyperparameters : np.ndarray, optional
        Vector of hyperparameters used by the GP initially.
        This class provides methods to train hyperparameters.
        The default is a random draw from a uniform distribution
        within hyperparameter_bounds, with a shape appropriate
        for the default kernel (D + 1), which is an anisotropic Matern
        kernel with automatic relevance determination (ARD). If gp2Scale is
        enabled, the default kernel changes to the anisotropic Wendland kernel.
    hyperparameter_bounds : np.ndarray, optional
        A 2d numpy array of shape (N x 2), where N is the number of needed hyperparameters.
        The default is None, in which case the hyperparameter_bounds are estimated from the domain size
        and the initial y_data. If the data changes significantly,
        the hyperparameters and the bounds should be changed/retrained. Initial hyperparameters and bounds
        can also be set in the train calls. The default only works for the default kernels.
    instrument_function : Callable, optional
         A function that takes data points (a list of dicts), and returns the same
         with the measurement data filled in. The function is
         expected to communicate with the instrument and perform measurements,
         populating fields of the data input. `y_data` and `noise variances` have to be filled in.
    init_dataset_size : int, optional
        If `x` and `y` are not provided and `dataset` is not provided,
        `init_dataset_size` must be provided. An initial
        dataset is constructed randomly with this length. The `instrument_function`
        is immediately called to measure values
        at these initial points.
    acquisition_function : Callable, optional
        The acquisition function accepts as input a numpy array
        of size V x D (such that V is the number of input
        points, and D is the parameter space dimensionality) and
        a `GPOptimizer` object. The return value is 1d array
        of length V providing 'scores' for each position,
        such that the highest scored point will be measured next.
        Built-in functions can be used by one of the following keys:
        `variance`, `relative information entropy`,
        `relative information entropy set`, `total correlation`.
        See GPOptimizer.ask() for a short explanation of these functions.
        In the multitask case, it is highly recommended to
        deploy a user-defined acquisition function due to the intricate relationship
        of posterior distributions at different points in the output space.
        If None, the default function `variance`, meaning
        `fvgp.GP.posterior_covariance` with variance_only = True will be used.
        The acquisition function can be a callable function of the form my_func(x,gpcam.GPOptimizer)
        which will be maximized (!!!), so make sure desirable new measurement points
        will be located at maxima.
    cost_function : Callable, optional
        A function encoding the cost of motion through the input space and the
        cost of a measurements. Its inputs are an
        `origin` (np.ndarray of size V x D), `x` (np.ndarray of size V x D),
        and the value of `cost_function_parameters`;
        `origin` is the starting position, and `x` is the destination position. The return value is a 1d array of
        length V describing the costs as floats. The 'score' from
        acquisition_function is divided by this returned cost to determine
        the next measurement point. If None, the default is a uniform cost of 1.
    cost_update_function : Callable, optional
        A function that updates the `cost_func_params` which are communicated to the `cost_function`.
        This function accepts as input
        costs (a list of cost values determined by `instrument_function`), bounds (a V x 2 numpy array) and a parameters
        object. The default is a no-op.
    cost_function_parameters : Any, optional
        An object that is communicated to the `cost_function` and `cost_update_function`. The default is `{}`.
    online : bool, optional
        The default is False. `online=True` will lead to calls to `gpOptimizer.tell(append=True)` which
        potentially saves a lot of time in the GP update.
        This, together with `calc_inv=True` leads to fast online performance.
    kernel_function : Callable, optional
        A symmetric positive semi-definite covariance function (a kernel)
        that calculates the covariance between
        data points. It is a function of the form k(x1,x2,hyperparameters, obj).
        The input x1 is a N1 x Di+Do array of positions, x2 is a N2 x Di+Do
        array of positions, the hyperparameters argument
        is a 1d array of length N depending on how many hyperparameters are initialized, and
        obj is an `fvgp.GP` instance. The default is a deep kernel with 2 hidden layers and
        a width of fvgp.fvGP.gp_deep_kernel_layer_width.
    prior_mean_function : Callable, optional
        A function that evaluates the prior mean at a set of input position. It accepts as input
        an array of positions (of shape N1 x Di+Do), hyperparameters
        and a `fvgp.GP` instance. The return value is a 1d array of length N1. If None is provided,
        `fvgp.GP._default_mean_function` is used.
    run_every_iteration : Callable, optional
        A function that is run at every iteration. It accepts as input a
        `gpcam.AutonomousExperimenterGP` instance. The default is a no-op.
    x_data : np.ndarray, optional
        Initial data point positions.
    y_data : np.ndarray, optional
        Initial data point values.
    noise_variances : np.ndarray, optional
        Initial data point observation variances.
    vp : np.ndarray, optional
        A 2d numpy array of shape (U x output_number), so that for each measurement position, the outputs
        are clearly defined by their positions in the output space.
        The default is np.array([0,1,2,3,...,output_number - 1]) for each
        point in the input space.
    communicate_full_dataset : bool, optional
        If True, the full dataset will be communicated to the `instrument_function`
        on each iteration. If False, only the
        newly suggested data points will be communicated. The default is False.
    compute_device : str, optional
        One of "cpu" or "gpu", determines how linear system solves are run. The default is "cpu".
    calc_inv : bool, optional
        If True, the algorithm calculates and stores the inverse of the covariance
        matrix after each training or update of the dataset or hyperparameters,
        which makes computing the posterior covariance faster. Together with `online=True`
        and `communicate_full_dataset=False` this leads to very fast online execution.
        The default is True. Note, the training will always use Cholesky or LU decomposition instead of the
        inverse for stability reasons.
    training_dask_client : distributed.client.Client, optional
        A Dask Distributed Client instance for distributed training. If None is provided, a new
        `dask.distributed.Client` instance is constructed.
    acq_func_opt_dask_client : distributed.client.Client, optional
        A Dask Distributed Client instance for distributed `acquisition_function` computation.
        If None is provided, a new
        `dask.distributed.Client` instance is constructed.
    info : bool, optional
        Specifies if info should be displayed. Default = False


    Attributes
    ----------
    x_data : np.ndarray
        Data point positions
    y_data : np.ndarray
        Data point values
    noise_variances : np.ndarray
        Data point observation variances
    data.dataset : list
        All data
    hyperparameter_bounds : np.ndarray
        A 2d array of floats of size J x 2, such that J is the length matching
        the length of `hyperparameters` defining
        the bounds for training.
    gp_optimizer : gpcam.GPOptimizer
        A GPOptimizer instance used for initializing a Gaussian process and performing optimization of the posterior.

    """

    def __init__(self,
                 input_space,
                 output_number,
                 hyperparameters=None,
                 hyperparameter_bounds=None,
                 instrument_function=None,
                 init_dataset_size=None,
                 acquisition_function="variance",
                 cost_function=None,
                 cost_update_function=None,
                 cost_function_parameters=None,
                 online=False,
                 kernel_function=None,
                 prior_mean_function=None,
                 noise_function=None,
                 run_every_iteration=None,
                 x_data=None, y_data=None, noise_variances=None, vp=None, dataset=None,
                 communicate_full_dataset=False,
                 compute_device="cpu",
                 calc_inv=True,
                 training_dask_client=None,
                 acq_func_opt_dask_client=None,
                 gp2Scale=False,
                 gp2Scale_dask_client=None,
                 gp2Scale_batch_size=10000,
                 ram_economy=True,
                 info=False,
                 args=None
                 ):
        ################################
        # getting the data ready#########
        ################################
        if cost_function_parameters is None:
            cost_function_parameters = {}
        dim = len(input_space)
        self.input_space = input_space
        self.instrument_function = instrument_function
        self.hyperparameter_bounds = hyperparameter_bounds
        self.hyperparameters = hyperparameters
        self.acquisition_function = acquisition_function
        self.run_every_iteration = run_every_iteration
        self.communicate_full_dataset = communicate_full_dataset
        self.async_train_in_progress = False
        self.training_dask_client = training_dask_client
        self.acq_func_opt_dask_client = acq_func_opt_dask_client
        self.args = args
        self.online = online
        self.vp = vp

        if init_dataset_size is None and x_data is None and dataset is None:
            raise Exception("Either provide length of initial data or an initial dataset")
        self.data = fvgpData(dim, input_space, output_number=output_number)

        if x_data is None and dataset is None:
            self.data.create_random_dataset(init_dataset_size)
            if instrument_function is None: raise Exception("You need to provide an instrument function.")
            self.data.update_dataset(self.instrument_function(self.data.dataset))
        elif dataset is not None:
            self.data.inject_dataset(list(np.load(dataset, allow_pickle=True)))
            self.hyperparameters = self.data.dataset[-1]["hyperparameters"]
        elif x_data is not None and y_data is not None and noise_variances is not None:
            self.data.update_dataset(self.data.arrays2data(x_data, y=y_data, v=noise_variances, vp=self.vp))
        elif x_data is not None and y_data is None:
            if instrument_function is None: raise Exception("You need to provide an instrument function.")
            self.data.update_dataset(self.instrument_function(self.data.arrays2data(x_data,
                                                                                    y=y_data, v=noise_variances)))
        else:
            raise Exception("No viable option for data given!")
        self.data.check_incoming_data()
        if self.data.nan_in_dataset(): self.data.clean_data_NaN()
        self.x_data, self.y_data, self.noise_variances, self.times, self.costs, self.vp = self.data.extract_data()
        self.init_dataset_size = len(self.x_data)
        self.multi_task = True
        ######################
        ######################
        ######################
        self.gp_optimizer = fvGPOptimizer(
            self.x_data, self.y_data,
            init_hyperparameters=hyperparameters,
            noise_variances=self.noise_variances,
            compute_device=compute_device,
            gp_kernel_function=kernel_function,
            gp_kernel_function_grad=None,
            gp_noise_function=noise_function,
            gp_noise_function_grad=None,
            gp_mean_function=prior_mean_function,
            gp_mean_function_grad=None,
            gp2Scale=gp2Scale,
            gp2Scale_dask_client=gp2Scale_dask_client,
            gp2Scale_batch_size=gp2Scale_batch_size,
            calc_inv=calc_inv,
            ram_economy=ram_economy,
            args=args,
            info=info,
            cost_function=cost_function,
            cost_function_parameters=cost_function_parameters,
            cost_update_function=cost_update_function)

        logger.info(inspect.cleandoc("""#
        ##################################################################################
        Autonomous Experimenter fvGP initialization successfully concluded
        now train(...) or train_async(...), and then go(...)
        ##################################################################################"""))

    def _extract_data(self):
        x, y, v, t, c, vp = self.data.extract_data()
        return x, y, v, t, c, vp
