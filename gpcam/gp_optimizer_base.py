#!/usr/bin/env python
import numpy as np
from loguru import logger
from fvgp import fvGP
from fvgp import GP
from . import surrogate_model as sm
import warnings
import random
from distributed import Client


# TODO (for gpCAM):


class GPOptimizerBase(GP):
    def __init__(
            self,
            x_data=None,
            y_data=None,
            init_hyperparameters=None,
            noise_variances=None,
            compute_device="cpu",
            kernel_function=None,
            kernel_function_grad=None,
            noise_function=None,
            noise_function_grad=None,
            prior_mean_function=None,
            prior_mean_function_grad=None,
            gp2Scale=False,
            gp2Scale_dask_client=None,
            gp2Scale_batch_size=10000,
            gp2Scale_linalg_mode=None,
            calc_inv=False,
            ram_economy=False,
            cost_function=None,
            logging=False,
            multi_task=False,
            args=None,
    ):
        self.cost_function = cost_function
        self.init_hyperparameters = init_hyperparameters
        self.compute_device = compute_device
        self.kernel_function = kernel_function
        self.kernel_function_grad = kernel_function_grad
        self.noise_function = noise_function
        self.noise_function_grad = noise_function_grad
        self.prior_mean_function = prior_mean_function
        self.prior_mean_function_grad = prior_mean_function_grad
        self._gp2Scale = gp2Scale
        self.gp2Scale_dask_client = gp2Scale_dask_client
        self.gp2Scale_batch_size = gp2Scale_batch_size
        self.gp2Scale_linalg_mode = gp2Scale_linalg_mode
        self.calc_inv = calc_inv
        self.ram_economy = ram_economy
        self._args = args
        self.logging = logging
        self.multi_task = multi_task
        self.x_out = None

        if logging is True:
            logger.enable("gpcam")
            logger.enable("fvgp")
        else:
            logger.disable("gpcam")
            logger.disable("fvgp")

        self.gp = False
        if x_data is not None and y_data is not None:
            self._initializeGP(x_data, y_data, noise_variances=noise_variances)
        else:
            warnings.warn("GP has not been initialized. Call tell() before using any class method.")

    @property
    def x_data(self):
        if self.gp: return super().x_data
        else: return None

    @property
    def y_data(self):
        if self.gp: return super().y_data
        else: return None

    @property
    def noise_variances(self):
        if self.gp: return super().noise_variances
        else: return None

    @property
    def args(self):
        if self.gp: return super().args
        else: return self._args

    @args.setter
    def args(self, a):
        if self.gp: GP.args.fset(self, a)
        else: self._args = a

    @property
    def input_space_dimension(self):
        if self.gp:
            input_space_dimension = self.input_set_dim
        else: input_space_dimension = None
        return input_space_dimension

    def _initializeGP(self, x_data, y_data, noise_variances=None):
        """
        Function to initialize a GP object.
        If data is prided at initialization this function is NOT needed.
        It has the same parameters as the initialization of the class.
        """
        if self.multi_task: self.x_out = np.arange(y_data.shape[1])
        else: self.x_out = None

        super().__init__(
            x_data,
            y_data,
            init_hyperparameters=self.init_hyperparameters,
            noise_variances=noise_variances,
            compute_device=self.compute_device,
            kernel_function=self.kernel_function,
            kernel_function_grad=self.kernel_function_grad,
            noise_function=self.noise_function,
            noise_function_grad=self.noise_function_grad,
            prior_mean_function=self.prior_mean_function,
            prior_mean_function_grad=self.prior_mean_function_grad,
            gp2Scale=self._gp2Scale,
            gp2Scale_dask_client=self.gp2Scale_dask_client,
            gp2Scale_batch_size=self.gp2Scale_batch_size,
            gp2Scale_linalg_mode=self.gp2Scale_linalg_mode,
            calc_inv=self.calc_inv,
            ram_economy=self.ram_economy,
            args=self._args
        )
        self.gp = True

    def get_data(self):
        """
        Function that provides access to the class attributes.

        Return
        ------
        dictionary of class attributes : dict
        """
        if not self.gp: return "GP not yet initialized; tell() data!"

        if not self.multi_task:
            return {
                "input dim": self.input_space_dimension,
                "x data": self.x_data,
                "y data": self.y_data,
                "measurement variances": self.likelihood.V,
                "hyperparameters": self.hyperparameters,
                "cost function": self.cost_function}
        elif self.multi_task:
            return {
                "input dim": self.input_space_dimension,
                "x data": self.fvgp_x_data,
                "y data": self.fvgp_y_data,
                "transformed x data": self.x_data,
                "transformed y data": self.y_data,
                "measurement variances": self.likelihood.V,
                "hyperparameters": self.hyperparameters,
                "cost function": self.cost_function}
        else:
            raise Exception("multi_task not defined")

        ############################################################################

    def evaluate_acquisition_function(self, x, x_out=None, acquisition_function="variance", origin=None, args=None):
        """
        Function to evaluate the acquisition function.

        Parameters
        ----------
        x : np.ndarray | list
            Point positions at which the acquisition function is evaluated. np.ndarray of shape (N x D) or list.
        x_out : np.ndarray, optional
            Point positions in the output space.
        acquisition_function : Callable, optional
            Acquisition function to execute. Callable with inputs (x,gpcam.gp_optimizer.GPOptimizer),
            where x is a V x D array of input x_data. The return value is a 1d array of length V.
            The default is `variance`.
        origin : np.ndarray, optional
            If a cost function is provided this 1d numpy array of length D is used as the origin of motion.
        args : any, optional
            Arguments that will be passed to the acquisition function as part of the gp_optimizer object.
            CAUTION: this will overwrite the args set at initialization.

        Return
        ------
        The acquisition function evaluations at all points x : np.ndarray
        """
        assert self.gp, "GP not yet initialized; tell() data!"
        if x_out is None: x_out = self.x_out
        if args is not None: self._args = args

        if self.cost_function and origin is None:
            warnings.warn("Warning: For the cost function to be active, an origin has to be provided.")
        x = np.array(x)
        try:
            res = sm.evaluate_acquisition_function(
                x, gpo=self, acquisition_function=acquisition_function, origin=origin, dim=self.input_space_dimension,
                cost_function=self.cost_function, x_out=x_out)
            return -res
        except Exception as ex:
            logger.error(ex)
            logger.error("Evaluating the acquisition function was not successful.")
            raise Exception("Evaluating the acquisition function was not successful.", ex)

        ############################################################################

    def tell(self, x, y, noise_variances=None, append=True, gp_rank_n_update=None):
        """
        This function can tell() the gp_optimizer class
        the data that was collected. The data will instantly be used to update the GP data.

        Parameters
        ----------
        x : np.ndarray | list
            Point positions to be communicated to the Gaussian Process; either a np.ndarray of shape (U x D)
            or a list.
        y : np.ndarray
            The values of the data points. Shape (V,No).
            It is possible that not every entry in `x_new`
            has all corresponding tasks available. In that case `y_new` may contain np.nan entries.
        noise_variances : np.ndarray, optional
            An numpy array or list defining the uncertainties/noise in the
            `y_data` in form of a point-wise variance. Shape (V, No).
            If `y_data` has np.nan entries, the corresponding
            `noise_variances` have to be np.nan.
            Note: if no noise_variances are provided here, the noise_function
            callable will be used; if the callable is not provided, the noise variances
            will be set to `abs(np.mean(y_data)) / 100.0`. If
            noise covariances are required (correlated noise), make use of the noise_function.
            Only provide a noise function OR `noise_variances`, not both.
        append : bool, optional
            Indication whether to append to or overwrite the existing dataset. Default = True.
            In the default case, data will be appended.
        gp_rank_n_update : bool , optional
            Indicates whether the GP marginal should be rank-n updated or recomputed. The default
            is `gp_rank_n_update=append`, meaning if data is only appended, the rank_n_update will
            be performed.
        """

        if gp_rank_n_update is None: gp_rank_n_update = append
        if self.gp:
            self.update_gp_data(x, y, noise_variances_new=noise_variances,
                                append=append, gp_rank_n_update=gp_rank_n_update)
        else:
            self._initializeGP(x, y, noise_variances=noise_variances)

    def ask(self,
            input_set,
            x_out=None,
            acquisition_function='variance',
            position=None,
            n=1,
            method="global",
            pop_size=20,
            max_iter=20,
            tol=1e-6,
            constraints=(),
            x0=None,
            vectorized=True,
            info=False,
            args=None,
            dask_client=None,
            batch_size=None):

        """
        Given that the acquisition device is at `position`, this function `ask()`s for
        `n` new optimal points within a given `input_set` (given as bounds or candidates)
        using the optimization setup `method`,
        `acquisition_function_pop_size`, `max_iter`, `tol`, `constraints`, and `x0`.
        This function can also choose the best candidate of a candidate set for Bayesian optimization
        on non-Euclidean input spaces.

        Parameters
        ----------
        input_set : np.ndarray | list
            Either a numpy array of floats of shape D x 2 describing the Euclidean
            search space or a set of candidates in the form of a list. If a candidate list
            is provided, `ask()` will evaluate the acquisition function on each
            element and return a sorted array of length `n`.
            This is usually desirable for non-Euclidean inputs but can be used either way. If candidates are
            Euclidean, they should be provided as a list of 1d np.ndarrays. In that case `vectorized = True` will
            lead to a vectorized acquisition function evaluation.
            The possibility of a candidate list together with user-defined acquisition functions also means
            that mixed discrete-continuous spaces can be considered here. The candidates will be directly
            given to the acquisition function.
        x_out : np.ndarray, optional
            The position indicating where in the output space the acquisition function should be evaluated.
            This array is of shape (No). This is only use the multi-task setting.
        position : np.ndarray, optional
            Current position in the input space. If a cost function is
            provided this position will be taken into account
            to guarantee a cost-efficient new suggestion. The default is None.
        n : int, optional
            The algorithm will try to return n suggestions for
            new measurements. This is either done by method = 'hgdl', or otherwise
            by maximizing the collective information gain (default).
        acquisition_function : Callable | str, optional
            The acquisition function accepts as input a numpy array
            of size V x D (such that V is the number of input
            points, and D is the parameter space dimensionality) and
            a :py:class:`GPOptimizer` object. The return value is 1d array
            of length V providing 'scores' for each position,
            such that the highest scored point will be measured next.
            In the single-task case (using :py:meth:`gpcam.GPOptimizer)
            the following built-in acquisition functions can be used:
            `ucb`,`lcb`,`maximum`,
            `minimum`, `variance`,`expected improvement`,
            `relative information entropy`,`relative information entropy set`,
            `probability of improvement`, `gradient`,`total correlation`,`target probability`.
            In the multi-task case (using :py:meth:`gpcam.fvGPOptimizer)
            the following built-in acquisition functions can be used:
            `variance`, `relative information entropy`,
            `relative information entropy set`, `total correlation`, `ucb`, `lcb`,
            and `expected improvement`.
            In the multi-task case, it is highly recommended to
            deploy a user-defined acquisition function due to the intricate relationship
            of posterior distributions at different points in the output space.
            If None, the default function `variance`, meaning
            :py:meth:`fvgp.GP.posterior_covariance` with variance_only = True will be used.
            The acquisition function can be a callable function of the form my_func(x,gpcam.GPOptimizer)
            which will be maximized (!!!), so make sure desirable new measurement points
            will be located at maxima.
            Explanations of the built-in acquisition functions:
            variance: simply the posterior variance;
            relative information entropy: the KL divergence of the prior over predictions and the posterior;
            relative information entropy set: the KL divergence of the prior;
            defined over predictions and the posterior point-by-point;
            ucb: upper confidence bound, posterior mean + 3. std;
            lcb: lower confidence bound, -(posterior mean - 3. std);
            maximum: finds the maximum of the current posterior mean;
            minimum: finds the maximum of the current posterior mean;
            gradient: puts focus on high-gradient regions;
            probability of improvement: as the name would suggest;
            expected improvement: as the name would suggest;
            total correlation: extension of mutual information to more than 2 random variables;
            target probability: probability of a target. This needs a dictionary
            args = {'a': lower bound, 'b': upper bound} to be defined.
        method : str, optional
            A string defining the method used to find the maximum
            of the acquisition function. Choose from `global`,
            `local`, `hgdl`. HGDL is an in-house hybrid optimizer
            that is comfortable on HPC hardware.
            The default is `global`.
        pop_size : int, optional
            An integer defining the number of individuals if `global`
            is chosen as method. The default is 20. For
            :py:mod:`hgdl` this will be overwritten
            by the `dask_client` definition.
        max_iter : int, optional
            This number defined the number of iterations
            before the optimizer is terminated. The default is 20.
        tol : float, optional
            Termination criterion for the local optimizer.
            The default is 1e-6.
        x0 : np.ndarray, optional
            A set of points as numpy array of shape N x D,
            used as starting location(s) for the optimization
            algorithms. The default is None.
        vectorized : bool, optional
            If your acquisition function is vectorized to return the
            solution to an array of inquiries as an array,
            this option makes the optimization faster if method = 'global'
            is used. The default is True but will be set to
            False if method is not global.
        info : bool, optional
            Print optimization information. The default is False.
        constraints : tuple of object instances, optional
            scipy constraints instances, depending on the used optimizer.
        args : any, optional
            Arguments that will be passed to the acquisition function as part of the gp_optimizer object.
            This will overwrite the args set at initialization.
        dask_client : distributed.client.Client, optional
            A Dask Distributed Client instance for distributed
            `acquisition_function` optimization. If None is provided,
            a new :py:class:`distributed.client.Client` instance is constructed for hgdl.
        batch_size : distributed.client.Client, optional
            If a candidate set (input set) and a dask client is provided, the acquisition function evaluations
            will be executed in parallel in batches of this size.

        Return
        ------
        Solution : {'x': np.array(maxima), "f_a(x)" : np.array(func_evals), "opt_obj" : opt_obj}
            Found maxima of the acquisition function, the associated function values and optimization object
            that, only in case of `method` = `hgdl` can be queried for solutions.
        """

        logger.debug("ask() initiated with hyperparameters: {}", self.hyperparameters)
        logger.debug("optimization method: {}", method)
        logger.debug("input_set:\n{}", input_set)
        logger.debug("acq func: {}", acquisition_function)

        assert self.gp, "GP not yet initialized; tell() data!"
        if args is not None: self._args = args
        if x_out is None: x_out = self.x_out
        assert isinstance(vectorized, bool)

        if isinstance(input_set, np.ndarray) and np.ndim(input_set) != 2:
            raise Exception("The input_set parameter has to be a 2d np.ndarray or a list.")

        #for user-defined acquisition functions, use "hgdl" if n>1 and no candidates
        dask_client_provided = False
        if isinstance(input_set, np.ndarray) and n > 1 and callable(acquisition_function):
            warnings.warn("Method set to hgdl for callable acq. func and n>1.")
            method = "hgdl"
            if dask_client is None:
                warnings.warn("Initiating dask client for `hgdl`")
                dask_client = Client()
                dask_client_provided = True

        #if method is hgdl but no client is provided, m ake one
        if dask_client is None and method == "hgdl":
            warnings.warn("Initiating dask client for `hgdl`")
            dask_client = Client()
            dask_client_provided = True

        #if n>1 and Euclidean search and method!='hgdl', then global optimization of total corr or similar
        if isinstance(input_set, np.ndarray) and n > 1 and method != "hgdl" and not callable(acquisition_function):
            vectorized = False
            method = "global"
            new_optimization_bounds = np.vstack([input_set for i in range(n)])
            input_set = new_optimization_bounds
            if acquisition_function != "total correlation" and acquisition_function != "relative information entropy":
                acquisition_function = "total correlation"
                warnings.warn("You specified n>1 and method != 'hgdl' in ask(). The acquisition function "
                              "has therefore been changed to 'total correlation'.")

        if acquisition_function == "total correlation" or acquisition_function == "relative information entropy":
            warnings.warn("I set vectorized=False for total corr. or rel. inf. entropy.")
            vectorized = False

        maxima, func_evals, opt_obj = sm.find_acquisition_function_maxima(
            self,
            acquisition_function,
            origin=position,
            number_of_maxima_sought=n,
            input_set=input_set,
            input_set_dim=self.input_space_dimension,
            optimization_method=method,
            optimization_pop_size=pop_size,
            optimization_max_iter=max_iter,
            optimization_tol=tol,
            cost_function=self.cost_function,
            optimization_x0=x0,
            constraints=constraints,
            vectorized=vectorized,
            x_out=x_out,
            info=info,
            dask_client=dask_client,
            batch_size=batch_size)
        if n > 1: return {'x': maxima.reshape(-1, self.input_space_dimension), "f_a(x)": func_evals,
                          "opt_obj": opt_obj}
        if dask_client_provided: dask_client.close()
        return {'x': np.array(maxima), "f_a(x)": np.array(func_evals), "opt_obj": opt_obj}

    def optimize(self,
                 *,
                 func,
                 search_space,
                 x_out=None,
                 hyperparameter_bounds=None,
                 train_at=(10, 20, 50, 100, 200),
                 x0=None,
                 acq_func='lcb',
                 max_iter=100,
                 callback=None,
                 break_condition=None,
                 ask_max_iter=20,
                 ask_pop_size=20,
                 method="global",
                 training_method="global",
                 training_max_iter=20
                 ):
        """
        This function is a light-weight optimization loop, using `tell()` and `ask()` repeatedly
        to optimize a given function, while retraining the GP regularly. For advanced customizations please
        use those three methods in a customized loop.

        Parameters
        ----------
        func : Callable
            The function to be optimized. The callable should be of the form def f(x),
            where `x` is an element of your search space. The return is a tuple of scalars or vectors (a,b) where
            `a` is a scalar/vector of function evaluations and `b` is a scalar/vector of noise variances.
            Scalar here applies when the function to be optimized is a scalar valued function.
            Vector here applies when the function to be optimized is a vector valued function.
        search_space : np.ndarray | list
            In the Euclidean case this should be a 2d np.ndarray of bounds in each direction of the input space.
            In the non-Euclidean case, this should be a list of all candidates.
        x_out : np.ndarray, optional
            The position indicating where in the output space the acquisition function should be evaluated.
            This array is of shape (No).
        hyperparameter_bounds : np.ndarray
            Bound of the hyperparameters for the training. The default will only work for the default kernel.
            Otherwise, please specify bounds for your hyperparameters.
        train_at : tuple, optional
            The list should contain the integers that indicate the data lengths
            at which to train the GP. The default = [10,20,50,100,200].
        x0 : np.ndarray, optional
            Starting position(s). Corresponding to the search space either elements of
            the candidate set in form of a list or elements of the Euclidean search space in the form of a
            2d np.ndarray.
        acq_func : Callable, optional
            Default lower-confidence bound(lcb) which means minimizing the `func`.
            The acquisition function should be formulated such that MAXIMIZING it will lead to the
            desired optimization (minimization or maximization) of `func`.
            For example `lcb` (the default) MAXIMIZES -(mean - 3.0 * standard dev) which is equivalent to minimizing
            (mean - 3.0 * standard dev) which leads to finding a minimum.
        max_iter : int, optional
            The maximum number of iterations. Default=10,000,000.
        callback : Callable, optional
            Function to be called in every iteration. Form: f(x_data, y_data)
        break_condition : Callable, optional
            Callable f(x_data, y_data) that should return `True` if run is complete, otherwise `False`.
        ask_max_iter : int, optional
            Default=20. Maximum number of iteration of the global and hybrid optimizer within `ask()`.
        ask_pop_size : int, optional
            Default=20. Population size of the global and hybrid optimizer.
        method : str, optional
            Default=`global`. Method of optimization of the acquisition function.
            One of `global, `local`, `hybrid`.
        training_method : str, optional
            Default=`global`. See :py:meth:`gpcam.GPOptimizer.train`
        training_max_iter : int, optional
            Default=20. See :py:meth:`gpcam.GPOptimizer.train`


        Return
        ------
        Full traces of function values `f(x)` and arguments `x` and the last entry: dict
            Form {'trace f(x)': self.y_data,
                  'trace x': self.x_data,
                  'f(x)': self.y_data[-1],
                  'x': self.x_data[-1]}
        """
        assert callable(func)
        assert isinstance(search_space, np.ndarray) or isinstance(search_space, list)
        assert isinstance(max_iter, int)

        if not x0:
            if isinstance(search_space, list): x0 = random.sample(search_space, 10)
            if isinstance(search_space, np.ndarray): x0 = np.random.uniform(low=search_space[:, 0],
                                                                            high=search_space[:, 1],
                                                                            size=(10, len(search_space)))
        result = list(map(func, x0))
        if self.multi_task:
            len_x_out = len(result[0][0])
            y = np.asarray(list(map(np.hstack, zip(*result)))).reshape(-1, len_x_out)[0:len(result)]
            v = np.asarray(list(map(np.hstack, zip(*result)))).reshape(-1, len_x_out)[len(result):]
        else:
            y, v = map(np.hstack, zip(*result))
        self.tell(x=x0, y=y, noise_variances=v, append=False)
        if x_out is None: x_out = self.x_out
        self.train(hyperparameter_bounds=hyperparameter_bounds)
        for i in range(max_iter):
            logger.debug("iteration {}", i)
            x_new = self.ask(search_space,
                             x_out,
                             acquisition_function=acq_func,
                             method=method,
                             pop_size=ask_pop_size,
                             max_iter=ask_max_iter,
                             tol=1e-6,
                             constraints=(),
                             info=False)["x"]
            y_new, v_new = func(x_new)
            if callable(callback): callback(self.x_data, self.y_data)
            self.tell(x=x_new, y=y_new, noise_variances=v_new, append=True)
            if len(self.x_data) in train_at: self.train(hyperparameter_bounds=hyperparameter_bounds,
                                                        method=training_method, max_iter=training_max_iter)
            if callable(break_condition):
                if break_condition(self.x_data, self.y_data): break
        return {'trace f(x)': self.y_data,
                'trace x': self.x_data,
                'f(x)': self.y_data[-1],
                'x': self.x_data[-1]}

    def __getstate__(self):  # Called when the object is pickled
        state = dict()
        state.update(dict(
            cost_function=self.cost_function,
            init_hyperparameters=self.init_hyperparameters,
            compute_device=self.compute_device,
            kernel_function=self.kernel_function,
            kernel_function_grad=self.kernel_function_grad,
            noise_function=self.noise_function,
            noise_function_grad=self.noise_function_grad,
            prior_mean_function=self.prior_mean_function,
            prior_mean_function_grad=self.prior_mean_function_grad,
            _gp2Scale=self._gp2Scale,
            gp2Scale_dask_client=None,
            gp2Scale_batch_size=self.gp2Scale_batch_size,
            gp2Scale_linalg_mode=self.gp2Scale_linalg_mode,
            calc_inv=self.calc_inv,
            ram_economy=self.ram_economy,
            _args=self._args,
            logging=self.logging,
            multi_task=self.multi_task,
            x_out=self.x_out,
            gp=self.gp, #True or False (whether initialized, not the object)
            ))
        if self.gp: state.update(super().__getstate__())
        return state

    def __setstate__(self, state):  # Called when the object is unpickled
        state['gp2Scale_dask_client'] = None
        self.__dict__.update(state)


