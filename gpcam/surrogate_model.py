#!/usr/bin/env python
import itertools
from functools import partial
import math
import numpy as np
from loguru import logger
import random
from hgdl.hgdl import HGDL
from scipy.optimize import differential_evolution as devo, minimize
from scipy.stats import norm
from functools import partial
import warnings


##########################################################################
def find_acquisition_function_maxima(gp, acquisition_function,
                                     origin=None,
                                     number_of_maxima_sought=1,
                                     optimization_bounds=None,
                                     optimization_method="global",
                                     optimization_pop_size=20,
                                     optimization_max_iter=10,
                                     optimization_tol=1e-6,
                                     optimization_x0=None,
                                     constraints=(),
                                     cost_function=None,
                                     cost_function_parameters=None,
                                     vectorized=True,
                                     candidates=None,
                                     x_out=None,
                                     dask_client=None,
                                     info=False):

    if candidates is None and optimization_bounds is None:
        raise Exception("optimization bounds or candidates have to be provided")
    bounds = optimization_bounds
    opt_obj = None

    func = partial(evaluate_acquisition_function, gp=gp, acquisition_function=acquisition_function,
                   origin=origin, number_of_maxima_sought=number_of_maxima_sought,
                   cost_function=cost_function, cost_function_parameters=cost_function_parameters, x_out=x_out)
    grad = partial(gradient, func=func)

    logger.info("====================================")
    logger.info(f"Finding acquisition function maxima via {optimization_method} method")
    logger.info("tolerance: {}", optimization_tol)
    logger.info("population size: {}", optimization_pop_size)
    logger.info("maximum number of iterations: {}", optimization_max_iter)
    logger.info("bounds: {}")
    logger.info(bounds)
    logger.info("cost function parameters: {}", cost_function_parameters)
    logger.info("====================================")
    if candidates is not None:
        if not callable(acquisition_function): warnings.warn("It is recommended to use a custom acquisition \
        function for solutions on candidate sets. Proceed with caution.")
        if isinstance(candidates, np.ndarray):
            random_indices = random.sample(range(0, len(candidates)),
                                           min(number_of_maxima_sought * 100, len(candidates)))
            candidates = candidates[random_indices]
            res = func(candidates)
            sort_indices = np.argsort(res)
            res = res[sort_indices]
            candidates = candidates[sort_indices]
            length = min(number_of_maxima_sought, len(res))
            opti, func_eval, opt_obj = candidates[0:length], res[0:length], None
        elif isinstance(candidates, list):
            choices1 = random.sample(candidates, k=min(number_of_maxima_sought * 100, len(candidates)))
            choices = [[choices1[i]] for i in range(len(choices1))]
            res = np.asarray(list(map(func, choices))).reshape(len(choices))
            sort_indices = np.argsort(res)
            res = res[sort_indices]
            sorted_choices = [choices[sort_index] for sort_index in sort_indices]
            choices = sorted_choices
            length = min(number_of_maxima_sought, len(choices))
            opti, func_eval, opt_obj = np.asarray(choices[0:length]), res[0:length], None
            print(opti)
        else:
            raise Exception("Candidates, if provided, have to be a list or a 2d np.ndarray.")

    elif optimization_method == "global":
        opti, func_eval = differential_evolution(
            func,
            optimization_bounds,
            tol=optimization_tol,
            x0=optimization_x0,
            popsize=optimization_pop_size,
            max_iter=optimization_max_iter,
            constraints=constraints,
            vectorized=vectorized,
            disp=info
        )
        opti = np.asarray(opti)
        func_eval = np.asarray(func_eval)

    elif optimization_method == "hgdl":
        opt_obj = HGDL(func,
                       grad,
                       bounds,
                       num_epochs=optimization_max_iter,
                       local_optimizer="L-BFGS-B",
                       constraints=constraints)

        ###optimization_max_iter, tolerance here
        if optimization_x0: optimization_x0 = optimization_x0.reshape(1, -1)
        opt_obj.optimize(dask_client=dask_client, x0=optimization_x0, tolerance=optimization_tol)
        res = opt_obj.get_final()
        opt_obj.cancel_tasks()
        res = res[0:min(len(res), number_of_maxima_sought)]
        opti = np.asarray([entry["x"] for entry in res])
        func_eval = np.asarray([entry["f(x)"] for entry in res])

    elif optimization_method == "hgdlAsync":
        ###run differential evo first if hxdy only returns stationary points
        ###then of hgdl is successful, stack results and return
        if constraints: logger.warning(
            "The HGDL won't adhere to constraints for the acquisition function. Use method 'local' or 'global'")
        opt_obj = HGDL(func,
                       grad,
                       bounds,
                       num_epochs=optimization_max_iter,
                       local_optimizer="L-BFGS-B",
                       constraints=constraints)

        ###optimization_max_iter, tolerance here
        if optimization_x0: optimization_x0 = optimization_x0.reshape(1, -1)
        opt_obj.optimize(dask_client=dask_client, x0=optimization_x0, tolerance=optimization_tol)
        opti = np.zeros((1, gp.input_dim))
        func_eval = np.zeros((1))

    elif optimization_method == "local":
        if optimization_x0 is not None and np.ndim(optimization_x0) == 1:
            x0 = optimization_x0
        elif optimization_x0 is not None and np.ndim(optimization_x0) == 2:
            x0 = optimization_x0[0]
        else:
            x0 = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=len(bounds))
        a = minimize(
            func,
            x0,
            method="L-BFGS-B",
            jac=grad,
            bounds=bounds,
            constraints=constraints,
            tol=optimization_tol,
            callback=None,
            options={"maxiter": optimization_max_iter}
        )
        opti = np.array([a["x"]])
        func_eval = np.array(a["fun"])
        if np.ndim(func_eval) == 0: func_eval = np.array([func_eval])
        if a["success"] is False:
            logger.warning(
                "local acquisition function optimization not successful, solution replaced with random point.")
            opti = np.array(x0)
            if opti.ndim != 2: opti = np.array([opti])
            func_eval = evaluate_acquisition_function(x0,
                                                      gp, acquisition_function, origin,
                                                      cost_function, cost_function_parameters)
            if np.ndim(func_eval) != 1: func_eval = np.array([func_eval])

    else:
        raise ValueError("Invalid acquisition function optimization method given.")
    if np.ndim(func_eval) != 1 or np.ndim(opti) != 2:
        logger.error("f(x): ", func_eval)
        logger.error("x: ", opti)
        raise Exception(
            "The output of the acquisition function optimization dim (f) != 1 or dim(x) != 2. Please check your "
            "acquisition function. It should return a 1-d numpy array")
    return opti, -func_eval, opt_obj


############################################################
############################################################
############################################################
############################################################
def evaluate_acquisition_function(x, gp=None, acquisition_function=None, origin=None, number_of_maxima_sought=1,
                                  cost_function=None, cost_function_parameters=None, x_out=None):
    ##########################################################
    ####this function evaluates a default or a user-defined acquisition function
    ##########################################################
    if isinstance(x, np.ndarray) and np.ndim(x) == 1: x = x.reshape(-1, gp.input_dim)
    if x_out is not None and np.ndim(x_out) != 2: raise Exception(
        "x_out in evaluate_acquisition_function has to be a 2d numpy array.")

    if cost_function is not None and origin is not None:
        cost_eval = cost_function(origin, x, cost_function_parameters)
    else:
        cost_eval = 1.0
    # for user defined acquisition function
    if callable(acquisition_function):
        return -acquisition_function(x, gp) / cost_eval
    else:
        obj_eval = evaluate_gp_acquisition_function(x, acquisition_function, gp, number_of_maxima_sought, x_out=x_out)
        obj_eval = -obj_eval / cost_eval
        return obj_eval


def evaluate_gp_acquisition_function(x, acquisition_function, gp, number_of_maxima_sought, x_out):
    ##this function will always spit out a 1d numpy array
    ##for certain functions, this array will only have one entry
    ##for the other the length == len(x)
    if isinstance(x, np.ndarray) and np.ndim(x) == 1: raise Exception(
        "1d array given in evaluate_gp_acquisition_function.")
    if x_out is None:
        if acquisition_function == "variance":
            res = gp.posterior_covariance(x, x_out=x_out, variance_only=True)["v(x)"]
            return res
        elif acquisition_function == "relative information entropy":
            res = -gp.gp_relative_information_entropy(x, x_out=x_out)["RIE"]
            return np.array([res])
        elif acquisition_function == "relative information entropy set":
            res = -gp.gp_relative_information_entropy_set(x, x_out=x_out)["RIE"]
            return res
        elif acquisition_function == "ucb":
            m = gp.posterior_mean(x, x_out=x_out)["f(x)"]
            v = gp.posterior_covariance(x, x_out=x_out, variance_only=True)["v(x)"]
            return m + 3.0 * np.sqrt(v)
        elif acquisition_function == "lcb":
            m = gp.posterior_mean(x, x_out=x_out)["f(x)"]
            v = gp.posterior_covariance(x, x_out=x_out, variance_only=True)["v(x)"]
            return -(m - 3.0 * np.sqrt(v))
        elif acquisition_function == "maximum":
            res = gp.posterior_mean(x, x_out=x_out)["f(x)"]
            return res
        elif acquisition_function == "gradient":
            mean_grad = gp.posterior_mean_grad(x, x_out=x_out)["df/dx"]
            std = np.sqrt(gp.posterior_covariance(x, x_out=x_out, variance_only=True)["v(x)"])
            res = np.linalg.norm(mean_grad, axis=1) * std
            return res
        elif acquisition_function == "minimum":
            res = gp.posterior_mean(x, x_out=x_out)["f(x)"]
            return -res
        elif acquisition_function == "probability of improvement":
            m = gp.posterior_mean(x, x_out=x_out)["f(x)"]
            std = np.sqrt(gp.posterior_covariance(x, x_out=x_out, variance_only=True)["v(x)"])
            last_best = np.max(gp.y_data)
            return norm.cdf((m - last_best) / (std + 1e-9))
        elif acquisition_function == "total correlation":
            return -np.array([gp.gp_total_correlation(x, x_out=x_out)["total correlation"]])
        elif acquisition_function == "expected improvement":
            m = gp.posterior_mean(x, x_out=x_out)["f(x)"]
            std = np.sqrt(gp.posterior_covariance(x, x_out=x_out, variance_only=True)["v(x)"])
            last_best = np.max(gp.y_data)
            a = (m - last_best)
            a[a < 0.] = 0.
            gamma = a / (std + 1e-9)
            pdf = norm.pdf(gamma)
            cdf = norm.cdf(gamma)
            return std * (gamma * cdf + pdf)
        elif acquisition_function == "target probability":
            try:
                a = gp.args["a"]
                b = gp.args["b"]
            except:
                raise Exception("Reading the arguments for acq func `target probability` failed.")
            mean = gp.posterior_mean(x, x_out=x_out)["f(x)"]
            cov = gp.posterior_covariance(x, x_out=x_out)["v(x)"] + 1e-9
            result = np.zeros((len(x)))
            for i in range(len(x)):
                result[i] = 0.5 * (math.erf((b - mean[i]) / np.sqrt(2. * cov[i]))) - math.erf(
                (a - mean[i]) / np.sqrt(2. * cov[i]))
            return result
        else:
            raise Exception("No valid acquisition function string provided.")
        raise ValueError(f'The requested acquisition function "{acquisition_function}" does not exist.')
    else:
        if acquisition_function == "variance":
            res = gp.posterior_covariance(x, x_out=x_out, variance_only=True)["v(x)"]
            return np.sum(res.reshape(len(x), len(x_out)), axis=1)
        elif acquisition_function == "relative information entropy":
            res = -gp.gp_relative_information_entropy(x, x_out=x_out)["RIE"]
            return np.array([res])
        elif acquisition_function == "relative information entropy set":
            res = -gp.gp_relative_information_entropy_set(x, x_out=x_out)["RIE"]
            return np.sum(res.reshape(len(x), len(x_out)), axis=1)
        elif acquisition_function == "total correlation":
            return -np.array([gp.gp_total_correlation(x, x_out=x_out)["total correlation"]])
        else:
            raise Exception("No valid acquisition function string provided.")
        raise ValueError(f'The requested acquisition function "{acquisition_function}" does not exist.')


def differential_evolution(func,
                           bounds,
                           tol,
                           popsize,
                           max_iter=100,
                           x0=None,
                           constraints=(),
                           disp=False,
                           vectorized=True):

    res = devo(partial(acq_function_vectorization_wrapper, func=func, vectorized=vectorized), bounds, tol=tol, x0=x0,
               maxiter=max_iter, popsize=popsize, polish=False, disp=disp, constraints=constraints,
               vectorized=vectorized)
    return [list(res["x"])], list([res["fun"]])


def acq_function_vectorization_wrapper(x, func=None, vectorized=False):
    if vectorized is True:
        acq = func(x.T)
    else:
        acq = func(x)
    return acq


def gradient(x, func=None):
    epsilon = 1e-6
    gradient = np.zeros((len(x)))
    for i in range(len(x)):
        new_point = np.array(x)
        new_point[i] += epsilon
        gradient[i] = (func(new_point) - func(x)) / epsilon
    return gradient
