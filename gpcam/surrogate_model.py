#!/usr/bin/env python
import itertools
from functools import partial
import math
import numpy as np
from loguru import logger

from hgdl.hgdl import HGDL
from scipy.optimize import differential_evolution as devo, minimize
from scipy.stats import norm


##########################################################################
def find_acquisition_function_maxima(gp, acquisition_function,
                                     origin, number_of_maxima_sought,
                                     optimization_bounds,
                                     acquisiiton_function_grad = None,
                                     optimization_method="global",
                                     optimization_pop_size=20,
                                     optimization_max_iter=10,
                                     optimization_tol=1e-6,
                                     optimization_x0=None,
                                     constraints = (),
                                     cost_function=None,
                                     cost_function_parameters=None,
                                     vectorized = True,
                                     args = {},
                                     dask_client=None,
                                     info = False):
    bounds = np.array(optimization_bounds)
    opt_obj = None
    logger.info("====================================")
    logger.info(f"Finding acquisition function maxima via {optimization_method} method")
    logger.info("tolerance: {}", optimization_tol)
    logger.info("population size: {}", optimization_pop_size)
    logger.info("maximum number of iterations: {}", optimization_max_iter)
    logger.info("bounds: {}")
    logger.info(bounds)
    logger.info("cost function parameters: {}", cost_function_parameters)
    logger.info("====================================")

    if optimization_method == "global":
        opti, func_eval = differential_evolution(
            evaluate_acquisition_function,
            optimization_bounds,
            tol=optimization_tol,
            x0 = optimization_x0,
            popsize=optimization_pop_size,
            max_iter=optimization_max_iter,
            origin=origin,
            constraints = constraints,
            gp=gp,
            number_of_maxima_sought = number_of_maxima_sought,
            acquisition_function=acquisition_function,
            cost_function=cost_function,
            cost_function_parameters=cost_function_parameters,
            vectorized = vectorized,
            args = args,
            disp = info
        )
        opti = np.asarray(opti)
        func_eval = np.asarray(func_eval)

    elif optimization_method == "hgdl":
        ###run differential evo first if hxdy only returns stationary points
        ###then of hgdl is successful, stack results and return
        if constraints: logger.warning("The HGDL won't adhere to constraints for the acquisition function. Use method 'local' or 'global'")
        opt_obj = HGDL(evaluate_acquisition_function,
                       evaluate_acquisition_function_gradient,
                       bounds,
                       num_epochs=optimization_max_iter,
                       local_optimizer="L-BFGS-B",
                       constraints = constraints,
                       args=(gp, acquisition_function, origin, number_of_maxima_sought, cost_function, cost_function_parameters, args))

        ###optimization_max_iter, tolerance here
        if optimization_x0: optimization_x0 = optimization_x0.reshape(1,-1)
        opt_obj.optimize(dask_client=dask_client, x0=optimization_x0, tolerance=optimization_tol)
        res = opt_obj.get_final()
        opt_obj.cancel_tasks()
        res = res[0:min(len(res),number_of_maxima_sought)]
        opti = np.asarray([entry["x"] for entry in res])
        func_eval = np.asarray([entry["f(x)"] for entry in res])

    elif optimization_method == "hgdlAsync":
        ###run differential evo first if hxdy only returns stationary points
        ###then of hgdl is successful, stack results and return
        if constraints: logger.warning("The HGDL won't adhere to constraints for the acquisition function. Use method 'local' or 'global'")
        opt_obj = HGDL(evaluate_acquisition_function,
                       evaluate_acquisition_function_gradient,
                       bounds,
                       num_epochs=optimization_max_iter,
                       local_optimizer="L-BFGS-B",
                       constraints = constraints,
                       args=(gp, acquisition_function, origin, number_of_maxima_sought, cost_function, cost_function_parameters, args))

        ###optimization_max_iter, tolerance here
        if optimization_x0: optimization_x0 = optimization_x0.reshape(1,-1)
        opt_obj.optimize(dask_client=dask_client, x0=optimization_x0, tolerance=optimization_tol)
        opti = None
        func_eval = None
        return opti, func_eval, opt_obj

    elif optimization_method == "local":
        if optimization_x0 is not None and np.ndim(optimization_x0) == 1:
            x0 = optimization_x0
        elif optimization_x0 is not None and np.ndim(optimization_x0) == 2:
            x0 = optimization_x0[0]
        else:
            x0 = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=len(bounds))
        a = minimize(
            evaluate_acquisition_function,
            x0,
            args=(gp, acquisition_function, origin, number_of_maxima_sought,cost_function, cost_function_parameters, args),
            method="L-BFGS-B",
            jac=evaluate_acquisition_function_gradient,
            bounds=bounds,
            constraints = constraints,
            tol=optimization_tol,
            callback=None,
            options={"maxiter": optimization_max_iter}
        )
        opti = np.array([a["x"]])
        func_eval = np.array(a["fun"])
        if np.ndim(func_eval) == 0: func_eval = np.array([func_eval])
        if a["success"] is False:
            logger.warning("local acquisition function optimization not successful, solution replaced with random point.")
            opti = np.array(x0)
            if opti.ndim != 2: opti = np.array([opti])
            func_eval = evaluate_acquisition_function(x0,
                                                      gp, acquisition_function, origin,
                                                      cost_function, cost_function_parameters)
            if np.ndim(func_eval) != 1: func_eval = np.array([func_eval])

    else: raise ValueError("Invalid acquisition function optimization method given.")
    if np.ndim(func_eval) != 1 or np.ndim(opti) != 2:
        logger.error("f(x): ", func_eval)
        logger.error("x: ", opti)
        raise Exception(
            "The output of the acquisition function optimization dim (f) != 1 or dim(x) != 2. Please check your "
            "acquisition function. It should return a 1-d numpy array")
    return opti, func_eval, opt_obj

############################################################
############################################################
############################################################
############################################################
def evaluate_acquisition_function(x, gp, acquisition_function, origin=None, number_of_maxima_sought = None,
                                  cost_function=None, cost_function_parameters=None, args = None):
    ##########################################################
    ####this function evaluates a default or a user-defined acquisition function
    ##########################################################
    if np.ndim(x) == 1: x = np.array([x])
    if cost_function is not None and origin is not None:
        cost_eval = cost_function(origin, x, cost_function_parameters)
    else:
        cost_eval = 1.0
    # for user defined acquisition function
    if callable(acquisition_function): return -acquisition_function(x, gp) / cost_eval
    else:
        obj_eval = evaluate_gp_acquisition_function(x, acquisition_function, gp, number_of_maxima_sought, args = args)
        obj_eval = -obj_eval / cost_eval
        return obj_eval


def evaluate_acquisition_function_gradient(x, gp, acquisition_function, origin=None, number_of_maxima_sought = None,
                                           cost_function=None, cost_function_parameters=None, acq_args = None):
    acquisition_gradient = gradient(evaluate_acquisition_function,
                                    x,
                                    1e-6,
                                    gp,
                                    acquisition_function,
                                    origin,
                                    number_of_maxima_sought,
                                    cost_function,
                                    cost_function_parameters,
                                    acq_args)
    return acquisition_gradient


def evaluate_gp_acquisition_function(x, acquisition_function, gp, number_of_maxima_sought, multi_task = False, args = {}):
    ##this function will always spit out a 1d numpy array
    ##for certain functions, this array will only have one entry
    ##for the other the length == len(x)
    if multi_task:
        try: x = x.reshape(-1,gp.orig_input_space_dim)
        except: raise Exception("x request in evaluate_gp_acquisition_function has wrong dimensionality.", x.shape)
    else:
        try: x = x.reshape(-1,gp.input_space_dim)
        except: raise Exception("x request in evaluate_gp_acquisition_function has wrong dimensionality.", x.shape)

    if acquisition_function == "variance":
        res = gp.posterior_covariance(x, variance_only=True)["v(x)"]
        return res
    elif acquisition_function == "covariance":
        res = gp.posterior_covariance(x)
        b = res["S"]
        sgn, logdet = np.linalg.slogdet(b)
        return np.array([np.sqrt(sgn * np.exp(logdet))])
    elif acquisition_function == "shannon_ig":
        res = gp.shannon_information_gain(x)["sig"]
        return np.array([res])
    elif acquisition_function == "shannon_ig_vec":
        res = gp.shannon_information_gain_vec(x)["sig(x)"]
        return res
    elif acquisition_function == "ucb":
        m = gp.posterior_mean(x)["f(x)"]
        v = gp.posterior_covariance(x, variance_only=True)["v(x)"]
        return m + 3.0 * np.sqrt(v)
    elif acquisition_function == "lcb":
        m = gp.posterior_mean(x)["f(x)"]
        v = gp.posterior_covariance(x, variance_only=True)["v(x)"]
        return -(m - 3.0 * np.sqrt(v))
    elif acquisition_function == "maximum":
        res = gp.posterior_mean(x)["f(x)"]
        return res
    elif acquisition_function == "gradient":
        mean_grad = gp.posterior_mean_grad(x)["df/dx"]
        std = np.sqrt(gp.posterior_covariance(x, variance_only = True)["v(x)"])
        res = np.linalg.norm(mean_grad, axis = 1) * std
        return res
    elif acquisition_function == "minimum":
        res = gp.posterior_mean(x)["f(x)"]
        return -res
    elif acquisition_function == "PI":
        m = gp.posterior_mean(x)["f(x)"]
        std = np.sqrt(gp.posterior_covariance(x, variance_only=True)["v(x)"])
        last_best = np.max(gp.y_data)
        return  norm.cdf((m - last_best)/(std+1e-9))
    elif acquisition_function == "total_correlation":
        return -np.array([gp.gp_total_correlation(x)["total correlation"]])
    elif acquisition_function == "expected_improvement":
        m = gp.posterior_mean(x)["f(x)"]
        std = np.sqrt(gp.posterior_covariance(x, variance_only=True)["v(x)"])
        last_best = np.max(gp.y_data)
        a = (m-last_best)
        a[a<0.] = 0.
        gamma = a / std
        pdf = norm.pdf(gamma)
        cdf = norm.cdf(gamma)
        return std * (gamma * cdf + pdf)
    elif acquisition_function == "target_probability":
        a = args["a"]
        b = args["b"]
        mean = gp.posterior_mean(x)["f(x)"]
        cov = gp.posterior_covariance(x)["v(x)"]
        result = np.zeros((len(x)))
        for i in range(len(x)):
            result[i] = 0.5 * (math.erf((b-mean[i])/np.sqrt(2.*cov[i])) -  math.erf((a-mean[i])/np.sqrt(2.*cov[i])))
        return result
    else: raise Exception("No valid acquisition function string provided.")

    raise ValueError(f'The requested acquisition function "{acquisition_function}" does not exist.')



def differential_evolution(ObjectiveFunction,
                           bounds,
                           tol,
                           popsize,
                           max_iter=100,
                           origin=None,
                           gp=None,
                           x0 = None,
                           acquisition_function=None,
                           cost_function=None,
                           constraints = (),
                           number_of_maxima_sought = None,
                           cost_function_parameters=None,
                           args = {},
                           disp = False,
                           vectorized = True):
    fun = partial(ObjectiveFunction, gp=gp, acquisition_function=acquisition_function, origin=origin, number_of_maxima_sought = number_of_maxima_sought,
                  cost_function=cost_function, cost_function_parameters=cost_function_parameters,  args = args)
    res = devo(partial(acq_function_vectorization_wrapper, func = fun, vectorized = vectorized), bounds, tol=tol,x0 = x0, maxiter=max_iter, popsize=popsize, polish=False, disp = disp, constraints = constraints, vectorized=vectorized)
    return [list(res["x"])], list([res["fun"]])

def acq_function_vectorization_wrapper(x, func = None,vectorized = False):
    if vectorized is True: acq = func(x.T)
    else: acq = func(x)
    return acq

def gradient(function, point, epsilon=1e-6, *args):
    """
    This function calculates the gradient of a function by using finite differences

    Extended description of function.

    Parameters:
    function (function object): the function the gradient should be computed of
    point (numpy array 1d): point at which the gradient should be computed

    optional:
    epsilon (float): the distance used for the evaluation of the function

    Returns:
    numpy array of gradient

    """
    gradient = np.zeros((len(point)))
    # args = args[0]
    for i in range(len(point)):
        new_point = np.array(point)
        new_point[i] += epsilon
        gradient[i] = (function(new_point, *args) - function(point, *args)) / epsilon
    return gradient

def bhattacharyya_distance(reference_distribution, test_distribution, dx):
    y1 = reference_distribution / (sum(reference_distribution) * dx)
    y2 = test_distribution / (sum(test_distribution) * dx)
    return sum(np.sqrt(y1 * y2)) * dx
