#!/usr/bin/env python
import itertools
from functools import partial

import numpy as np
from loguru import logger

from hgdl.hgdl import HGDL
from scipy.optimize import differential_evolution as devo, minimize


def evaluate_acquisition_function(x, gp, acquisition_function, origin=None,
                                  cost_function=None, cost_function_parameters=None):
    ##########################################################
    ####this function evaluates a default or a user-defined acquisition function
    ##########################################################
    if x.ndim == 1: x = np.array([x])
    if cost_function is not None and origin is not None and cost_function_parameters is not None:
        cost_eval = cost_function(origin, x, cost_function_parameters)
    else:
        cost_eval = 1.0
    # for user defined acquisition function
    if callable(acquisition_function):
        return -acquisition_function(x, gp) / cost_eval
    obj_eval = evaluate_gp_acquisition_function(x, acquisition_function, gp)
    # if no user defined acquisition function is used
    obj_eval = obj_eval / cost_eval
    return -obj_eval


def evaluate_acquisition_function_gradient(x, gp, acquisition_function, origin=None,
                                           cost_function=None, cost_function_parameters=None):
    acquisition_gradient = gradient(evaluate_acquisition_function,
                                    x,
                                    1e-5,
                                    gp,
                                    acquisition_function,
                                    origin,
                                    cost_function,
                                    cost_function_parameters)
    return acquisition_gradient


def evaluate_acquisition_function_hessian(x, gp, acquisition_function, origin=None,
                                          cost_function=None, cost_function_parameters=None):
    acquisition_hessian = hessian(evaluate_acquisition_function,
                                  x,
                                  1e-5,
                                  gp,
                                  acquisition_function,
                                  origin,
                                  cost_function,
                                  cost_function_parameters)
    return acquisition_hessian


def evaluate_gp_acquisition_function(x, acquisition_function, gp):
    ##this function will always spit out a 1d numpy array
    ##for certain functions, this array will only have one entry
    ##for the other the length == len(x)
    if len(x.shape) == 1: x = np.array([x])
    if acquisition_function == "variance":
        res = gp.posterior_covariance(x, variance_only=True)["v(x)"]
        return res
    if acquisition_function == "covariance":
        res = gp.posterior_covariance(x)
        b = res["S(x)"]
        sgn, logdet = np.linalg.slogdet(b)
        return np.array([np.sqrt(sgn * np.exp(logdet))])
    ###################more here: shannon_ig  for instance
    elif acquisition_function == "shannon_ig":
        res = gp.shannon_information_gain(x)["sig"]
        return np.array([res])
    elif acquisition_function == "ucb":
        m = gp.posterior_mean(x)["f(x)"]
        v = gp.posterior_covariance(x, variance_only=True)["v(x)"]
        return m + 3.0 * np.sqrt(v)
    elif acquisition_function == "maximum":
        res = gp.posterior_mean(x)["f(x)"]
        return res
    elif acquisition_function == "minimum":
        res = gp.posterior_mean(x)["f(x)"]
        return -res


##########################################################################
def find_acquisition_function_maxima(gp, acquisition_function,
                                     origin, number_of_maxima_sought,
                                     optimization_bounds,
                                     optimization_method="global",
                                     optimization_pop_size=20,
                                     optimization_max_iter=10,
                                     optimization_tol=1e-6,
                                     optimization_x0=None,
                                     constraints = (),
                                     cost_function=None,
                                     cost_function_parameters=None,
                                     dask_client=None):
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
            popsize=optimization_pop_size,
            max_iter=optimization_max_iter,
            origin=origin,
            constraints = constraints,
            gp=gp,
            acquisition_function=acquisition_function,
            cost_function=cost_function,
            cost_function_parameters=cost_function_parameters
        )
        opti = np.asarray(opti)
        func_eval = np.asarray(func_eval)
    elif optimization_method == "hgdl":
        ###run differential evo first if hxdy only returns stationary points
        ###then of hgdl is successful, stack results and return
        a = HGDL(evaluate_acquisition_function,
                 evaluate_acquisition_function_gradient,
                 bounds,
                 evaluate_acquisition_function_hessian,
                 num_epochs=optimization_max_iter,
                 local_optimizer="L-BFGS-B",
                 constraints = constraints,
                 args=(gp, acquisition_function, origin, cost_function, cost_function_parameters))

        #####optimization_max_iter, tolerance here
        if optimization_x0: optimization_x0 = optimization_x0.reshape(1,-1)
        a.optimize(dask_client=dask_client, x0=optimization_x0, tolerance=optimization_tol)
        res = a.get_final()
        a.cancel_tasks()
        opt_obj = a
        opti = res['x'][0:min(len(res['x']),number_of_maxima_sought)]
        func_eval = res['f(x)'][0:min(len(res['x']),number_of_maxima_sought)]

    elif optimization_method == "local":
        if optimization_x0 is not None and optimization_x0.ndim == 1:
            x0 = optimization_x0
        elif optimization_x0 is not None and optimization_x0.ndim == 2:
            x0 = optimization_x0[0]
        else:
            x0 = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=len(bounds))
        a = minimize(
            evaluate_acquisition_function,
            x0,
            args=(gp, acquisition_function, origin, cost_function, cost_function_parameters),
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
        if func_eval.ndim == 0: func_eval = np.array([func_eval])
        if a["success"] is False:
            logger.warning("local acquisition function optimization not successful, solution replaced with random point.")
            opti = np.array(x0)
            if opti.ndim != 2: opti = np.array([opti])
            func_eval = evaluate_acquisition_function(x0,
                                                      gp, acquisition_function, origin,
                                                      cost_function, cost_function_parameters)
            if func_eval.ndim != 1: func_eval = np.array([func_eval])
    else:
        raise ValueError("Invalid acquisition function optimization method given.")
    if func_eval.ndim != 1 or opti.ndim != 2:
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
def differential_evolution(ObjectiveFunction,
                           bounds,
                           tol,
                           popsize,
                           max_iter=100,
                           origin=None,
                           gp=None,
                           acquisition_function=None,
                           cost_function=None,
                           constraints = (),
                           cost_function_parameters=None):
    fun = partial(ObjectiveFunction, gp=gp, acquisition_function=acquisition_function, origin=origin,
                  cost_function=cost_function, cost_function_parameters=cost_function_parameters)
    res = devo(fun, bounds, tol=tol, maxiter=max_iter, popsize=popsize, polish=False, constraints = constraints)
    return [list(res["x"])], list([res["fun"]])


def normed_gaussian_function(x, mean, sigma2):
    return (1.0 / np.sqrt(2.0 * np.pi * sigma2)) * np.exp(
        -((x - mean) ** 2) / (2.0 * sigma2)
    )


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


def hessian(function, point, epsilon=1e-3, *args):
    """
    This function calculates the hessian of a function by using finite differences

    Extended description of function.

    Parameters:
    function (function object): the function, the hessian should be computed of
    point (numpy array 1d): point at which the gradient should be computed

    optional:
    epsilon (float): the distance used for the evaluation of the function

    Returns:
    numpy array of hessian

    """
    hessian = np.zeros((len(point), len(point)))
    for i in range(len(point)):
        for j in range(len(point)):
            new_point1 = np.array(point)
            new_point2 = np.array(point)
            new_point3 = np.array(point)
            new_point4 = np.array(point)

            new_point1[i] = new_point1[i] + epsilon
            new_point1[j] = new_point1[j] + epsilon

            new_point2[i] = new_point2[i] + epsilon
            new_point2[j] = new_point2[j] - epsilon

            new_point3[i] = new_point3[i] - epsilon
            new_point3[j] = new_point3[j] + epsilon

            new_point4[i] = new_point4[i] - epsilon
            new_point4[j] = new_point4[j] - epsilon

            hessian[i, j] = \
                (function(new_point1, *args) - function(new_point2, *args) - function(new_point3,
                                                                                      *args) + function(new_point4,
                                                                                                        *args)) \
                / (4.0 * (epsilon ** 2))
    return hessian


def bhattacharyya_distance(reference_distribution, test_distribution, dx):
    y1 = reference_distribution / (sum(reference_distribution) * dx)
    y2 = test_distribution / (sum(test_distribution) * dx)
    return sum(np.sqrt(y1 * y2)) * dx


def cast_to_index_set(x_input, x_output, mode='cartesian product'):
    n_orig = len(x_input)
    tasks = len(x_output)
    if mode == 'cartesian product':
        new_points = np.zeros((len(x_input) * len(x_output), len(x_input[0]) + len(x_output[0])))
        counter = 0
        for element in itertools.product(x_input, x_output):
            new_points[counter] = np.concatenate([element[0], element[1]], axis=0)
            counter += 1  ###can't we append?
    elif mode == 'stack':
        new_points = np.column_stack([x_input, x_output])
    return new_points
