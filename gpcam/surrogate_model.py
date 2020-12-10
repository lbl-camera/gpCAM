#!/usr/bin/env python
import random
import numpy as np
import itertools
import matplotlib.pyplot as plt
from  .optimization import differential_evolution
from scipy.optimize import minimize


def evaluate_objective_function(x, gp, objective_function,origin = None,
        cost_function = None, cost_function_parameters = None):
    ##########################################################
    ####this function evaluates a default or a user-defined objective function
    ##########################################################
    if x.ndim == 1:x = np.array([x])
    if cost_function is not None and origin is not None and cost_function_parameters is not None:
        cost_eval = cost_function(origin,x,cost_function_parameters)
    else:
        cost_eval = np.array([1.0])
    #for user defined objective function
    if callable(objective_function):
        return -objective_function(x,gp)/cost_eval
    obj_eval = evaluate_gp_objective_function(x, objective_function, gp)
    #if no user defined objective function is used
    obj_eval = obj_eval / cost_eval
    return -obj_eval

def evaluate_objective_function_gradient(x, gp, objective_function, origin = None,
        cost_function = None, cost_function_parameters = None):
    objective_gradient = gradient(evaluate_objective_function, x, 1e-5, gp,objective_function,origin,cost_function,cost_function_parameters)
    return objective_gradient

def evaluate_objective_function_hessian(x, gp, objective_function,origin = None,
        cost_function = None, cost_function_parameters = None):
    objective_hessian = hessian(evaluate_objective_function, x, 1e-5, gp,objective_function,origin,cost_function,cost_function_parameters)
    return objective_hessian

def evaluate_gp_objective_function(x,objective_function,gp):
    ##this function will always spit out a 1d numpy array
    ##for certain functions, this array will only have one entry
    ##for the other the length == len(x)
    if len(x.shape) == 1: x = np.array([x])
    if objective_function == "variance":
        x = cast_to_index_set(x,gp.value_positions[-1], mode = 'cartesian product')
        res = gp.posterior_covariance(x)
        return b
    if objective_function == "covariance":
        x = cast_to_index_set(x,gp.value_positions[-1], mode = 'cartesian product')
        res = gp.posterior_covariance(x)
        b = res["S(x)"]
        sgn, logdet = np.linalg.slogdet(b)
        return np.array([np.sqrt(sgn * np.exp(logdet))])
    ###################more here: shannon_ig  for instance
    elif objective_function == "shannon_ig":
        x = cast_to_index_set(x,gp.value_positions[-1], mode = 'cartesian product')
        res = gp.shannon_information_gain(x)["sig"]
        return np.array([res])
    elif objective_function == "upper_confidence":
        x = cast_to_index_set(x,gp.value_positions[-1], mode = 'cartesian product')
        m = gp.posterior_mean(x)["f(x)"]
        v = gp.posterior_covariance(x)["v(x)"]
        return m + 3.0*v
    elif objective_function == "maximum":
        x = cast_to_index_set(x,gp.value_positions[-1], mode = 'cartesian product')
        res = gp.posterior_mean(x)["f(x)"]
        return res
    elif objective_function == "minimum":
        x = cast_to_index_set(x,gp.value_positions[-1], mode = 'cartesian product')
        res = gp.posterior_mean(x)["f(x)"]
        return -res

##########################################################################
def find_objective_function_maxima(gp,objective_function,
        origin,number_of_maxima_sought,
        optimization_bounds,
        optimization_method = "global",
        optimization_pop_size = 20,
        optimization_max_iter = 200,
        optimization_tol = 10e-6,
        cost_function = None,
        cost_function_parameters = None,
        dask_client = False):
    bounds = np.array(optimization_bounds)
    print("====================================")
    print("finding objective function maxima...")
    print("optimization method ",optimization_method)
    print("adjusted tolerance: ", optimization_tol)
    print("population size: ", optimization_pop_size)
    print("maximum number of iterations: ",optimization_max_iter)
    print("bounds: ")
    print(bounds)
    print("cost function parameters: ", cost_function_parameters)
    print("====================================")

    if optimization_method == "global":
        opti, func_eval = differential_evolution(
            evaluate_objective_function,
            optimization_bounds,
            tol = optimization_tol,
            popsize = optimization_pop_size,
            max_iter = optimization_max_iter,
            origin = origin,
            gp = gp,
            objective_function = objective_function,
            cost_function = cost_function,
            cost_function_parameters = cost_function_parameters
        )
        opti = np.asarray(opti)
        func_eval = np.asarray(func_eval)
    elif optimization_method == "hgdl":
        ###run differential evo first if hxdy only returns stationary points
        ###then of hgdl is successful, stack results and return
        from hgdl.hgdl import HGDL
        a = HGDL(evaluate_objective_function,
                    evaluate_objective_function_gradient,
                    evaluate_objective_function_hessian,
                    optimization_bounds, number_of_walkers = optimization_pop_size,
                    verbose = False, maxEpochs = optimization_max_iter,
                    args = (gp,objective_function,origin,cost_function,cost_function_parameters))
        #####optimization_max_iter, tolerance here
        a.optimize(dask_client = dask_client)
        res = a.get_latest(number_of_maxima_sought)
        opti = res['x']
        func_eval = res['func evals']
    elif optimization_method == "local":
        x0 = np.random.uniform(low = bounds[:,0],high = bounds[:,1],size = len(bounds))
        a = minimize(
            evaluate_objective_function,
            x0,
            args = (gp,objective_function,origin,cost_function,cost_function_parameters),
            method="L-BFGS-B",
            jac=evaluate_objective_function_gradient,
            bounds = bounds,
            tol = optimization_tol,
            callback = None,
            options = {"maxiter": optimization_max_iter}
            )
        opti = np.array([a["x"]])
        func_eval = np.array(a["fun"])
        if a["success"] is False:
            print("local objective function optimization not successful, solution replaced with random point.")
            opti = np.array(x0)
            if opti.ndim !=2 : opti = np.array([opti])
            func_eval = evaluate_objective_function(x0,
                    gp,objective_function,origin,
                    cost_function,cost_function_parameters)
            if func_eval.ndim != 1: func_eval = np.array([func_eval])
    else:
        raise ValueError("Invalid objective function optimization method given.")

    if func_eval.ndim != 1 or opti.ndim != 2:
        print("f(x): ",func_eval)
        print("x: ",opti)
        raise Exception("The output of the objective function optimization is not 2 dimensional. Please check your objective function.")
    return opti,func_eval

############################################################
############################################################
############################################################
############################################################

def normed_gaussian_function(x, mean, sigma2):
    return (1.0 / np.sqrt(2.0 * np.pi * sigma2)) * np.exp(
        -((x - mean) ** 2) / (2.0 * sigma2)
    )
def gradient(function, point, epsilon = 1e-6,*args):
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
    #args = args[0]
    for i in range(len(point)):
        new_point = np.array(point)
        new_point[i] += epsilon
        gradient[i] = (function(new_point,*args) - function(point,*args))/ epsilon
    return gradient

def hessian(function, point, epsilon = 1e-3, *args):
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
    hessian = np.zeros((len(point),len(point)))
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

            hessian[i,j] = \
            (function(new_point1,*args) - function(new_point2,*args) - function(new_point3,*args) +  function(new_point4,*args))\
            / (4.0*(epsilon**2))
    return hessian


def bhattacharyya_distance(reference_distribution, test_distribution, dx):
    y1 = reference_distribution / (sum(reference_distribution)*dx)
    y2 = test_distribution / (sum(test_distribution)*dx)
    return sum(np.sqrt(y1*y2)) * dx


def cast_to_index_set(x_input,x_output,mode = 'cartesian product'):
    n_orig = len(x_input)
    tasks = len(x_output)
    if mode == 'cartesian product':
        new_points = np.zeros((len(x_input) * len(x_output), len(x_input[0]) + len(x_output[0])))
        counter = 0
        for element in itertools.product(x_input, x_output):
            new_points[counter] = np.concatenate([element[0], element[1]], axis=0)
            counter += 1   ###can't we append?
    elif mode == 'stack':
        new_points = np.column_stack([x_input,x_output])
    return new_points

