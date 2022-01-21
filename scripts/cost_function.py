from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution as devo


# define global variable here:
# a = 1.0


def l2_cost(origin, x, arguments=None):
    offset = arguments["offset"]
    slope = arguments["slope"]
    return slope * np.linalg.norm(np.abs(np.subtract(origin, x)), axis=1) + offset


def l1_cost(origin, x, arguments=None):
    offset = arguments["offset"]
    slope = arguments["slope"]
    d = np.abs(np.subtract(origin, x))
    c = (d * slope) + offset
    n = np.sum(c, axis=1)
    return n


########################################
######update functions##################
########################################

def _update_cost_function(costs,
                          parameters,
                          cost_func: Callable,
                          compute_cost_misfit_func: Callable):
    print("Cost adjustment in progress...")
    print("old cost parameters: ", parameters)
    bounds = 0.0
    # print(bounds)
    # input()
    ###remove out-liers:
    origins = []
    points = []
    motions = []
    c = []
    cost_per_motion = []
    for i in range(len(costs)):
        origins.append(costs[i]["origin"])
        points.append(costs[i]["point"])
        motions.append(abs(costs[i]["origin"] - costs[i]["point"]))
        c.append(costs[i]["cost"])
        cost_per_motion.append(costs[i]["cost"] / cost_func(costs[i]["origin"], costs[i]["point"], parameters))
    mean_costs_per_distance = np.mean(np.asarray(cost_per_motion))
    sd = np.std(np.asarray(cost_per_motion))
    for element in cost_per_motion:
        if (
                element >= mean_costs_per_distance - 2.0 * sd
                and element <= mean_costs_per_distance + 2.0 * sd
        ):
            continue
        else:
            motions.pop(cost_per_motion.index(element))
            c.pop(cost_per_motion.index(element))
            origins.pop(cost_per_motion.index(element))
            points.pop(cost_per_motion.index(element))
            cost_per_motion.pop(cost_per_motion.index(element))
    res = devo(compute_cost_misfit_func,
               bounds,
               args=(origins, points, c),
               tol=1e-6,
               disp=True,
               maxiter=300,
               popsize=20,
               polish=False)
    arguments = {"offset": res["x"][0], "slope": res["x"][1]}
    print("New cost parameters: ", arguments)
    return arguments


def update_l2_cost_function(costs, parameters):
    return _update_cost_function(costs, parameters, l2_cost, compute_l2_cost_misfit)


def update_l1_cost_function(costs, parameters):
    return _update_cost_function(costs, parameters, l1_cost, compute_l1_cost_misfit)


########################################################
########################################################
########################################################
########################################################


def linear_cost(c, x):
    return c * x


def linear_cost_derivative(c, x):
    return c


def linear_cost_derivatice2(c, x):
    return 0.0


def _compute_cost_misfit(params, origins, points, costs, cost_func: Callable):
    parameters = {"offset": params[0], "slope": params[1]}
    sum1 = 0.0
    for idx in range(len(points)):
        sum1 = sum1 + (
                (cost_func(origins[idx], points[idx], parameters) - costs[idx]) ** 2)
    return sum1


def compute_l2_cost_misfit(params, origins, points, costs):
    return _compute_cost_misfit(params, origins, points, costs, l2_cost)

def compute_l1_cost_misfit(params, origins, points, costs):
    return _compute_cost_misfit(params, origins, points, costs, l1_cost)
