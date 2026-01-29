#!/usr/bin/env python
import math
import numpy as np
from loguru import logger
from hgdl.hgdl import HGDL
from scipy.optimize import differential_evolution as devo, minimize
from scipy.stats import norm
from functools import partial
import warnings


##########################################################################
def find_acquisition_function_maxima(gpo, acquisition_function, *,
                                     origin=None,
                                     number_of_maxima_sought=1,
                                     input_set=None,
                                     input_set_dim=None,
                                     optimization_method="global",
                                     optimization_pop_size=20,
                                     optimization_max_iter=10,
                                     optimization_tol=1e-6,
                                     optimization_x0=None,
                                     constraints=(),
                                     cost_function=None,
                                     vectorized=True,
                                     x_out=None,
                                     dask_client=None,
                                     batch_size=10,
                                     info=False):
    bounds = None
    candidates = None
    if input_set is None:
        raise Exception("input_set has to be provided either as a list of Numpy array")
    if isinstance(input_set, np.ndarray):
        bounds = input_set
    elif isinstance(input_set, list):
        candidates = input_set
    else:
        raise Exception("input_set not given in an allowed format")
    opt_obj = None
    func = partial(evaluate_acquisition_function, gpo=gpo,
                   acquisition_function=acquisition_function,
                   origin=origin, dim=input_set_dim,
                   cost_function=cost_function,
                   x_out=x_out)
    grad = partial(gradient, func=func)

    logger.debug("====================================")
    logger.debug(f"Finding acquisition function maxima via {optimization_method} method")
    logger.debug("tolerance: {}", optimization_tol)
    logger.debug("population size: {}", optimization_pop_size)
    logger.debug("maximum number of iterations: {}", optimization_max_iter)
    logger.debug("bounds:")
    logger.debug(bounds)
    logger.debug("====================================")
    if candidates is not None:
        if vectorized is False:
            if dask_client is not None:
                logger.debug("Mapping the acquisition function evaluation over dask workers in batches of size", batch_size)
                res = np.asarray(list(dask_client.gather(dask_client.map(func, candidates, batch_size=batch_size)))).\
                    reshape(len(candidates))
            else:
                logger.debug("Calling the acquisition function on candidates sequentially")
                res = np.asarray(list(map(func, candidates))).reshape(len(candidates))
        else:
            if dask_client is not None:
                logger.debug("Calling the acquisition function on parallelized chunks of size ", batch_size)
                tasks = list(divide_chunks(candidates, batch_size))
                res = np.asarray(list(dask_client.gather(dask_client.map(func, tasks)))).reshape(len(candidates))
            else:
                logger.debug("Calling the acquisition function on all candidates in parallel.")
                res = np.asarray(func(candidates)).reshape(len(candidates))
        sort_indices = np.argsort(res)
        res = res[sort_indices]
        sorted_candidates = [candidates[sort_index] for sort_index in sort_indices]
        candidates = sorted_candidates
        length = min(number_of_maxima_sought, len(candidates))
        opti, func_eval, opt_obj = np.asarray(candidates[0:length]), res[0:length], None

    elif optimization_method == "global":
        opti, func_eval = differential_evolution(
            func,
            input_set,
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
        if dask_client is None: raise Exception("Please provide a dask_client")
        if optimization_x0 is not None: optimization_x0 = optimization_x0.reshape(1, -1)
        opt_obj.optimize(dask_client=dask_client, x0=optimization_x0, tolerance=optimization_tol)
        res = opt_obj.get_final()
        opti = np.asarray([entry["x"] for entry in res])
        func_eval = np.asarray([entry["f(x)"] for entry in res])
        idx = filter_similar_rows(opti, tol=0.01)
        print(opti, idx)
        opti = opti[idx]
        func_eval = func_eval[idx]

        if len(opti) < number_of_maxima_sought:
            warnings.warn("An insufficient number of unique optima identified. " +
                          "Try `total correlation` or the use of candidates by providing them as a list to ask(). ")
        opti = opti[0:min(len(opti), number_of_maxima_sought)]
        func_eval = func_eval[0:min(len(func_eval), number_of_maxima_sought)]

    elif optimization_method == "hgdlAsync":
        opt_obj = HGDL(func,
                       grad,
                       bounds,
                       num_epochs=optimization_max_iter,
                       local_optimizer="L-BFGS-B",
                       constraints=constraints)
        if dask_client is None: raise Exception("Please provide a dask_client")
        if optimization_x0 is not None: optimization_x0 = optimization_x0.reshape(1, -1)
        opt_obj.optimize(dask_client=dask_client, x0=optimization_x0, tolerance=optimization_tol)
        opti = np.zeros((1, input_set_dim))
        func_eval = np.zeros(1)

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
            options={"maxiter": optimization_max_iter,
                     'disp': info}
        )
        opti = np.array([a["x"]])
        func_eval = np.array(a["fun"])
        if np.ndim(func_eval) == 0: func_eval = np.array([func_eval])
        if a["success"] is False:
            logger.warning(
                "local acquisition function optimization not successful, solution replaced with random point.")
            opti = np.array(x0)
            if opti.ndim != 2: opti = np.array([opti])
            func_eval = evaluate_acquisition_function(x0, gpo=gpo, acquisition_function=acquisition_function,
                                                      origin=origin, dim=input_set_dim, cost_function=cost_function,
                                                      x_out=x_out)
            if np.ndim(func_eval) != 1: func_eval = np.array([func_eval])
    else:
        raise ValueError("Invalid acquisition function optimization method given: ", optimization_method)
    if np.ndim(func_eval) != 1:
        logger.error("f_a(x): ", func_eval)
        logger.error("x: ", opti)
        raise Exception(
            "The output of the acquisition function optimization dim (f) != 1 or dim(x) != 2. Please check your "
            "acquisition function. It should return a 1-d numpy array")
    return opti, -func_eval, opt_obj


############################################################
############################################################
############################################################
############################################################
def evaluate_acquisition_function(x, *, gpo=None, acquisition_function=None, origin=None, dim=None,
                                  cost_function=None, x_out=None):
    ##########################################################
    ####this function evaluates a default or a user-defined acquisition function
    ##########################################################
    if isinstance(x, np.ndarray):
        if np.ndim(x) == 1:
            x = x.reshape(-1, dim)
        elif np.ndim(x) > 2:
            raise Exception("Wrong input dim in `x`.")
    elif isinstance(x, list) and isinstance(x[0], np.ndarray):
        try: x = np.asarray(x).reshape(len(x), dim)
        finally: x = x

    if x_out is not None and np.ndim(x_out) != 1: raise Exception(
        "x_out in evaluate_acquisition_function has to be a 1d numpy array.")

    if cost_function is not None and origin is not None:
        cost_eval = cost_function(origin, x)
    else:
        cost_eval = 1.0
    # for user defined acquisition function
    if callable(acquisition_function):
        return -acquisition_function(x, gpo) / cost_eval
    else:
        obj_eval = evaluate_gp_acquisition_function(x, acquisition_function, gpo, x_out=x_out)
        obj_eval = -obj_eval / cost_eval
    return obj_eval


def evaluate_gp_acquisition_function(x, acquisition_function, gpo, x_out):
    ##this function will always spit out a 1d numpy array because it assumes several `x`.
    ##For certain functions, this array will only have one entry
    ##for the other the length == len(x)
    if isinstance(x, np.ndarray) and np.ndim(x) == 1: raise Exception(
        "1d array given in evaluate_gp_acquisition_function. It has to be 2d")
    if x_out is None:
        all_acq_func = ["variance", "relative information entropy", "relative information entropy set",
                        "ucb", "lcb", "maximum", "minimum", "gradient", "expected improvement",
                        "probability of improvement", "target probability", "total correlation"]
        if acquisition_function == "variance":
            res = np.sqrt(gpo.posterior_covariance(x, variance_only=True)["v(x)"])
            return res
        elif acquisition_function == "relative information entropy":
            res = -gpo.gp_relative_information_entropy(x)["RIE"]
            return np.array([res])
        elif acquisition_function == "relative information entropy set":
            res = -gpo.gp_relative_information_entropy_set(x)["RIE"]
            return res
        elif acquisition_function == "ucb":
            m = gpo.posterior_mean(x)["m(x)"]
            v = gpo.posterior_covariance(x, variance_only=True)["v(x)"]
            return m + 3.0 * np.sqrt(v)
        elif acquisition_function == "lcb":
            m = gpo.posterior_mean(x)["m(x)"]
            v = gpo.posterior_covariance(x, variance_only=True)["v(x)"]
            return -(m - 3.0 * np.sqrt(v))
        elif acquisition_function == "maximum":
            res = gpo.posterior_mean(x)["m(x)"]
            return res
        elif acquisition_function == "gradient":
            mean_grad = gpo.posterior_mean_grad(x)["dm/dx"]
            std = np.sqrt(gpo.posterior_covariance(x, variance_only=True)["v(x)"])
            res = np.linalg.norm(mean_grad, axis=1) * std
            return res
        elif acquisition_function == "minimum":
            res = gpo.posterior_mean(x)["m(x)"]
            return -res
        elif acquisition_function == "probability of improvement":
            m = gpo.posterior_mean(x)["m(x)"]
            std = np.sqrt(gpo.posterior_covariance(x, variance_only=True)["v(x)"])
            last_best = np.max(gpo.y_data)
            return norm.cdf((m - last_best) / (std + 1e-9))
        elif acquisition_function == "total correlation":
            return -np.array([gpo.gp_total_correlation(x)["total correlation"]])
        elif acquisition_function == "expected improvement":
            m = gpo.posterior_mean(x)["m(x)"]
            std = np.sqrt(gpo.posterior_covariance(x, variance_only=True)["v(x)"])
            last_best = np.max(gpo.y_data)
            a = (m - last_best)
            a[a < 0.] = 0.
            gamma = a / (std + 1e-9)
            pdf = norm.pdf(gamma)
            cdf = norm.cdf(gamma)
            return std * (gamma * cdf + pdf)
        elif acquisition_function == "target probability":
            try:
                a = gpo.args["a"]
                b = gpo.args["b"]
            except:
                raise Exception("Reading the arguments for acq func `target probability` failed.")
            mean = gpo.posterior_mean(x, x_out=x_out)["m(x)"].reshape(len(x))
            cov = gpo.posterior_covariance(x, x_out=x_out)["v(x)"].reshape(len(x)) + 1e-9
            result = np.zeros((len(x)))
            for i in range(len(x)):
                result[i] = 0.5 * (math.erf((b - mean[i]) / np.sqrt(2. * cov[i]))) - math.erf(
                    (a - mean[i]) / np.sqrt(2. * cov[i]))
            return result
        else:
            raise Exception("No valid acquisition function string provided. Choose from ", all_acq_func)

    else:
        all_acq_func = ["variance", "relative information entropy", "relative information entropy set",
                        "ucb", "lcb", "expected improvement", "total correlation"]
        if acquisition_function == "variance":
            res = gpo.posterior_covariance(x, x_out=x_out, variance_only=True)["v(x)"]
            return np.sum(res, axis=1)
        elif acquisition_function == "relative information entropy":
            res = -gpo.gp_relative_information_entropy(x, x_out=x_out)["RIE"]
            return np.array([res])
        elif acquisition_function == "relative information entropy set":
            res = -gpo.gp_relative_information_entropy_set(x, x_out=x_out)["RIE"]
            return res
        elif acquisition_function == "total correlation":
            return -np.array([gpo.gp_total_correlation(x, x_out=x_out)["total correlation"]])
        elif acquisition_function == "ucb":
            m = gpo.posterior_mean(x, x_out=x_out)["m(x)"]
            av_m = np.sum(m, axis=1)
            v = gpo.posterior_covariance(x, x_out=x_out, variance_only=True)["v(x)"]
            av_v = np.sum(v, axis=1)
            return av_m + 3.0 * np.sqrt(av_v)
        elif acquisition_function == "lcb":
            m = gpo.posterior_mean(x, x_out=x_out)["m(x)"]
            av_m = np.sum(m, axis=1)
            v = gpo.posterior_covariance(x, x_out=x_out, variance_only=True)["v(x)"]
            av_v = np.sum(v, axis=1)
            return -(av_m - 3.0 * np.sqrt(av_v))
        elif acquisition_function == "expected improvement":
            m = gpo.posterior_mean(x, x_out=x_out)["m(x)"]
            m = np.sum(m, axis=1)
            std = np.sqrt(gpo.posterior_covariance(x, x_out=x_out, variance_only=True)["v(x)"])
            std = np.sum(std, axis=1)
            last_best = np.max(gpo.y_data)
            a = (m - last_best)
            a[a < 0.] = 0.
            gamma = a / (std + 1e-9)
            pdf = norm.pdf(gamma)
            cdf = norm.cdf(gamma)
            return std * (gamma * cdf + pdf)
        else:
            raise Exception("No valid acquisition function string provided. Choose from ", all_acq_func)


def differential_evolution(func,
                           bounds,
                           tol,
                           popsize,
                           max_iter=100,
                           x0=None,
                           constraints=(),
                           disp=False,
                           vectorized=True):
    if vectorized: updating = 'deferred'
    else: updating = 'immediate'
    res = devo(partial(acq_function_vectorization_wrapper, func=func, vectorized=vectorized), bounds, tol=tol, x0=x0,
               maxiter=max_iter, popsize=popsize, polish=False, disp=disp, constraints=constraints,
               vectorized=vectorized, updating=updating)
    return [list(res["x"])], list([res["fun"]])


def acq_function_vectorization_wrapper(x, func=None, vectorized=False):
    if vectorized is True:
        acq = func(x.T)
    else:
        acq = func(x)
    return acq


def gradient(x, func=None):
    epsilon = 1e-6
    grad = np.zeros(len(x))
    for i in range(len(x)):
        new_point = np.array(x)
        new_point[i] += epsilon
        grad[i] = (func(new_point)[0] - func(x)[0]) / epsilon
    return grad


def filter_similar_rows(arr, tol=1.):
    rounded = np.round(arr / tol) * tol
    idx = np.unique(rounded, return_index=True, axis=0)[1]
    return np.sort(idx)


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
