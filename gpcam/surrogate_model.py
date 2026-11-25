#!/usr/bin/env python
import copy
import math
import numpy as np
from loguru import logger
from hgdl.hgdl import HGDL
from scipy.optimize import differential_evolution as devo, minimize
from scipy.stats import norm
from functools import partial
import warnings


#: Acquisition functions that score a *set* of points with a single number rather than
#: scoring each point on its own. Only these can express whether a batch is jointly
#: informative, so only these can be optimized over subsets of a candidate list.
SET_VALUED_ACQUISITION_FUNCTIONS = frozenset({"total correlation", "relative information entropy"})


def _greedy_batch_from_candidates(func, candidates, n, acquisition_function):
    """Choose ``n`` candidates that are jointly valuable, by greedy forward selection.

    Ranking candidates individually and taking the top ``n`` -- what ask() does for the
    point-wise acquisition functions -- says nothing about whether the batch is any good
    as a batch. Nothing prevents the best scorers from sitting on top of one another, and
    in practice they do: on 500 candidates in the unit square the five individually best
    points came out 0.018 apart, five copies of the same experiment.

    Choosing the genuinely best subset is combinatorial -- 8e17 subsets for 5 out of
    10000 -- but it does not have to be enumerated. Greedy forward selection takes the
    best single point, then the point that best complements it, and so on, which is
    ``n`` passes over the candidates instead of "n choose N" subsets. Criteria of this
    kind are approximately submodular, for which greedy is a standard and well-founded
    choice (Nemhauser et al. 1978; Krause, Singh & Guestrin 2008 study exactly this
    problem -- placing k sensors under a GP -- and report near-optimal results).

    Cost is ``n * len(candidates)`` set evaluations: about 16 s for 5 out of 10000, and
    linear in both, so it stays usable as either grows.

    ``func`` returns the *negated* acquisition (the optimizers minimize), so the best
    candidate at each step is the one that minimizes it.
    """
    remaining = list(range(len(candidates)))
    n = min(n, len(remaining))
    chosen, chosen_idx = [], []
    for step in range(n):
        scores = [func(chosen + [candidates[j]]) for j in remaining]
        scores = [float(np.asarray(s).reshape(-1)[0]) for s in scores]
        best = remaining[int(np.argmin(scores))]
        chosen.append(candidates[best])
        chosen_idx.append(best)
        remaining.remove(best)
        logger.debug("greedy batch selection picked {} of {}: candidate {}", step + 1, n, best)
    # one number for the batch, not n numbers for n points: the whole point of a
    # set-valued acquisition is that the set is what carries the value
    set_value = float(np.asarray(func(chosen)).reshape(-1)[0])
    return np.asarray(chosen), np.array([set_value]), None


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
    if candidates is not None and number_of_maxima_sought > 1 and \
            acquisition_function in SET_VALUED_ACQUISITION_FUNCTIONS:
        opti, func_eval, opt_obj = _greedy_batch_from_candidates(
            func, candidates, number_of_maxima_sought, acquisition_function)

    elif candidates is not None:
        if number_of_maxima_sought > 1 and not callable(acquisition_function):
            warnings.warn(
                f"ask() is returning the {number_of_maxima_sought} individually best "
                f"candidates under `{acquisition_function}`, which scores one point at a "
                f"time. They are not mutually optimal and are often near-duplicates, since "
                f"nothing stops the top scorers from sitting on top of each other. For a "
                f"batch chosen to be jointly informative use "
                f"{sorted(SET_VALUED_ACQUISITION_FUNCTIONS)}, which score a whole set.")
        elif number_of_maxima_sought > 1:
            warnings.warn(
                f"ask() is returning the {number_of_maxima_sought} individually best "
                f"candidates under a user-supplied acquisition function. gpCAM cannot tell "
                f"whether it scores a set or a single point, so it is applied point by "
                f"point and the result is not mutually optimal. For a jointly informative "
                f"batch use {sorted(SET_VALUED_ACQUISITION_FUNCTIONS)}.")
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
    # unreachable in practice: scipy rejects a non-scalar acquisition before this point
    # for `global`, and the `local` and candidate-list paths normalize the shape
    if np.ndim(func_eval) != 1:  # pragma: no cover - defensive
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
        try:
            x = np.asarray(x).reshape(len(x), dim)
        except Exception:
            pass

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
                        "probability of improvement", "target probability", "total correlation",
                        "knowledge gradient", "noisy expected improvement"]
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
            m = gpo.posterior_mean(x)["m(x)"].reshape(len(x))
            std = np.sqrt(gpo.posterior_covariance(x, variance_only=True)["v(x)"]).reshape(len(x))
            last_best = np.max(gpo.y_data)
            return _expected_improvement(m - last_best, std)
        elif acquisition_function == "knowledge gradient":
            return knowledge_gradient(x, gpo, x_out=None)
        elif acquisition_function == "noisy expected improvement":
            return noisy_expected_improvement(x, gpo, x_out=None)
        elif acquisition_function == "target probability":
            try:
                a = gpo.args["a"]
                b = gpo.args["b"]
            except (KeyError, TypeError):
                raise Exception("Reading the arguments for acq func `target probability` failed.")
            mean = gpo.posterior_mean(x, x_out=x_out)["m(x)"].reshape(len(x))
            cov = gpo.posterior_covariance(x, x_out=x_out)["v(x)"].reshape(len(x)) + 1e-9
            result = np.zeros((len(x)))
            for i in range(len(x)):
                result[i] = 0.5 * (math.erf((b - mean[i]) / np.sqrt(2. * cov[i])) - math.erf(
                    (a - mean[i]) / np.sqrt(2. * cov[i])))
            return result
        else:
            raise Exception("No valid acquisition function string provided. Choose from ", all_acq_func)

    else:
        all_acq_func = ["variance", "relative information entropy", "relative information entropy set",
                        "ucb", "lcb", "expected improvement", "total correlation",
                        "knowledge gradient", "noisy expected improvement"]
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
            # Scalarized to the task-summed objective g(x) = sum_t f(x, t), as ucb/lcb
            # and knowledge gradient / noisy EI do.
            m = np.sum(gpo.posterior_mean(x, x_out=x_out)["m(x)"].reshape(len(x), len(x_out)), axis=1)
            v = gpo.posterior_covariance(x, x_out=x_out, variance_only=True)["v(x)"].reshape(len(x), len(x_out))
            # Var(sum_t f) under task independence -- sqrt(sum of variances), not the
            # sum of standard deviations (which assumes perfect correlation). Matches
            # ucb/lcb above; `knowledge gradient` / `noisy expected improvement` get
            # this exactly from the cross-task covariance at O((N*T)^2) cost.
            std = np.sqrt(np.sum(v, axis=1))
            # The incumbent must use the same scalarization as `m`: the best *task-summed*
            # value. `np.max(gpo.y_data)` is a single (point, task) entry and is not
            # comparable with a sum over tasks.
            last_best = _incumbent_task_sum(gpo, x_out)
            return _expected_improvement(m - last_best, std)
        elif acquisition_function == "knowledge gradient":
            return knowledge_gradient(x, gpo, x_out=x_out)
        elif acquisition_function == "noisy expected improvement":
            return noisy_expected_improvement(x, gpo, x_out=x_out)
        else:
            raise Exception("No valid acquisition function string provided. Choose from ", all_acq_func)


def _data_input_points(gpo):
    """The input points of the data, in every multi-task mode.

    This is the one place that knows how each mode stores its data: ``'independent'``
    and ``'kronecker'`` keep the inputs as given; ``'cartesian'`` stores one row
    ``[x, task]`` per observed (input, output) pair, task-major and with missing outputs
    left out, so its inputs are the distinct ``x``, in the order they first appear --
    the order of the data, whenever the first output is observed everywhere (for a
    non-Euclidean list, the ``x`` of each pair; duplicates do no harm where this is used).
    """
    x = gpo.x_data
    if gpo.multi_task_mode != "cartesian": return x
    if isinstance(x, list): return [p[0] for p in x]
    first = np.unique(x[:, :-1], axis=0, return_index=True)[1]
    return x[np.sort(first), :-1]


def _incumbent_task_sum(gpo, x_out):
    """The incumbent of multi-task ``expected improvement``: the best posterior mean,
    summed over the outputs ``x_out``, at the data's input points.

    It has to match how the candidates' mean is scalarized, otherwise the "improvement"
    compares a sum over outputs against something that is not one. Summing the
    observations instead fails whenever an output is missing at a point (a NaN in the
    'cartesian' mode, which made every EI value NaN), and treats noisy observations as
    exact; the posterior mean is defined at every data point, for every output, in every
    mode.
    """
    m = gpo.posterior_mean(_data_input_points(gpo), x_out=x_out)["m(x)"]
    return float(np.max(np.sum(m, axis=1)))


def _expected_improvement(imp, std):
    """Closed-form expected improvement from the improvement mean and its std.

    ``imp = mu - incumbent`` and ``std`` are 1d arrays of the same length. Returns
    ``imp * Phi(z) + std * phi(z)`` with ``z = imp / std``, which is the textbook EI:
    non-negative, smooth, and strictly increasing in ``imp``.

    The improvement is deliberately *not* clipped before forming ``z`` -- doing so
    pins ``z`` to 0 at every candidate below the incumbent, collapsing EI to the
    constant ``std * phi(0)`` and turning it into a scaled ``variance`` acquisition
    (issue #55). Clipping is unnecessary: the expectation is already non-negative.

    The ``np.maximum`` guards the case ``std <~ 1e-9``, where ``z`` is dominated by
    the epsilon rather than the real std and the two terms can cancel to a small
    negative value.
    """
    z = imp / (std + 1e-9)
    return np.maximum(imp * norm.cdf(z) + std * norm.pdf(z), 0.)


##########################################################################
# Knowledge gradient and noisy expected improvement
#
# Both are lookahead acquisition functions that reason about the posterior of the
# *objective* rather than the raw (noisy) observations, which is why they need the
# cross-covariance between candidates and reference points, not just point-wise
# variance. For multi-task GPs (``x_out is not None``) both operate on the
# task-summed objective g(x) = sum_t f(x, t), matching how the built-in multi-task
# ``expected improvement`` / ``ucb`` scalarize the outputs.
##########################################################################
def _acq_arg(gpo, key, default):
    """Read an optional acquisition setting from ``gpo.args`` (a dict or None)."""
    a = getattr(gpo, "args", None)
    if isinstance(a, dict) and key in a and a[key] is not None:
        return a[key]
    return default


def _reference_points(gpo, x_out, cap, rng):
    """The distinct input points of the data: the reference set R of KG and NEI.

    For multi-task GPs these are the data's input points (:func:`_data_input_points`).
    Replicated measurements are one input, not several: subsampling *rows* spent the cap
    on repeats and could drop the best input, which lowers the incumbent. Above ``cap``
    distinct inputs, the ``(cap + 1) // 2`` with the highest (task-summed) posterior mean
    are kept -- the incumbent lives there -- plus a random rest; the rng is seeded, so the
    subsample is stable within one ``ask()``.
    """
    if x_out is None:
        xr = np.asarray(gpo.x_data, dtype=float)
    else:
        xr = np.asarray(_data_input_points(gpo), dtype=float)
    xr = np.unique(xr, axis=0)
    if len(xr) > cap:
        mu = np.asarray(gpo.posterior_mean(xr, x_out=x_out)["m(x)"]).reshape(len(xr), -1).sum(axis=1)
        top = np.argsort(mu)[::-1][:(cap + 1) // 2]
        rest = np.setdiff1d(np.arange(len(xr)), top)
        rest = np.sort(rng.choice(rest, size=cap - len(top), replace=False))
        xr = xr[np.concatenate([top, rest])]
    return xr


def _assumed_observation_noise(gpo, x_out):
    """Assumed observation-noise variance for the (scalarized) objective.

    Averages the per-observation noise variances. fvgp lets a ``noise_function``
    return either a 1-d vector of variances or a **full (N, N) noise covariance**
    (``addKV`` and ``add_noise`` both accept a 2-d ``V``), so the matrix case has to
    be handled: the per-observation variances are its diagonal. Averaging the whole
    matrix instead would divide by N -- for uncorrelated noise expressed as a
    diagonal matrix that understates the noise by exactly a factor of N.

    For the task-summed multi-task objective the relevant quantity is
    ``Var(sum_t eps(x, t))``. With a full covariance over the 'cartesian' mode's
    (point, task) pairs that is the sum of each point's (T, T) noise block, which
    accounts for cross-task noise correlation exactly; the ordering is task-major
    (k = point + Npts*task). In the other modes the noise is one variance per input,
    the same for every output and independent between outputs, and with only
    per-observation variances the tasks have to be assumed uncorrelated: both give
    ``len(x_out)`` times the mean.
    """
    base = None
    task_summed = False          # True once `base` is already Var(sum over tasks)
    try:
        v = np.asarray(gpo.get_data()["measurement variances"], dtype=float)
        if v.size and v.ndim == 2 and v.shape[0] == v.shape[1]:
            # the (T, T) blocks exist only for a full 'cartesian' grid with every output asked for
            T = len(x_out) if gpo.multi_task_mode == "cartesian" and len(x_out) == gpo.output_num else 0
            if T > 0 and v.shape[0] == T * len(_data_input_points(gpo)):
                # exact Var(sum_t eps) per point, averaged over the data points
                P = v.shape[0] // T
                blocks = v.reshape(T, P, T, P)               # [task, point, task, point]
                base = float(np.mean([blocks[:, i, :, i].sum() for i in range(P)]))
                task_summed = True
            else:
                base = float(np.mean(np.diag(v)))
        elif v.size:
            base = float(np.mean(v))
    except Exception:
        base = None
    if base is None or not np.isfinite(base) or base <= 0.0:
        base, task_summed = 1e-6, False
    if x_out is not None and not task_summed:
        base *= len(x_out)       # tasks assumed uncorrelated
    return base


def _scalarized_blocks(gpo, x_ref, x_cand, x_out):
    """Posterior mean and covariance of the (task-summed) objective on [x_ref; x_cand].

    Returns
    -------
    mu_ref   : (M,)   posterior mean at reference points
    cov_ref  : (M, M) posterior covariance among reference points
    mu_cand  : (N,)   posterior mean at candidates
    var_cand : (N,)   posterior variance at candidates
    cross    : (M, N) posterior covariance between reference points and candidates
    """
    M, N = len(x_ref), len(x_cand)
    stack = np.vstack([x_ref, x_cand])
    mean = np.asarray(gpo.posterior_mean(stack, x_out=x_out)["m(x)"])
    S = np.asarray(gpo.posterior_covariance(stack, x_out=x_out)["S"])
    if x_out is None:
        mu = mean.reshape(-1)
        cov = S
    else:
        # fvgp's structured multi-task results: m(x) is (P, T) and S is (P, P, T, T) with
        # S[i, j, t, u] = Cov(f(x_i, t), f(x_j, u)), in every multi-task mode, so summing
        # the output axes gives Cov(sum_t f(., t), sum_u f(., u)) without touching the
        # task-major order of the flat arrays.
        mu = mean.sum(axis=1)
        cov = S.sum(axis=(2, 3))
    cov = 0.5 * (cov + cov.T)
    mu_ref, mu_cand = mu[:M], mu[M:]
    cov_ref = cov[:M, :M]
    cross = cov[:M, M:]
    var_cand = np.clip(np.diag(cov)[M:], 0.0, None)
    return mu_ref, cov_ref, mu_cand, var_cand, cross


def _expected_max_of_affine(a, b):
    """Exact ``E[max_i (a_i + b_i Z)]`` for a standard normal ``Z``.

    Computed by intersecting the lines ``a_i + b_i z`` to find the upper envelope,
    then integrating each dominant segment against the normal density. This is the
    core of the correlated knowledge-gradient computation
    (Frazier, Powell & Dayanik 2009).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    order = np.lexsort((a, b))          # by slope b, ties broken by intercept a
    a, b = a[order], b[order]
    keep = np.concatenate([b[1:] != b[:-1], [True]])   # dedupe equal slopes, keep max a
    a, b = a[keep], b[keep]
    n = len(a)
    if n == 1:
        return float(a[0])
    idx = [0]
    z = [-np.inf]                       # z[k] = left breakpoint of dominant line idx[k]
    for i in range(1, n):
        while True:
            j = idx[-1]
            zi = (a[j] - a[i]) / (b[i] - b[j])   # crossing of line j and line i (b[i]>b[j])
            if len(idx) > 1 and zi <= z[-1]:
                idx.pop(); z.pop()               # line j never reaches the envelope
            else:
                break
        idx.append(i); z.append(zi)
    a, b = a[idx], b[idx]
    zl = np.array(z)
    zr = np.concatenate([zl[1:], [np.inf]])
    # On (zl_k, zr_k) line k is the max; E[(a_k+b_k Z)1{zl<Z<zr}] uses
    # E[Z 1{alpha<Z<beta}] = phi(alpha) - phi(beta).
    return float(np.sum(a * (norm.cdf(zr) - norm.cdf(zl))
                        + b * (norm.pdf(zl) - norm.pdf(zr))))


def _jitter_cholesky(cov):
    """Cholesky factor of ``cov`` with adaptive jitter; eigen-floor as last resort."""
    cov = 0.5 * (cov + cov.T)
    n = len(cov)
    jit = 1e-9 * (np.trace(cov) / max(n, 1) + 1e-12)
    for _ in range(8):
        try:
            return np.linalg.cholesky(cov + jit * np.eye(n))
        except np.linalg.LinAlgError:
            jit *= 10.0
    w, Q = np.linalg.eigh(cov)
    return Q @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))


def knowledge_gradient(x, gpo, x_out=None):
    """Knowledge-gradient acquisition (maximized).

    KG(x) is the expected increase in the maximum of the posterior mean after a
    (fantasized) measurement at ``x``, over the reference set of data points plus
    ``x`` itself (the KGCP scheme, Scott, Frazier & Powell 2011). Returns a 1d array
    of one non-negative score per row of ``x``.

    Optional ``gpo.args`` keys: ``kg_reference_set_size`` (default 100),
    ``kg_seed`` (default 0).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    cap = int(_acq_arg(gpo, "kg_reference_set_size", 100))
    rng = np.random.default_rng(int(_acq_arg(gpo, "kg_seed", 0)))
    x_ref = _reference_points(gpo, x_out, cap, rng)
    mu_ref, _cov_ref, mu_cand, var_cand, cross = _scalarized_blocks(gpo, x_ref, x, x_out)
    noise = _assumed_observation_noise(gpo, x_out)
    base_ref = np.max(mu_ref)
    out = np.empty(len(x))
    for j in range(len(x)):
        denom = np.sqrt(max(var_cand[j] + noise, 1e-12))
        a = np.append(mu_ref, mu_cand[j])                    # intercepts: current mean
        b = np.append(cross[:, j], var_cand[j]) / denom      # slopes: predictive update
        out[j] = _expected_max_of_affine(a, b) - max(base_ref, mu_cand[j])
    out[out < 0.0] = 0.0                                      # KG is non-negative in theory
    return out


def noisy_expected_improvement(x, gpo, x_out=None):
    """Noisy expected-improvement acquisition (maximized).

    Standard EI treats the incumbent (best value so far) as known, which is wrong
    under observation noise. Noisy-EI (Letham et al. 2019) averages EI over the
    posterior distribution of the incumbent: the best value across Monte-Carlo
    samples of the objective at the observed points. Returns a 1d array of one
    non-negative score per row of ``x``. Common random numbers (a seeded rng) keep
    the score smooth across candidate evaluations.

    The candidate is *not* independent of the incumbent: each sample
    ``f_ref = mu_ref + L z`` of the objective at the reference points also moves
    ``f(x)``, which given that sample is Gaussian with mean ``mu_x + w^T z`` and
    variance ``var_x - |w|^2``, ``w = L^{-1} Cov(f_ref, f(x))``. EI is evaluated in
    closed form under that conditional, as in Letham et al. Ignoring the
    cross-covariance (gpCAM <= 8.7.0) gave a positive score at and near the
    incumbent, where ``f(x)`` cannot exceed ``max f_ref`` by much.

    Optional ``gpo.args`` keys: ``nei_samples`` (default 128),
    ``nei_reference_set_size`` (default 100), ``nei_seed`` (default 0).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return _nei_increment(x, gpo, _nei_state(gpo, x_out), x_out)


def _nei_state(gpo, x_out, x_pending=None):
    """The joint posterior sample of the objective at the reference set, for one ``ask()``.

    Holds the reference points ``x`` (R, then any pending points P in order), their
    posterior mean ``mu``, a square factor ``L`` with ``L L^T = Cov(f(R u P))``, the
    standard normals ``Z`` (one row per point, ``nei_samples`` columns), the generator
    that draws further rows, and ``M``, the incumbent max f(R u P) of each sample.

    Rows of ``Z`` are drawn as points are added. A generator's draws are sequential, so
    this is the same sample as drawing every row up front: the reference sample is fixed
    within one ``ask()`` (common random numbers), and a batch built from pending points
    ``P`` sees exactly the draws a longer batch whose first picks were ``P`` would.
    """
    n_samples = int(_acq_arg(gpo, "nei_samples", 128))
    cap = int(_acq_arg(gpo, "nei_reference_set_size", 100))
    rng = np.random.default_rng(int(_acq_arg(gpo, "nei_seed", 0)))
    x_ref = _reference_points(gpo, x_out, cap, rng)
    mu, cov, _, _, _ = _scalarized_blocks(gpo, x_ref, x_ref[:0], x_out)
    L = _jitter_cholesky(cov)                                # L L^T = cov (+ jitter)
    Z = rng.standard_normal((len(x_ref), n_samples))
    state = {"x": x_ref, "mu": mu, "L": L, "Z": Z, "rng": rng,
             "M": np.max(mu[:, None] + L @ Z, axis=0)}
    if x_pending is not None and len(x_pending):
        state = _nei_append(state, gpo, x_pending, x_out)
    return state


def _nei_append(state, gpo, x_new, x_out):
    """Add points to the reference set of an NEI state, one at a time, exactly.

    For a new point p with ``c = Cov(f(R u P), f(p))``: ``l = L^{-1} c`` and
    ``d = sqrt(Var f(p) - |l|^2)``, so ``[[L, 0], [l^T, d]]`` factors the extended
    covariance and ``f(p) = mu_p + l^T z + d z_p`` is its sample. Earlier rows are
    untouched. ``L`` need only be square (the eigen-floor fallback of
    :func:`_jitter_cholesky` qualifies), since ``l`` is a general solve. A point already
    in the set has ``d = 0``: its sample repeats the existing one, so it cannot raise
    the incumbent. Only the factor's diagonal entry is floored (at a 1e-10 relative
    jitter), to keep ``L`` invertible for later solves.
    """
    x_new = np.atleast_2d(np.asarray(x_new, dtype=float))
    X, mu, L, Z, M = state["x"], state["mu"], state["L"], state["Z"], state["M"]
    for p in x_new:
        _, _, mu_p, var_p, c = _scalarized_blocks(gpo, X, p[None, :], x_out)
        m = len(X)
        l = np.linalg.solve(L, c[:, 0])                      # (m,)
        resid = var_p[0] - l @ l
        floor = 1e-10 * max(np.mean(np.sum(L**2, axis=1)), var_p[0], 1e-300)
        if len(Z) == m:
            Z = np.vstack([Z, state["rng"].standard_normal((1, Z.shape[1]))])
        f_p = mu_p[0] + l @ Z[:m] + np.sqrt(max(resid, 0.0)) * Z[m]   # (K,) samples of f(p)
        L = np.block([[L, np.zeros((m, 1))], [l[None, :], np.array([[np.sqrt(max(resid, floor))]])]])
        X, mu, M = np.vstack([X, p]), np.append(mu, mu_p[0]), np.maximum(M, f_p)
    return dict(state, x=X, mu=mu, L=L, Z=Z, M=M)


def _nei_increment(x, gpo, state, x_out):
    """NEI's value of adding ``x`` to the reference set, Delta(x | P) = E[(f(x) - max f(R u P))^+].

    Closed form per sample: given ``f(R u P) = mu + L z``, f(x) is Gaussian with mean
    ``mu_x + w^T z`` and variance ``var_x - |w|^2``, ``w = L^{-1} Cov(f(R u P), f(x))``,
    and EI against that sample's incumbent ``M`` is exact. With no pending points this
    is the single-point NEI; for a batch, the increments of the greedy picks add up to
    the batch's q-NEI (Letham et al. 2019).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    X, L, Z, M = state["x"], state["L"], state["Z"], state["M"]
    _, _, mu_cand, var_cand, cross = _scalarized_blocks(gpo, X, x, x_out)
    W = np.linalg.solve(L, cross)                            # (m, N)
    cond_mean = mu_cand[:, None] + W.T @ Z[:len(X)]          # (N, K)
    cond_var = var_cand - np.sum(W**2, axis=0)               # (N,)
    std = np.sqrt(np.maximum(cond_var, 1e-12))               # (N,)
    gamma = (cond_mean - M[None, :]) / std[:, None]
    ei = std[:, None] * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
    return np.maximum(np.mean(ei, axis=1), 0.)


class _NEIIncrement:
    """NEI against a fixed reference sample, as an acquisition callable ``f(x, gpo)``.

    Module-level so that the greedy driver's acquisition pickles for Dask candidate maps.
    """
    def __init__(self, state, x_out):
        self.state, self.x_out = state, x_out

    def __call__(self, x, gpo):
        return _nei_increment(x, gpo, self.state, self.x_out)


##########################################################################
# Greedy batches: pick the best point, condition on it, repeat
##########################################################################
def _greedy_copy(gpo):
    """A private copy of the GP to add pseudo-points to; the user's GP is never touched.

    fvgp's default prior mean (the mean of ``y_data``) and default noise
    (``(mean|y_data| / 100)^2``) are recomputed from the data on every append, so a
    pseudo-point would shift them and with them the posterior everywhere -- by ~5e-3 with
    8 data points. On the copy both are bound to a snapshot of the original data instead.
    User-supplied mean and noise functions are left alone: they see the pseudo-points,
    which is what conditioning on pending points means for them.

    This writes fvgp internals (``prior.mean_function``, ``prior._default_mean``,
    ``prior.m_n_params``, ``likelihood.noise_function``, ``likelihood.v_n_params``);
    ``test_greedy_copy_freezes_the_data_dependent_defaults`` fails if they change.
    """
    if getattr(gpo, "gp2Scale", False):
        raise NotImplementedError(
            "Greedy batches are not available for gp2Scale GPs: they condition a copy of the "
            "GP on the picks, and a copy has no Dask client. Use batch_mode='joint' or "
            "'independent', or ask for one point at a time.")
    work = copy.deepcopy(gpo)
    snapshot = copy.deepcopy(work.data)
    prior, likelihood = work.prior, work.likelihood
    if prior._default_mean:
        frozen = copy.copy(prior)
        frozen.data = snapshot
        prior.mean_function, prior._default_mean, prior.m_n_params = frozen._default_mean_function, False, 2
    if getattr(likelihood.noise_function, "__func__", None) is type(likelihood)._default_noise_function:
        frozen = copy.copy(likelihood)
        frozen.data = snapshot
        likelihood.noise_function, likelihood.v_n_params = frozen._default_noise_function, 2
    return work


def _pseudo_noise(gpo):
    """The noise variance of a pseudo-point, or None where fvgp decides it.

    fvgp takes noise variances in an update exactly when the GP was built with them. Then
    a pseudo-point gets ``args['batch_pseudo_noise']``, or the mean of the data's noise
    variances: the noise a real measurement would be expected to have. Otherwise the noise
    function, or the frozen default, applies to it as to any data point.
    """
    if gpo.noise_variances is None: return None
    explicit = _acq_arg(gpo, "batch_pseudo_noise", None)
    if explicit is not None: return float(explicit)
    return float(np.nanmean(np.asarray(gpo.noise_variances, dtype=float)))


def _add_pseudo_points(work, x_new, x_out, noise):
    """Append ``x_new`` to ``work`` with the posterior mean as its value (kriging believer).

    The value is the posterior mean in GP space and goes through ``update_gp_data``, not
    ``tell()``, so the Log/Logit transforms are not applied twice; it leaves the posterior
    mean unchanged and shrinks the variance around the new points (a rank-n update, no
    retraining). A multi-task point gets every output -- Kronecker needs the full grid --
    except in 'cartesian' mode, where outputs outside ``x_out`` stay unobserved (NaN).
    """
    k = len(x_new)
    mode = work.multi_task_mode
    if mode is None:
        y = np.asarray(work.posterior_mean(x_new)["m(x)"], dtype=float).reshape(k)
        nv = None if noise is None else np.full(k, noise)
    else:
        outputs = np.arange(work.output_num)
        y = np.asarray(work.posterior_mean(x_new, x_out=outputs)["m(x)"], dtype=float).reshape(k, len(outputs))
        if mode == "cartesian":
            if x_out is not None: y[:, ~np.isin(outputs, x_out)] = np.nan
            nv = None if noise is None else np.where(np.isnan(y), np.nan, noise)
        else:
            nv = None if noise is None else np.full(k, noise)
    work.update_gp_data(x_new, y, noise_variances_new=nv, append=True, rank_n_update=True)


def _index_of(candidates, pick):
    """Position of ``pick`` in a candidate list (the optimizer returns a copy of it)."""
    for i, c in enumerate(candidates):
        try:
            if np.array_equal(np.asarray(c), np.asarray(pick)): return i
        except Exception:  # pragma: no cover - candidates numpy cannot compare
            continue
    return None  # pragma: no cover - the pick always comes from the list


def greedy_batch(gpo, acquisition_function, n, input_set, *, x_out=None, x_pending=None,
                 origin=None, cost_function=None, input_set_dim=None, **optimizer_kwargs):
    """Choose ``n`` points together under any acquisition function, by greedy selection.

    Each step maximizes the single-point acquisition function -- over bounds with the
    usual optimizer (``optimizer_kwargs`` are those of
    :func:`find_acquisition_function_maxima`), or over a candidate list, from which each
    pick is removed -- and then conditions on the pick before the next step:

    * ``noisy expected improvement``: the pick joins the reference set whose sampled
      maximum is the incumbent (:func:`_nei_append`). Exact sequential-greedy q-NEI: the
      per-pick values add up to the batch's q-NEI, and a repeat of a pick is worth 0.
    * everything else, callables included: the pick is added to a private copy of the GP
      (:func:`_greedy_copy`) as a pseudo-point whose value is the posterior mean
      (:func:`_add_pseudo_points`). The posterior mean stays, the variance shrinks around
      the pick, and the acquisition -- evaluated on the copy -- moves on. Hyperparameters
      are not retrained.

    Points in ``x_pending`` (chosen, not yet measured) are conditioned on first. With a
    cost function and an ``origin``, the origin moves to each pick in turn.

    Returns ``(x, f_a(x), None)``: the picks in the order chosen, and each pick's
    acquisition value (divided by the cost, as ``ask()`` reports it) on the conditioned
    GP at the moment it was chosen. A candidate list shorter than ``n`` gives all of it.
    """
    candidates = list(input_set) if isinstance(input_set, list) else None
    nei = isinstance(acquisition_function, str) and acquisition_function == "noisy expected improvement"
    if n > 1 and candidates is None and isinstance(acquisition_function, str) and \
            acquisition_function in ("maximum", "minimum"):
        warnings.warn(f"`{acquisition_function}` depends on the posterior mean only, which "
                      f"pseudo-points do not change, so every greedy step returns the same point. "
                      f"Use `ucb`/`lcb` or `noisy expected improvement` for a batch.")
    euclidean = isinstance(gpo.x_data, np.ndarray)
    as_points = (lambda p: np.atleast_2d(np.asarray(p, dtype=float))) if euclidean else (lambda p: [p])
    has_pending = x_pending is not None and len(x_pending) > 0
    if nei:
        work, state = gpo, _nei_state(gpo, x_out, x_pending=x_pending if has_pending else None)
    else:
        work, noise = _greedy_copy(gpo), _pseudo_noise(gpo)
        if has_pending: _add_pseudo_points(work, x_pending, x_out, noise)
    picks, values = [], []
    for step in range(n):
        if candidates is not None and not candidates: break
        acq = _NEIIncrement(state, x_out) if nei else acquisition_function
        x_t, f_t, _ = find_acquisition_function_maxima(
            work, acq, origin=origin, number_of_maxima_sought=1,
            input_set=candidates if candidates is not None else input_set,
            input_set_dim=input_set_dim, cost_function=cost_function, x_out=x_out,
            **optimizer_kwargs)
        pick = np.asarray(x_t)[0]
        if candidates is not None: pick = candidates.pop(_index_of(candidates, pick))
        picks.append(pick)
        values.append(float(np.asarray(f_t).reshape(-1)[0]))
        logger.debug("greedy batch: pick {} of {}: {}", step + 1, n, pick)
        if step == n - 1: break
        if nei: state = _nei_append(state, gpo, as_points(pick), x_out)
        else: _add_pseudo_points(work, as_points(pick), x_out, noise)
        if cost_function is not None and origin is not None: origin = pick
    return np.asarray(picks), np.asarray(values), None


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
    f0 = np.asarray(func(x)).reshape(-1)[0]
    for i in range(len(x)):
        new_point = np.array(x, dtype=float)
        new_point[i] += epsilon
        grad[i] = (np.asarray(func(new_point)).reshape(-1)[0] - f0) / epsilon
    return grad


def filter_similar_rows(arr, tol=1.):
    rounded = np.round(arr / tol) * tol
    idx = np.unique(rounded, return_index=True, axis=0)[1]
    return np.sort(idx)


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
