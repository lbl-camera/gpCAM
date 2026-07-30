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
            # observation. `np.max(gpo.y_data)` is a single (point, task) entry and is not
            # comparable with a sum over tasks.
            last_best = np.max(_observed_task_sums(gpo, x_out))
            return _expected_improvement(m - last_best, std)
        elif acquisition_function == "knowledge gradient":
            return knowledge_gradient(x, gpo, x_out=x_out)
        elif acquisition_function == "noisy expected improvement":
            return noisy_expected_improvement(x, gpo, x_out=x_out)
        else:
            raise Exception("No valid acquisition function string provided. Choose from ", all_acq_func)


def _observed_task_sums(gpo, x_out):
    """Per-data-point observations summed over the tasks named by ``x_out``.

    This is the incumbent scalarization for multi-task ``expected improvement``: it has
    to match how the posterior mean is scalarized, otherwise the "improvement" compares
    a sum over tasks against something that is not one.

    ``gpo.fvgp_y_data`` is the original ``(N, No)`` observation array. (``gpo.y_data``
    is the flattened product-space vector in task-major order, so neither its ``max``
    nor a ``reshape(-1, No)`` of it gives per-point task sums.)

    ``x_out`` holds output-space coordinates; in the usual discrete-task setting these
    are the integer task indices, so the matching columns can be selected when only a
    subset of tasks is being asked about. For genuinely continuous (function-valued)
    output coordinates there are no columns to select and every task is summed -- which
    is the same thing whenever ``x_out`` is the full task set, the default.
    """
    y = np.asarray(gpo.fvgp_y_data, dtype=float)
    idx = np.asarray(x_out)
    if idx.ndim == 1 and np.all(np.isfinite(idx)) and np.all(idx == np.round(idx)):
        idx = np.round(idx).astype(int)
        if idx.min() >= 0 and idx.max() < y.shape[1]:
            y = y[:, idx]
    return np.sum(y, axis=1)


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
    """Input-space data points used as the reference/fantasy set.

    For multi-task GPs ``gpo.x_data`` lives in the product (input x task) space, so
    the unique input points come from ``get_data()['x data']``. Capped (subsampled)
    for scalability; the subsample is stable within one ``ask()`` because the rng is
    seeded.
    """
    if x_out is None:
        xr = np.asarray(gpo.x_data, dtype=float)
    else:
        xr = np.asarray(gpo.get_data()["x data"], dtype=float)
    if len(xr) > cap:
        xr = xr[rng.choice(len(xr), size=cap, replace=False)]
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
    ``Var(sum_t eps(x, t))``. With a full covariance that is the sum of each point's
    (T, T) noise block, which accounts for cross-task noise correlation exactly; the
    product-space ordering is task-major (k = point + Npts*task), as in
    :func:`_scalarized_blocks`. With only per-observation variances available the
    tasks have to be assumed uncorrelated, giving ``len(x_out)`` times the mean.
    """
    base = None
    task_summed = False          # True once `base` is already Var(sum over tasks)
    try:
        v = np.asarray(gpo.get_data()["measurement variances"], dtype=float)
        if v.size and v.ndim == 2 and v.shape[0] == v.shape[1]:
            T = len(x_out) if x_out is not None else 0
            if T > 0 and v.shape[0] % T == 0:
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
    S_flat = np.asarray(gpo.posterior_covariance(stack, x_out=x_out)["S_flat"])
    if x_out is None:
        mu = mean.reshape(-1)
        cov = S_flat
    else:
        # Flat product-space ordering is task-major (k = point + Npts*task), so the
        # (Npts*T, Npts*T) covariance is a T x T grid of (Npts, Npts) blocks; summing
        # the task blocks gives Cov(sum_t f(.,t), sum_t' f(.,t')). See the v.reshape(
        # ..., order='F') convention in fvgp.posterior_covariance.
        T, P = len(x_out), M + N
        mu = mean.reshape(P, T).sum(axis=1)
        cov = S_flat.reshape(T, P, T, P).sum(axis=(0, 2))
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

    Optional ``gpo.args`` keys: ``nei_samples`` (default 128),
    ``nei_reference_set_size`` (default 100), ``nei_seed`` (default 0).
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n_samples = int(_acq_arg(gpo, "nei_samples", 128))
    cap = int(_acq_arg(gpo, "nei_reference_set_size", 100))
    rng = np.random.default_rng(int(_acq_arg(gpo, "nei_seed", 0)))
    x_ref = _reference_points(gpo, x_out, cap, rng)
    mu_ref, cov_ref, mu_cand, var_cand, _cross = _scalarized_blocks(gpo, x_ref, x, x_out)
    L = _jitter_cholesky(cov_ref)
    Z = rng.standard_normal((len(mu_ref), n_samples))
    f_ref = mu_ref[:, None] + L @ Z                          # (M, K) incumbent samples
    y_star = np.max(f_ref, axis=0)                           # (K,) sampled incumbents
    std = np.sqrt(np.maximum(var_cand, 1e-12))               # (N,)
    d = mu_cand[:, None] - y_star[None, :]                   # (N, K)
    gamma = d / std[:, None]
    ei = std[:, None] * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
    return np.mean(ei, axis=1)


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
