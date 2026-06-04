#!/usr/bin/env python

"""Tests for `gpcam` package."""
import unittest
import numpy as np
from gpcam import GPOptimizer
from gpcam import fvGPOptimizer
import time
from gpcam.kernels import *
from dask.distributed import Client
from distributed.utils_test import gen_cluster, client, loop, cluster_fixture, loop_in_thread, cleanup
import copy
N = 20
dim = 2


def ac_func1(x, obj):
    r1 = obj.posterior_mean(x)["m(x)"]
    r2 = obj.posterior_covariance(x)["v(x)"]
    m_index = np.argmin(obj.y_data)
    m = obj.x_data[m_index]
    std_model = np.sqrt(r2)
    return -(r1 + 3.0 * std_model)

def mt_kernel(x1,x2,hps):
    d = get_distance_matrix(x1,x2)
    return np.exp(-d)

def my_noise(x,hps):
    return np.zeros((len(x))) + 0.5


def skernel(x1,x2,hps):
    #The kernel follows the mathematical definition of a kernel. This
    #means there is no limit to the variety of kernels you can define.
    d = get_distance_matrix(x1,x2)
    return hps[0] * matern_kernel_diff1(d,hps[1])

def meanf(x, hps):
    #This is a simple mean function but it can be arbitrarily complex using many hyperparameters.
    return np.sin(hps[2] * x[:,0])

def cost_f(origin, x):
    #module-level so it pickles by reference (used by the serialization test)
    return np.ones(len(x))

#class TestgpCAM(unittest.TestCase):
#    """Tests for `gpcam` package."""

def test_basic_1task(client):
    """Set up test fixtures, if any."""
    x = np.random.rand(N, dim)
    y = np.sin(x[:, 0])
    index_set_bounds = np.array([[0., 1.], [0., 1.]])
    hps_bounds = np.array([[0.001, 1e1], [0.001, 100], [0.001, 100]])
    hps_guess = np.ones((3))
    gp = GPOptimizer(x, y, args={'a': 1.5, 'b':2.})
    gp.tell(x,y)
    gp.train(hyperparameter_bounds=hps_bounds, max_iter = 2)

    gp.get_data()
    gp.evaluate_acquisition_function(np.array([[0.0,0.6],[0.1,0.2]]))
    gp.train(hyperparameter_bounds=hps_bounds, max_iter = 100)
    gp.train(hyperparameter_bounds=hps_bounds, max_iter = 100)
    gp.train(hyperparameter_bounds=hps_bounds, method='global', max_iter = 2)
    gp.train(hyperparameter_bounds=hps_bounds, method='local', max_iter = 2)
    gp.train(hyperparameter_bounds=hps_bounds, method='mcmc', max_iter=3)
    gp.train(hyperparameter_bounds=hps_bounds, method='hgdl', max_iter=3, dask_client=client)

    opt_obj = gp.train(hyperparameter_bounds = hps_bounds, dask_client=client, asynchronous = True, max_iter = 100)
    for i in range(5):
        gp.update_hyperparameters(opt_obj)
        time.sleep(1)
    gp.stop_training(opt_obj)
    print("client", client)
    acquisition_functions = ["variance","relative information entropy","relative information entropy set",
                    "ucb","lcb","maximum","minimum","gradient","expected improvement",
                        "probability of improvement", "target probability", "total correlation"]

    for acq_func in acquisition_functions:
        gp.evaluate_acquisition_function(np.array([[0.0,0.6],[0.1,0.2]]), acquisition_function = acq_func)
    gp.ask(index_set_bounds, max_iter = 2)

def test_basic_multi_task(client):
    """Set up test fixtures, if any."""
    x = np.random.rand(N, dim)
    y = np.zeros((len(x),2))
    y[:,0] = np.sin(x[:, 0])
    y[:,1] = np.sin(x[:, 1])
    index_set_bounds = np.array([[0., 1.], [0., 1.]])
    hps_bounds = np.array([[0.001, 1e9], [0.001, 100], [0.001, 100]])
    hps_guess = np.ones((3))
    gp = fvGPOptimizer(x,y, kernel_function = mt_kernel, init_hyperparameters = np.array([1.,1.,1.]))
    gp.tell(x,y)
    gp.get_data()
    gp.evaluate_acquisition_function(np.array([[0.0,0.6],[0.1,0.2]]), x_out = np.array([0.,1.]))
    gp.train(hyperparameter_bounds=hps_bounds, method='global', max_iter = 2)
    gp.train(hyperparameter_bounds=hps_bounds, method='local', max_iter = 2)
    gp.train(hyperparameter_bounds=hps_bounds, method='mcmc', max_iter=2)
    gp.train(hyperparameter_bounds=hps_bounds, method='hgdl', max_iter=2, dask_client=client)

    opt_obj = gp.train(hyperparameter_bounds=hps_bounds, dask_client=client, asynchronous=True, max_iter = 100)
    for i in range(5):
        gp.update_hyperparameters(opt_obj)
        time.sleep(0.1)
    gp.stop_training(opt_obj)
    acquisition_functions = ["variance","relative information entropy","relative information entropy set","total correlation", "ucb", "expected improvement"]
    for acq_func in acquisition_functions:
        gp.evaluate_acquisition_function(np.array([[0.0,0.6],[0.1,0.2]]), np.array([0,1]), acquisition_function = acq_func)
    gp.ask(index_set_bounds,np.array([0.,1.]), max_iter = 2)
    gp.ask(index_set_bounds, max_iter = 2)

def test_optimizers():
    def f1(x):
        if np.ndim(x) == 1: return (np.sin(5. * x) + np.cos(10. * x) + (2.* (x-0.4)**2) * np.cos(100. * x)), 0.01
        else: return (np.sin(5. * x[:,0]) + np.cos(10. * x[:,0]) + (2.* (x[:,0]-0.4)**2) * np.cos(100. * x[:,0])), np.zeros(len(x)) + 0.01

    def f2(x):
        if np.ndim(x) == 1:
            res = np.array([f1(x)[0], -f1(x)[0]/3.]).reshape(2), np.array([0.01,0.01])
            return res
        else:
            res = np.column_stack([f1(x)[0], -f1(x)[0]/3.]).reshape(len(x),2),\
            np.array([np.zeros(len(x)) + 0.01, np.zeros(len(x)) + 0.01]).reshape(len(x),2)
        return res
    my_gp1 = GPOptimizer()
    result = my_gp1.optimize(func = f1, search_space =  np.array([[0,1]]), max_iter = 10)


    my_gp2 = fvGPOptimizer()
    result = my_gp2.optimize(func = f2, x_out = np.array([0,1]), search_space =  np.array([[0,1]]), max_iter = 10)

def test_acq_funcs(client):
    import numpy as np
    from gpcam.gp_optimizer import GPOptimizer

    #initialize some data
    x_data = np.random.uniform(size = (10,3))
    y_data = np.sin(np.linalg.norm(x_data, axis = 1))


    #initialize the GPOptimizer
    my_gpo = GPOptimizer(x_data, y_data, args = {'a':2.,'b':3.})

    #tell() it some data

    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="relative information entropy set")
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 5, acquisition_function="relative information entropy")
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="relative information entropy")
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="probability of improvement")
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="total correlation")
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="variance")
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="ucb")
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="lcb")
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="maximum")
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="minimum")
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 5, acquisition_function="gradient", method = "local", dask_client=client)
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="gradient", method = "local")
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 5, acquisition_function="variance", method = "hgdl", dask_client=client)
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="target probability", method = "local")

    r = my_gpo.ask([np.array([0.,1.,.5])], n = 1, acquisition_function="target probability", vectorized = False)
    r = my_gpo.ask([np.array([0.,1.,.5])], n = 1, acquisition_function="variance", vectorized = False)
    r = my_gpo.ask([np.array([0.,1.,.5])], n = 1, acquisition_function="ucb", vectorized = False)

def test_pickle():
    import numpy as np
    from gpcam.gp_optimizer import GPOptimizer
    import pickle

    #initialize some data
    x_data = np.random.uniform(size = (10,3))
    y_data = np.sin(np.linalg.norm(x_data, axis = 1))

    #TEST0
    #tests empty gp pickling
    my_gpo = GPOptimizer()
    pickle.loads(pickle.dumps(my_gpo))

    #TEST1
    #initialize the GPOptimizer
    my_gpo = GPOptimizer(x_data, y_data, args = {'a':2.,'b':3.})

    #pickle the GPOptimizer
    stash = pickle.dumps(my_gpo)

    #unpickle the GPOptimizer
    my_gpo2 = pickle.loads(stash)

    r = my_gpo2.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]))

    #assert checks that my_gpo2 is same as my_gpo
    assert np.all(my_gpo.x_data == my_gpo2.x_data)
    assert np.all(my_gpo.y_data == my_gpo2.y_data)
    assert np.all(my_gpo.likelihood.V == my_gpo2.likelihood.V)
    assert np.all(my_gpo.posterior_mean(np.array([[1.,1,1],[2.,2.,2.]]))["m(x)"] == my_gpo2.posterior_mean(np.array([[1,1,1],[2,2,2]]))["m(x)"])
    assert np.all(my_gpo.hyperparameters == my_gpo2.hyperparameters)
    assert np.all(my_gpo.prior.K == my_gpo2.prior.K)

    #TEST2
    #initialize the GPOptimizer
    my_gpo = GPOptimizer(x_data,y_data,
        init_hyperparameters = np.ones((3))/10.,  # We need enough of those for kernel, noise, and prior mean functions
        kernel_function=skernel,
        prior_mean_function=meanf,
        noise_function=my_noise,
        )
    

    #pickle the GPOptimizer
    stash = pickle.dumps(my_gpo)

    #unpickle the GPOptimizer
    my_gpo2 = pickle.loads(stash)

    r = my_gpo2.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]))

    #assert checks that my_gpo2 is same as my_gpo
    assert np.all(my_gpo.x_data == my_gpo2.x_data)
    assert np.all(my_gpo.y_data == my_gpo2.y_data)
    assert np.all(my_gpo.likelihood.V == my_gpo2.likelihood.V)
    assert np.all(my_gpo.posterior_mean(np.array([[1.,1,1],[2.,2.,2.]]))["m(x)"] == my_gpo2.posterior_mean(np.array([[1,1,1],[2,2,2]]))["m(x)"])
    assert np.all(my_gpo.hyperparameters == my_gpo2.hyperparameters)
    assert np.all(my_gpo.prior.K == my_gpo2.prior.K)

    #TEST3
    #initialize the GPOptimizer

    my_gpo = GPOptimizer(x_data,y_data,
        init_hyperparameters = np.ones((4))/10.,  # We need enough of those for kernel, noise, and prior mean functions
        )

    #pickle the GPOptimizer
    stash = pickle.dumps(my_gpo)

    #unpickle the GPOptimizer
    my_gpo2 = pickle.loads(stash)

    r = my_gpo2.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]))

    #assert checks that my_gpo2 is same as my_gpo
    assert np.all(my_gpo.x_data == my_gpo2.x_data)
    assert np.all(my_gpo.y_data == my_gpo2.y_data)
    assert np.all(my_gpo.likelihood.V == my_gpo2.likelihood.V)
    assert np.all(my_gpo.posterior_mean(np.array([[1.,1,1],[2.,2.,2.]]))["m(x)"] == my_gpo2.posterior_mean(np.array([[1,1,1],[2,2,2]]))["m(x)"])
    assert np.all(my_gpo.hyperparameters == my_gpo2.hyperparameters)
    assert np.all(my_gpo.prior.K == my_gpo2.prior.K)

    def is_pickle_equal(obj):
        # Get class and instance attributes before pickling
        cls = type(obj)
        before_class = {k: v for k, v in cls.__dict__.items() if not k.startswith('__')}.keys()
        before_instance = dict(obj.__dict__).keys()

        # Pickle and unpickle
        obj2 = pickle.loads(pickle.dumps(obj))

        # Get attributes after pickling
        cls2 = type(obj2)
        after_class = {k: v for k, v in cls2.__dict__.items() if not k.startswith('__')}.keys()
        after_instance = dict(obj2.__dict__).keys()

        # Compare everything
        if before_class != after_class: print(before_class, after_class)
        if before_instance != after_instance: print(before_instance, after_instance)

        return before_class == after_class and before_instance == after_instance


    my_gpo = GPOptimizer(x_data,y_data,
            init_hyperparameters = np.ones((4))/10.,
            args = {"sfdf": 4.})
    my_gpo.train(max_iter = 100)
    my_gpo.tell(x_data, y_data)


    assert is_pickle_equal(my_gpo)
    assert is_pickle_equal(my_gpo.prior)
    assert is_pickle_equal(my_gpo.likelihood)
    assert is_pickle_equal(my_gpo.marginal_likelihood)
    assert is_pickle_equal(my_gpo.trainer)
    assert is_pickle_equal(my_gpo.posterior)
    assert is_pickle_equal(my_gpo.data)
    assert is_pickle_equal(my_gpo.marginal_likelihood.kv)

    #TEST4
    #gpcam-level config attributes must round-trip by VALUE (not just key presence)
    def cfg_equal(a, b):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(a, b)
        return a is b or a == b

    my_gpo = GPOptimizer(x_data, y_data,
        init_hyperparameters=np.ones((4)) / 10.,
        compute_device="cpu",
        linalg_mode="Chol",
        ram_economy=True,
        gp2Scale_batch_size=5000,
        cost_function=cost_f,
        args={"k": 7.})
    my_gpo2 = pickle.loads(pickle.dumps(my_gpo))
    for attr in ["cost_function", "init_hyperparameters", "compute_device",
                 "kernel_function", "kernel_function_grad",
                 "noise_function", "noise_function_grad",
                 "prior_mean_function", "prior_mean_function_grad",
                 "_gp2Scale", "gp2Scale_batch_size", "_linalg_mode",
                 "ram_economy", "_args", "logging", "multi_task", "x_out", "gp"]:
        assert cfg_equal(getattr(my_gpo, attr), getattr(my_gpo2, attr)), attr
    assert my_gpo2._dask_client is None

    #TEST5
    #multi-task (fvGPOptimizer) pickling: exercises multi_task=True and x_out
    x_mt = np.random.uniform(size=(10, 2))
    y_mt = np.column_stack([np.sin(x_mt[:, 0]), np.cos(x_mt[:, 1])])
    fv = fvGPOptimizer(x_mt, y_mt, kernel_function=mt_kernel,
                       init_hyperparameters=np.array([1., 1., 1.]))
    fv2 = pickle.loads(pickle.dumps(fv))
    assert fv2.multi_task is True
    assert np.array_equal(fv.x_out, fv2.x_out)
    assert np.all(fv.x_data == fv2.x_data)
    assert np.all(fv.y_data == fv2.y_data)
    assert np.all(fv.hyperparameters == fv2.hyperparameters)
    assert is_pickle_equal(fv)


def test_transformed_gp():
    import numpy as np
    import pickle
    import warnings as _warnings
    from scipy.special import expit
    from gpcam import GPOptimizer, LogGPOptimizer, LogitGPOptimizer

    np.random.seed(42)
    x = np.random.uniform(0, 1, size=(20, 2))
    xp = np.array([[0.3, 0.4], [0.7, 0.2]])

    # ---- LogGPOptimizer: (0, inf) ----
    y_pos = np.exp(np.sin(x[:, 0])) + 0.1
    log_gp = LogGPOptimizer(x, y_pos)
    log_gp.tell(x, y_pos)
    # round-trip
    assert np.allclose(log_gp._inverse(log_gp._forward(y_pos)), y_pos)
    # evaluate_posterior shape/keys/ordering
    ep = log_gp.evaluate_posterior(xp)
    assert set(ep.keys()) == {"median", "mean", "std", "lower", "upper", "level"}
    assert np.all(ep["median"] > 0) and np.all(ep["lower"] > 0) and np.all(ep["upper"] > 0)
    assert np.all(ep["lower"] < ep["median"]) and np.all(ep["median"] < ep["upper"])
    # median = exp(mu); mean matches the lognormal closed form
    mu = log_gp.posterior_mean(xp)["m(x)"]
    var = log_gp.posterior_covariance(xp, variance_only=True)["v(x)"]
    assert np.allclose(ep["median"], np.exp(mu))
    assert np.allclose(ep["mean"], np.exp(mu + var / 2.0))
    # domain validation
    try:
        LogGPOptimizer(x, np.array([1.0] * 19 + [0.0]))
        raise AssertionError("LogGPOptimizer should reject y <= 0")
    except ValueError:
        pass

    # ---- LogitGPOptimizer: [0, 1] with boundary clipping ----
    y_logit = np.clip(0.5 + 0.4 * np.sin(x[:, 0]), 0.0, 1.0)
    y_logit[0], y_logit[1] = 0.0, 1.0  # force clipping
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        logit_gp = LogitGPOptimizer(x, y_logit, n_samples=2000)
        logit_gp.tell(x, y_logit)
    assert any("clipped" in str(wi.message) for wi in w)
    ep = logit_gp.evaluate_posterior(xp)
    assert np.all((ep["lower"] > 0) & (ep["upper"] < 1))
    assert np.all((ep["median"] > 0) & (ep["median"] < 1))
    assert np.all(ep["lower"] < ep["upper"])
    # median = sigmoid(mu); MC moments finite
    mu = logit_gp.posterior_mean(xp)["m(x)"]
    assert np.allclose(ep["median"], expit(mu))
    assert np.all(np.isfinite(ep["mean"])) and np.all(np.isfinite(ep["std"]))

    # ---- Identity hooks: GPOptimizer.evaluate_posterior bundles the Gaussian ----
    g = GPOptimizer(x, np.sin(x[:, 0]))
    g.tell(x, np.sin(x[:, 0]))
    ep = g.evaluate_posterior(xp)
    mu_id = g.posterior_mean(xp)["m(x)"]
    assert np.allclose(ep["median"], mu_id) and np.allclose(ep["mean"], mu_id)

    # ---- return_samples=True: shape, finiteness, and per-class distribution support ----
    n_pts = xp.shape[0]
    n_samp = 4000
    # identity: samples are real-valued Gaussians
    ep_id = g.evaluate_posterior(xp, return_samples=True, n_samples=n_samp)
    assert ep_id["samples"].shape == (n_pts, n_samp)
    assert np.all(np.isfinite(ep_id["samples"]))
    # log: samples are strictly positive (lognormal)
    ep_log_s = log_gp.evaluate_posterior(xp, return_samples=True, n_samples=n_samp)
    assert ep_log_s["samples"].shape == (n_pts, n_samp)
    assert np.all(ep_log_s["samples"] > 0)
    # logit: samples are strictly inside (0, 1) (logistic-normal)
    ep_logit_s = logit_gp.evaluate_posterior(xp, return_samples=True, n_samples=n_samp)
    assert ep_logit_s["samples"].shape == (n_pts, n_samp)
    assert np.all((ep_logit_s["samples"] > 0) & (ep_logit_s["samples"] < 1))
    # sample mean approximates the reported mean (loose tolerance: 3 std-error)
    assert np.all(np.abs(ep_logit_s["samples"].mean(axis=1) - ep_logit_s["mean"])
                  < 3 * ep_logit_s["std"] / np.sqrt(n_samp))

    # ---- Single-point query returns 1-d arrays (regression: posterior_mean used to scalarize) ----
    ep_single = log_gp.evaluate_posterior(np.array([[0.5, 0.5]]), return_samples=True, n_samples=500)
    assert ep_single["median"].shape == (1,)
    assert ep_single["lower"].shape == (1,) and ep_single["upper"].shape == (1,)
    assert ep_single["samples"].shape == (1, 500)
    _ = ep_single["median"][0]  # must be indexable

    # ---- Pickling: data + extra Logit attrs survive ----
    log_gp2 = pickle.loads(pickle.dumps(log_gp))
    assert np.allclose(log_gp.y_data, log_gp2.y_data)
    assert np.allclose(log_gp.evaluate_posterior(xp)["median"],
                       log_gp2.evaluate_posterior(xp)["median"])
    logit_gp2 = pickle.loads(pickle.dumps(logit_gp))
    assert logit_gp2.eps == logit_gp.eps
    assert logit_gp2.n_samples == logit_gp.n_samples






