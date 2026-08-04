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
    acquisition_functions = ["variance","relative information entropy","relative information entropy set","total correlation", "ucb", "expected improvement", "knowledge gradient", "noisy expected improvement"]
    for acq_func in acquisition_functions:
        gp.evaluate_acquisition_function(np.array([[0.0,0.6],[0.1,0.2]]), np.array([0,1]), acquisition_function = acq_func)
    gp.ask(index_set_bounds, np.array([0,1]), acquisition_function="knowledge gradient", max_iter = 2)
    gp.ask(index_set_bounds, np.array([0,1]), acquisition_function="noisy expected improvement", max_iter = 2)
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
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="knowledge gradient")
    r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="noisy expected improvement")

    r = my_gpo.ask([np.array([0.,1.,.5])], n = 1, acquisition_function="target probability", vectorized = False)
    r = my_gpo.ask([np.array([0.,1.,.5])], n = 1, acquisition_function="variance", vectorized = False)
    r = my_gpo.ask([np.array([0.,1.,.5])], n = 1, acquisition_function="ucb", vectorized = False)

    # knowledge gradient and noisy expected improvement return one non-negative
    # score per candidate (like expected improvement)
    grid = np.random.uniform(size=(12, 3))
    for acq in ["knowledge gradient", "noisy expected improvement"]:
        v = my_gpo.evaluate_acquisition_function(grid, acquisition_function=acq)
        assert v.shape == (12,)
        assert np.all(np.isfinite(v)) and np.all(v >= -1e-9)
    r = my_gpo.ask([np.array([0.,1.,.5]), np.array([0.5,0.5,0.5])], n = 1,
                   acquisition_function="knowledge gradient", vectorized = False)
    r = my_gpo.ask([np.array([0.,1.,.5]), np.array([0.5,0.5,0.5])], n = 1,
                   acquisition_function="noisy expected improvement", vectorized = False)

def test_expected_improvement_not_degenerate():
    """Regression test for issue #55.

    The improvement must be standardized *before* it is clipped. Clipping first pins
    z to 0 at every candidate below the incumbent, so EI collapses to the constant
    std * norm.pdf(0) = 0.3989 * std -- i.e. a scaled `variance` acquisition that
    carries no information about the posterior mean.
    """
    from scipy.stats import norm

    x_data = np.random.uniform(size=(15, 2))
    y_data = np.sin(np.linalg.norm(x_data, axis=1))
    gpo = GPOptimizer(x_data, y_data)

    grid = np.random.uniform(size=(200, 2))
    ei = gpo.evaluate_acquisition_function(grid, acquisition_function="expected improvement")
    m = gpo.posterior_mean(grid)["m(x)"].reshape(len(grid))
    std = np.sqrt(gpo.posterior_covariance(grid, variance_only=True)["v(x)"]).reshape(len(grid))
    incumbent = np.max(gpo.y_data)

    assert ei.shape == (len(grid),)
    assert np.all(np.isfinite(ei)) and np.all(ei >= 0.)

    below = m < incumbent
    assert below.sum() > 10, "test needs candidates below the incumbent to be meaningful"
    ratio = ei[below] / std[below]
    # the bug made this ratio exactly norm.pdf(0) at every sub-incumbent candidate
    assert len(np.unique(np.round(ratio, 8))) > 1
    assert ratio.max() - ratio.min() > 0.05
    assert np.all(ratio < norm.pdf(0.))

    # ei/std is a monotone function of z = (m - incumbent)/std, which is what the clip
    # destroyed: it pinned z to 0 and made the ratio constant. (Monotone in z, not in m
    # -- candidates with equal improvement but different std have different z.)
    z = (m - incumbent) / (std + 1e-9)
    ratio_by_z = (ei / std)[np.argsort(z)]
    assert np.all(np.diff(ratio_by_z) > -1e-12)

    # matches the textbook closed form
    expected = (m - incumbent) * norm.cdf(z) + std * norm.pdf(z)
    assert np.allclose(ei, np.maximum(expected, 0.), atol=1e-12)


def test_expected_improvement_closed_form():
    """The EI helper, on a controlled grid where std is fixed (issue #55)."""
    from scipy.stats import norm
    from gpcam.surrogate_model import _expected_improvement

    imp = np.linspace(-5., 5., 1001)
    ei = _expected_improvement(imp, np.ones_like(imp))
    # strictly increasing in the improvement; the clipped version was flat for imp < 0
    assert np.all(np.diff(ei) > 0.)
    assert np.all(ei > 0.)
    # at imp == 0 the value is exactly std * phi(0)
    assert np.isclose(_expected_improvement(np.zeros(1), np.ones(1))[0], norm.pdf(0.))
    # deep below the incumbent EI vanishes rather than plateauing at 0.3989 * std
    assert _expected_improvement(np.array([-50.]), np.ones(1))[0] < 1e-12
    # for a near-deterministic candidate EI approaches the improvement itself
    assert np.isclose(_expected_improvement(np.array([3.]), np.array([1e-9]))[0], 3., atol=1e-6)
    # non-negative and finite for degenerate/extreme inputs
    for i, s in [(-1e6, 1e-12), (-1e3, 1e-3), (0., 0.), (1e3, 0.), (-40., 1.)]:
        v = _expected_improvement(np.array([i]), np.array([s]))[0]
        assert np.isfinite(v) and v >= 0.


def test_expected_improvement_multi_task_scalarization():
    """Issue #55, multi-task branch: the incumbent must use the same scalarization as
    the posterior mean (a sum over tasks), and the spread must be the std of that sum
    -- not the sum of the per-task stds."""
    from scipy.stats import norm

    x_data = np.random.uniform(size=(12, 2))
    # deliberately mismatched task scales: max(y_data) over the flattened product space
    # is dominated by task 1 and is not comparable to a sum over tasks
    y_data = np.column_stack([np.sin(np.linalg.norm(x_data, axis=1)),
                              100. * np.cos(np.linalg.norm(x_data, axis=1))])
    gpo = fvGPOptimizer(x_data, y_data)
    x_out = np.array([0, 1])

    grid = np.random.uniform(size=(25, 2))
    ei = gpo.evaluate_acquisition_function(grid, x_out=x_out,
                                           acquisition_function="expected improvement")
    assert ei.shape == (len(grid),)
    assert np.all(np.isfinite(ei)) and np.all(ei >= 0.)

    m = np.sum(gpo.posterior_mean(grid, x_out=x_out)["m(x)"].reshape(len(grid), len(x_out)), axis=1)
    v = gpo.posterior_covariance(grid, x_out=x_out, variance_only=True)["v(x)"].reshape(len(grid), len(x_out))
    std = np.sqrt(np.sum(v, axis=1))
    incumbent = np.max(np.sum(gpo.fvgp_y_data, axis=1))
    z = (m - incumbent) / (std + 1e-9)
    expected = (m - incumbent) * norm.cdf(z) + std * norm.pdf(z)
    assert np.allclose(ei, np.maximum(expected, 0.), atol=1e-12)

    # the incumbent is the best task-summed observation, not the largest single entry
    # of the task-major product-space vector
    assert incumbent != np.max(gpo.y_data)
    # ...and not what a naive reshape of the product-space vector would give
    naive = np.max(np.sum(np.asarray(gpo.y_data).reshape(-1, len(x_out)), axis=1))
    assert not np.isclose(incumbent, naive)


def test_assumed_observation_noise_matrix_valued():
    """fvgp accepts a full (N, N) noise covariance from a noise_function, not just a
    1-d vector of variances. The per-observation variances are then its diagonal --
    averaging the whole matrix divides by N. For the multi-task task-sum, a full
    covariance gives Var(sum_t eps) exactly, including cross-task correlation."""
    from gpcam.surrogate_model import _assumed_observation_noise

    sig2, rho, N = 0.04, 0.6, 30

    def noise_vector(x, hps):
        return np.full(len(x), sig2)

    def noise_diag_matrix(x, hps):
        return np.diag(np.full(len(x), sig2))

    def noise_correlated(x, hps):
        c = np.full((len(x), len(x)), rho * sig2)
        np.fill_diagonal(c, sig2)
        return c

    x = np.random.uniform(size=(N, 2))
    y = np.sin(np.linalg.norm(x, axis=1))
    for f in (noise_vector, noise_diag_matrix, noise_correlated):
        gpo = GPOptimizer(x, y, noise_function=f, init_hyperparameters=np.ones(3))
        assert np.isclose(_assumed_observation_noise(gpo, None), sig2), f.__name__

    # multi-task: T tasks, Var(eps_0 + eps_1) = 2*sig2 (+ 2*rho*sig2 if correlated)
    xm = np.random.uniform(size=(15, 2))
    ym = np.column_stack([np.sin(xm[:, 0]), np.cos(xm[:, 1])])
    x_out = np.array([0, 1])
    for f, exact in ((noise_vector, 2 * sig2),
                     (noise_diag_matrix, 2 * sig2),
                     (noise_correlated, 2 * sig2 + 2 * rho * sig2)):
        gpo = fvGPOptimizer(xm, ym, noise_function=f, init_hyperparameters=np.ones(4))
        assert np.isclose(_assumed_observation_noise(gpo, x_out), exact), f.__name__

    # degenerate input still yields a usable positive scalar
    gpo = GPOptimizer(x, y)
    assert _assumed_observation_noise(gpo, None) > 0.


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

    # ---- LogitGPOptimizer with custom range=[a, b] ----
    a, b = 2.0, 5.0
    # data uniformly in (a, b) with a boundary value to exercise clipping
    y_range = np.linspace(a + 0.1, b - 0.1, 20)
    y_range[0] = a              # boundary -> should be clipped after normalization
    y_range[-1] = b
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        bounded_gp = LogitGPOptimizer(x, y_range, range=(a, b), n_samples=2000)
        bounded_gp.tell(x, y_range)
    assert any("clipped" in str(wi.message) for wi in w)
    # round-trip on interior points
    interior = np.linspace(a + 0.05, b - 0.05, 7)
    assert np.allclose(bounded_gp._inverse(bounded_gp._forward(bounded_gp._prepare(interior))),
                       interior, atol=1e-9)
    # evaluate_posterior outputs all live inside (a, b)
    ep_b = bounded_gp.evaluate_posterior(xp, return_samples=True, n_samples=2000)
    assert np.all((ep_b["lower"] > a) & (ep_b["upper"] < b))
    assert np.all((ep_b["median"] > a) & (ep_b["median"] < b))
    assert np.all((ep_b["samples"] > a) & (ep_b["samples"] < b))
    # invalid range raises
    try:
        LogitGPOptimizer(x, y_range, range=(5.0, 2.0))
        raise AssertionError("LogitGPOptimizer should reject range with lower >= upper")
    except ValueError:
        pass
    # range survives pickling
    bounded_gp2 = pickle.loads(pickle.dumps(bounded_gp))
    assert bounded_gp2.range == bounded_gp.range == (a, b)
    # default range=(0, 1) is unchanged
    default_gp = LogitGPOptimizer(x, np.clip(0.5 + 0.3 * np.sin(x[:, 0]), 0.0, 1.0), n_samples=1000)
    assert default_gp.range == (0.0, 1.0)








def test_candidate_batch_is_jointly_optimized_for_set_valued_acquisitions():
    """With a candidate list and n>1, a set-valued acquisition must choose a batch that
    is good *as a batch*.

    Ranking candidates individually and taking the top n says nothing about whether the
    batch is any good together, and the best scorers routinely land on top of each
    other. Only `total correlation` and `relative information entropy` score a whole set
    with one number, so only they can express this.
    """
    import warnings
    from gpcam.surrogate_model import SET_VALUED_ACQUISITION_FUNCTIONS

    np.random.seed(0)
    x_data = np.random.rand(40, 2)
    y_data = np.sin(np.linalg.norm(x_data, axis=1))
    gpo = GPOptimizer(x_data, y_data)
    candidates = [np.random.rand(2) for _ in range(200)]

    def closest_pair(points):
        pts = np.asarray(points)
        return float(np.min([np.linalg.norm(a - b)
                             for i, a in enumerate(pts) for b in pts[i + 1:]]))

    def set_value(points):
        return float(gpo.evaluate_acquisition_function(
            np.asarray(points), acquisition_function="total correlation")[0])

    joint = gpo.ask(candidates, n=5, acquisition_function="total correlation")
    assert np.asarray(joint["x"]).shape == (5, 2)
    # one number for the batch, and it really is the batch's value
    assert np.asarray(joint["f_a(x)"]).shape == (1,)
    assert np.isclose(joint["f_a(x)"][0], set_value(joint["x"]))

    # the independently-ranked batch is worse as a set, and far more clustered
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        independent = gpo.ask(candidates, n=5, acquisition_function="variance",
                              vectorized=False)
    assert set_value(joint["x"]) > set_value(independent["x"])
    assert closest_pair(joint["x"]) > 5.0 * closest_pair(independent["x"])

    # both set-valued acquisitions take this path
    for acq in sorted(SET_VALUED_ACQUISITION_FUNCTIONS):
        r = gpo.ask(candidates[:60], n=4, acquisition_function=acq)
        assert np.asarray(r["x"]).shape == (4, 2)
        assert np.asarray(r["f_a(x)"]).shape == (1,)


def test_candidate_batch_warns_when_it_cannot_be_jointly_optimized():
    """A point-wise acquisition cannot express whether a batch is jointly good, so
    ask() must say so rather than implying the n points were chosen together."""
    import warnings
    np.random.seed(1)
    x_data = np.random.rand(25, 2)
    y_data = np.sin(np.linalg.norm(x_data, axis=1))
    gpo = GPOptimizer(x_data, y_data)
    candidates = [np.random.rand(2) for _ in range(50)]

    def messages(**kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gpo.ask(candidates, **kwargs)
        return [str(w.message) for w in caught]

    assert any("not mutually optimal" in m for m in
               messages(n=5, acquisition_function="variance", vectorized=False))

    # a user callable cannot be introspected, so it is warned about too
    def my_acq(x, gp_obj):
        return gp_obj.posterior_covariance(x, variance_only=True)["v(x)"]
    assert any("user-supplied acquisition function" in m for m in
               messages(n=3, acquisition_function=my_acq, vectorized=False))

    # a single point is not a batch, so no warning either way
    assert not any("not mutually optimal" in m for m in
                   messages(n=1, acquisition_function="variance", vectorized=False))
    assert not any("not mutually optimal" in m for m in
                   messages(n=5, acquisition_function="total correlation"))


def test_candidate_batch_handles_small_pools():
    """Asking for more points than exist must return the pool, not fail."""
    np.random.seed(2)
    x_data = np.random.rand(20, 2)
    y_data = np.sin(np.linalg.norm(x_data, axis=1))
    gpo = GPOptimizer(x_data, y_data)
    candidates = [np.random.rand(2) for _ in range(3)]
    r = gpo.ask(candidates, n=10, acquisition_function="total correlation")
    assert np.asarray(r["x"]).shape == (3, 2)
    assert np.asarray(r["f_a(x)"]).shape == (1,)
