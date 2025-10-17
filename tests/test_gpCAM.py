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
    gp.train(hyperparameter_bounds=hps_bounds)
    gp.train(hyperparameter_bounds=hps_bounds)
    gp.train(hyperparameter_bounds=hps_bounds, method='global', max_iter = 2)
    gp.train(hyperparameter_bounds=hps_bounds, method='local', max_iter = 2)
    gp.train(hyperparameter_bounds=hps_bounds, method='mcmc', max_iter=3)
    gp.train(hyperparameter_bounds=hps_bounds, method='hgdl', max_iter=3, dask_client=client)

    opt_obj = gp.train_async(hyperparameter_bounds = hps_bounds, dask_client=client)
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

    opt_obj = gp.train_async(hyperparameter_bounds=hps_bounds, dask_client=client)
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
    my_gpo.train()
    my_gpo.tell(x_data, y_data)


    assert is_pickle_equal(my_gpo)
    assert is_pickle_equal(my_gpo.prior)
    assert is_pickle_equal(my_gpo.likelihood)
    assert is_pickle_equal(my_gpo.marginal_density)
    assert is_pickle_equal(my_gpo.trainer)
    assert is_pickle_equal(my_gpo.posterior)
    assert is_pickle_equal(my_gpo.data)
    assert is_pickle_equal(my_gpo.marginal_density.KVlinalg)






