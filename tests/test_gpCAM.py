#!/usr/bin/env python

"""Tests for `gpcam` package."""
import unittest
import numpy as np
from gpcam import AutonomousExperimenterGP
from gpcam import AutonomousExperimenterFvGP
from gpcam import GPOptimizer
from gpcam import fvGPOptimizer
import time

def ac_func1(x, obj):
    r1 = obj.posterior_mean(x)["f(x)"]
    r2 = obj.posterior_covariance(x)["v(x)"]
    m_index = np.argmin(obj.y_data)
    m = obj.x_data[m_index]
    std_model = np.sqrt(r2)
    return -(r1 + 3.0 * std_model)

def instrument(data, instrument_dict=None):
    for entry in data:
        entry["y_data"] = np.sin(np.linalg.norm(entry["x_data"]))
    return data
def instrument2(data, instrument_dict=None):
    for entry in data:
        entry["y_data"] = np.array([np.sin(np.linalg.norm(entry["x_data"])), 10. * np.sin(np.linalg.norm(entry["x_data"]))])
        entry["output positions"] = np.array([[0],[1]])
    return data

def mt_kernel(x1,x2,hps,obj):
    d = obj.get_distance_matrix(x1,x2)
    return np.exp(-d)



class TestgpCAM(unittest.TestCase):
    """Tests for `gpcam` package."""

    def test_basic_1task(self, dim=2, N=20):
        """Set up test fixtures, if any."""
        x = np.random.rand(N, dim)
        y = np.sin(x[:, 0])
        index_set_bounds = np.array([[0., 1.], [0., 1.]])
        hps_bounds = np.array([[0.001, 1e1], [0.001, 100], [0.001, 100]])
        hps_guess = np.ones((3))
        ###################################################################################
        gp = GPOptimizer(x,y)
        gp.tell(x,y,noise_variances = np.ones(y.shape))
        gp.train(hyperparameter_bounds=hps_bounds, max_iter = 2)

        gp.get_data()
        gp.evaluate_acquisition_function(np.array([[0.0,0.6],[0.1,0.2]]))
        gp.train(hyperparameter_bounds=hps_bounds)
        gp.train(hyperparameter_bounds=hps_bounds)
        gp.train(hyperparameter_bounds=hps_bounds, method='global', max_iter = 2)
        gp.train(hyperparameter_bounds=hps_bounds, method='local', max_iter = 2)
        gp.train(hyperparameter_bounds=hps_bounds, method='mcmc', max_iter=3)
        gp.train(hyperparameter_bounds=hps_bounds, method='hgdl', max_iter=3)

        opt_obj = gp.train_async(hyperparameter_bounds = hps_bounds)
        for i in range(5):
            gp.update_hyperparameters(opt_obj)
            time.sleep(1)
        gp.stop_training(opt_obj)
        acquisition_functions = ["variance","relative information entropy","relative information entropy set",
                        "ucb","lcb","maximum","minimum","gradient","expected improvement",
                         "probability of improvement", "target probability", "total correlation"]
        gp.args = args={'a': 1.5, 'b':2.}
        for acq_func in acquisition_functions:
            gp.evaluate_acquisition_function(np.array([[0.0,0.6],[0.1,0.2]]), acquisition_function = acq_func)
        gp.ask(index_set_bounds, max_iter = 2)

    def test_basic_multi_task(self, dim=2, N=20):
        """Set up test fixtures, if any."""
        x = np.random.rand(N, dim)
        y = np.zeros((len(x),2))
        y[:,0] = np.sin(x[:, 0])
        y[:,1] = np.sin(x[:, 1])
        index_set_bounds = np.array([[0., 1.], [0., 1.]])
        hps_bounds = np.array([[0.001, 1e9], [0.001, 100], [0.001, 100]])
        hps_guess = np.ones((3))
        ###################################################################################
        gp = fvGPOptimizer(x,y, gp_kernel_function = mt_kernel, init_hyperparameters = np.array([1.,1.,1.]), hyperparameter_bounds = hps_bounds)
        gp.tell(x,y,noise_variances = np.ones(y.shape))
        gp.get_data()
        gp.evaluate_acquisition_function(np.array([[0.0,0.6],[0.1,0.2]]), x_out = np.array([[0.],[1.]]))
        gp.train(hyperparameter_bounds=hps_bounds, method='global', max_iter = 2)
        gp.train(hyperparameter_bounds=hps_bounds, method='local', max_iter = 2)
        gp.train(hyperparameter_bounds=hps_bounds, method='mcmc', max_iter=2)
        gp.train(hyperparameter_bounds=hps_bounds, method='hgdl', max_iter=2)

        opt_obj = gp.train_async(hyperparameter_bounds=hps_bounds)
        for i in range(5):
            gp.update_hyperparameters(opt_obj)
            time.sleep(1)
        gp.stop_training(opt_obj)
        acquisition_functions = ["variance","relative information entropy","relative information entropy set","total correlation"]
        for acq_func in acquisition_functions:
            gp.evaluate_acquisition_function(np.array([[0.0,0.6],[0.1,0.2]]), np.array([[0.],[1.]]), acquisition_function = acq_func)
        gp.ask(index_set_bounds,np.array([[0.],[1.]]), max_iter = 2)

    def test_ae(self):
        ##set up your parameter space
        input_space = np.array([[3.0,45.8],
                                [4.0,47.0]])

        ##set up some hyperparameters, if you have no idea, set them to 1 and make the training bounds large
        init_hyperparameters = np.array([1,1,1])
        hps_bounds =  np.array([[0.01,100],[0.01,100.0],[0.01,100]])

        ##let's initialize the autonomous experimenter ...
        my_ae = AutonomousExperimenterGP(input_space, hyperparameters=init_hyperparameters,
                                        hyperparameter_bounds=hps_bounds,instrument_function = instrument,
                                        init_dataset_size=10)
        #...train...
        my_ae.data.inject_dataset(my_ae.data.dataset)
        my_ae.data.inject_arrays(np.array([[0.,0.1],[1.,1.]]), y = np.array([3.,4.]), v = np.array([.1,.2]), info = [{"f": 2.}, {'d':3.}])
        my_ae = AutonomousExperimenterGP(input_space, hyperparameters=init_hyperparameters,
                                        hyperparameter_bounds=hps_bounds,instrument_function = instrument,
                                        init_dataset_size=4)

        my_ae.train()
        my_ae.train_async()
        my_ae.update_hps()
        my_ae.kill_training()

        #...and run. That's it. You successfully executed an autonomous experiment.
        my_ae.go(N = 20)


    def test_fvae(self):
        ##set up your parameter space
        input_space = np.array([[3.0,45.8],
                                [4.0,47.0]])

        ##set up some hyperparameters, if you have no idea, set them to 1 and make the training bounds large
        init_hyperparameters = np.array([1,1,1])
        hps_bounds =  np.array([[0.01,100],[0.01,100.0],[0.01,100]])

        ##let's initialize the autonomous experimenter ...
        my_ae = AutonomousExperimenterFvGP(input_space, 2, hyperparameters=init_hyperparameters, kernel_function = mt_kernel,
                                        hyperparameter_bounds=hps_bounds,instrument_function = instrument2,
                                        init_dataset_size=10)
        #...train...
        my_ae.data.inject_dataset(my_ae.data.dataset)
        my_ae.data.inject_arrays(np.array([[0.,0.1],[1.,1.]]), y = np.array([[3.,4.],[5.,9.]]), v = np.array([[.1,.2],[0.01,0.03]]), vp = np.array([[[2.0,3.0]],[[1.,2.]]]),info = [{"f": 2.}, {'d':3.}])
        my_ae = AutonomousExperimenterFvGP(input_space,2, hyperparameters=init_hyperparameters, kernel_function = mt_kernel,
                                        hyperparameter_bounds=hps_bounds,instrument_function = instrument2,
                                        init_dataset_size=4)

        my_ae.train()
        my_ae.train_async()
        my_ae.update_hps()
        my_ae.kill_training()

        #...and run. That's it. You successfully executed an autonomous experiment.
        my_ae.go(N = 10)



    def test_acq_funcs(self):
        import numpy as np
        from gpcam.gp_optimizer import GPOptimizer

        #initialize some data
        x_data = np.random.uniform(size = (10,3))
        y_data = np.sin(np.linalg.norm(x_data, axis = 1))


        #initialize the GPOptimizer
        my_gpo = GPOptimizer(x_data, y_data)
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
        r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 5, acquisition_function="gradient", method = "local")
        r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 5, acquisition_function="gradient", method = "hgdl")
        my_gpo.args = {'a':2.,'b':3.}
        r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="target probability", method = "local")


