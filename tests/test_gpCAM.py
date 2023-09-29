#!/usr/bin/env python

"""Tests for `gpcam` package."""
import unittest
import numpy as np
from gpcam import AutonomousExperimenterGP
from gpcam import GPOptimizer


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



class TestgpCAM(unittest.TestCase):
    """Tests for `gpcam` package."""

    def test_setUp(self, dim=2, N=20):
        """Set up test fixtures, if any."""
        x = np.random.rand(N, dim)
        y = np.sin(x[:, 0])
        index_set_bounds = np.array([[0., 1.], [0., 1.]])
        hyperparameter_bounds = np.array([[0.001, 1e9], [0.001, 100], [0.001, 100]])
        hps_guess = np.ones((3))
        ###################################################################################
        gp = GPOptimizer(x,y)
        gp.tell(x,y,noise_variances = np.ones(y.shape))
        gp.train_gp(hyperparameter_bounds=hyperparameter_bounds)

    def test_single_task(self, dim=2, N=20, write_data_cube=False):
        """Test something."""
        x = np.random.rand(N, dim)
        y = np.sin(x[:, 0])

        ######################################################
        def kernel_l2_single_task(x1, x2, hyperparameters, obj):
            hps = hyperparameters
            distance_matrix = np.zeros((len(x1), len(x2)))
            for i in range(len(x1[0])):
                distance_matrix += abs(np.subtract.outer(x1[:, i], x2[:, i]) / hps[i + 1]) ** 2
            distance_matrix = np.sqrt(distance_matrix)
            return hps[0] * obj.exponential_kernel(distance_matrix, 1)  # + noise

        index_set_bounds = np.array([[0., 1.], [0., 1.]])
        hyperparameter_bounds = np.array([[0.001, 1e9], [0.001, 100], [0.001, 100]])
        hps_guess = np.ones((3))
        ###################################################################################
        gp = GPOptimizer(x,y)
        gp.train_gp(hyperparameter_bounds)
        ######################################################
        ######################################################
        ######################################################
        print("evaluating acquisition function at [0.5,0.5,0.5]")
        print("=======================")
        r1 = gp.evaluate_acquisition_function(np.array([0.5, 0.5]), acquisition_function="shannon_ig")
        r2 = gp.evaluate_acquisition_function(np.array([0.5, 0.5]), acquisition_function=ac_func1)
        print("results: ", r1, r2)
        print()
        print("getting data from gp optimizer:")
        print("=======================")
        r = gp.get_data()
        print(r)
        print()
        print("ask()ing for new suggestions")
        print("=======================")
        r = gp.ask(np.array([[0.,1.],[0.,1.]]))
        print(r)
        print()
        print("getting the maximum (remember that this means getting the minimum of -f(x)):")
        print("=======================")
        r = gp.ask(np.array([[0.,1.],[0.,1.]]),acquisition_function="maximum")
        print(r)
        print("getting the minimum:")
        print("=======================")
        r = gp.ask(np.array([[0.,1.],[0.,1.]]),acquisition_function="minimum")
        print(r)
        print()

    def test_ae(self):
        ##set up your parameter space
        input_space = np.array([[3.0,45.8],
                              [4.0,47.0]])

        ##set up some hyperparameters, if you have no idea, set them to 1 and make the training bounds large
        init_hyperparameters = np.array([1,1,1])
        hyperparameter_bounds =  np.array([[0.01,100],[0.01,100.0],[0.01,100]])

        ##let's initialize the autonomous experimenter ...
        my_ae = AutonomousExperimenterGP(input_space, init_hyperparameters,
                                        hyperparameter_bounds,instrument_func = instrument,
                                        init_dataset_size=10)
        #...train...
        my_ae.train()
        my_ae.train_async()
        my_ae.update_hps()
        my_ae.kill_training()

        #...and run. That's it. You successfully executed an autonomous experiment.
        my_ae.go(N = 100)

        print("END")

    def test_acq_funcs(self):
        import numpy as np
        from gpcam.gp_optimizer import GPOptimizer

        #initialize some data
        x_data = np.random.uniform(size = (100,3))
        y_data = np.sin(np.linalg.norm(x_data, axis = 1))


        #initialize the GPOptimizer
        my_gpo = GPOptimizer(x_data, y_data)
        #tell() it some data
        my_gpo.train_gp(np.array([[0.001,100],[0.001,100],[0.001,100],[0.001,100]]))

        #let's make a prediction
        print("an example posterior mean at x = 0.44 :",my_gpo.posterior_mean(np.array([[0.44,1.,1.]])))
        print("")
        #now we can ask for a new point

        r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 5, acquisition_function="shannon_ig_multi")
        r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="shannon_ig")
        r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="shannon_ig_vec")
        r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="variance")
        r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="covariance")
        r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="ucb")
        r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="maximum")
        r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 5, acquisition_function="gradient", method = "local")
        r = my_gpo.ask(np.array([[0.,1.],[0.,1.],[0.,1.]]),n = 1, acquisition_function="target_probability", method = "local", args = {"a":1.,"b":2.})


