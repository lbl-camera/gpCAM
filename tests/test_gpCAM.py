#!/usr/bin/env python

"""Tests for `gpcam` package."""
import unittest
import numpy as np
from gpcam.autonomous_experimenter import AutonomousExperimenterGP
from gpcam.gp_optimizer import GPOptimizer


def ac_func1(x, obj):
    r1 = obj.posterior_mean(x)["f(x)"]
    r2 = obj.posterior_covariance(x)["v(x)"]
    m_index = np.argmin(obj.data_y)
    m = obj.data_x[m_index]
    std_model = np.sqrt(r2)
    return -(r1 + 3.0 * std_model)

def instrument(data, instrument_dict=None):
    for entry in data:
        entry["value"] = np.sin(np.linalg.norm(entry["position"]))
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
        gp = GPOptimizer(dim, index_set_bounds)
        gp.tell(x, y)
        gp.init_gp(hps_guess)
        gp.train_gp(hyperparameter_bounds)

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
            # if len(x1) == len(x2): noise = np.identity(len(x1)) * hps[2]
            # else: noise = 0.0
            return hps[0] * obj.exponential_kernel(distance_matrix, 1)  # + noise

        index_set_bounds = np.array([[0., 1.], [0., 1.]])
        hyperparameter_bounds = np.array([[0.001, 1e9], [0.001, 100], [0.001, 100]])
        hps_guess = np.ones((3))
        ###################################################################################
        gp = GPOptimizer(dim, index_set_bounds)
        gp.tell(x, y)
        gp.init_gp(hps_guess, gp_kernel_function=kernel_l2_single_task)
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
        r = gp.ask()
        print(r)
        print()
        print("getting the maximum (remember that this means getting the minimum of -f(x)):")
        print("=======================")
        r = gp.ask(acquisition_function="maximum")
        print(r)
        print("getting the minimum:")
        print("=======================")
        r = gp.ask(acquisition_function="minimum")
        print(r)
        print()

    def test_ae(self):
        ##set up your parameter space
        parameters = np.array([[3.0,45.8],
                              [4.0,47.0]])

        ##set up some hyperparameters, if you have no idea, set them to 1 and make the training bounds large
        init_hyperparameters = np.array([1,1,1])
        hyperparameter_bounds =  np.array([[0.01,100],[0.01,100.0],[0.01,100]])

        ##let's initialize the autonomous experimenter ...
        my_ae = AutonomousExperimenterGP(parameters, init_hyperparameters,
                                        hyperparameter_bounds,instrument_func = instrument,
                                        init_dataset_size=10)
        #...train...
        my_ae.train()

        #...and run. That's it. You successfully executed an autonomous experiment.
        my_ae.go(N = 100)

        print("END")
