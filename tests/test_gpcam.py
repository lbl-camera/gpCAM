#!/usr/bin/env python

"""Tests for `gpcam` package."""
import numpy as np
from gpcam.gp_optimizer import GPOptimizer
import matplotlib.pyplot as plt



import unittest


class TestgpCAM(unittest.TestCase):
    """Tests for `gpcam` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_single_task(self,dim = 2, N = 100):
        """Test something."""
        x = np.random.rand(100,dim)
        y = np.empty((len(x),1))
        y[:,0] = np.sin(x[:,0])
        ######################################################
        ######################################################
        ######################################################
        #y = y/np.max(y)
        index_set_bounds = np.array([[0.,1.],[0.,1.]])
        hyperparameter_bounds = np.array([[0.001,1e9],[0.001,100],[0.001,100]])
        hps_guess = np.ones((3))
        ###################################################################################
        gp = GPOptimizer(2,1,1, index_set_bounds)
        gp.tell(x,y)
        gp.init_gp(hps_guess)
        gp.train_gp(hyperparameter_bounds)
        ######################################################
        ######################################################
        ######################################################
        print("evaluating objective function at [0.5,0.5,0.5]")
        print("=======================")
        r = gp.evaluate_objective_function(np.array([0.5,0.5]),objective_function = "shannon_ig")
        print("result: ",r)
        input("Continue with ENTER")
        print("getting data from gp optimizer:")
        print("=======================")
        r = gp.get_data()
        print(r)
        input("Continue with ENTER")
        print("ask()ing for new suggestions")
        print("=======================")
        r = gp.ask()
        print(r)
        input("Continue with ENTER")
        print("getting the maximum (remember that this means getting the minimum of -f(x)):")
        print("=======================")
        r = gp.ask(objective_function = "maximum")
        print(r)
        print("getting the minimum:")
        print("=======================")
        r = gp.ask(objective_function = "minimum")
        print(r)
        input("Continue with ENTER")
        print("Writing interpolation to file...")
        print("=======================")


        ar3d = np.empty((50,50))
        l = np.empty((50*50,4))
        x = np.linspace(0,1,50)
        y = np.linspace(0,1,50)
        counter = 0
        for i in range(50):
            print("done ",((i+1.0)/50.0)*100.," percent")
            for j in range(50):
                res = gp.gp.posterior_mean(np.array([[x[i],y[j]]]))
                ar3d[i,j] = res["f(x)"]
                l[counter,0] = x[i]
                l[counter,1] = y[j]
                l[counter,3] = res["f(x)"] / 10000.0
                counter += 1

        file_name = "data_list.csv"
        np.savetxt(file_name,l, delimiter = ",",header = 'x coord, y coord, z_coord, scalar')
        print("==================================================")
        print("data cube written in 'data_list.csv'; you can use paraview to visualize it")
        print("END")

