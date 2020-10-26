import numpy as np

def himmel_blau(gp_obj,x,hyperparameters):
    #if len(x.shape) == 1: x = np.array([x])
    return (x[:,0] ** 2 + x[:,1] - 11.0) ** 2 + (x[:,0] + x[:,1] ** 2 - 7.0) ** 2

def example_mean(gp_obj,x,hyperparameters):
    return himmel_blau(x)
