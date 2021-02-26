import numpy as np
from fvgp.fvgp import FVGP
from . import surrogate_model as sm
import time


class GPExpertsOptimizer():
    """
    This class can combine a set of Gaussian Experts (GPOptimizer objects)
    and combine them.
    This es especially usefull for large-scale GPs
    
    Attributes:
        a list of GPOptimizer Objects
    """
    def __init__(self,gp_optimizer_obj_list):
        self.gpol = gp_optimizer_obj_list
        self.kl_div = self.gpol[0].gp.kl_div
        self.kl_div_grad = self.gpol[0].gp.kl_div_grad
        self.entropy = self.gpol[0].gp.entropy

    def predict(x):
        S = self.gpol[0].gp.posterior_covariance["S(x)"]
        mu = self.gpol[0].gp.posterior_mean["f(x)"]
        for i in range(1,len(self.gpol)):
            cov = self.gpol[i].gp.posterior_covariance["S(x)"]
            inv = np.linalg.inv(S + cov)
            mu = cov @ inv @ mu + S @ inv @ self.gpol[i].gp.posterior_mean["f(x)"]
            S  = S @ inv @ cov
        return {"f(x)":mu, "S(x)": S}

    def ask(self, position = None, n = 1,
            acquisition_function = "covariance",
            optimization_bounds = None,
            optimization_method = "global",
            optimization_pop_size = 20,
            optimization_max_iter = 20,
            optimization_tol = 10e-6,
            dask_client = False):
    return 0

