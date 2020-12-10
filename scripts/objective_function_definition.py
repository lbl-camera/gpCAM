import numpy as np
import matplotlib.pyplot as plt

###############################
#README:
#objective functions can be defined here following the templates below
#the objective function is a function defined as f=f(x), X --> R
#x is the point where to evaluate the objective function
#obj is the gp object that contains functions such as posterior_mean()
#and posterior_covariance(), shannon_information_gain(),...
#the return is either a scalar or a 1d array
#that contains the objective function values for a set of points x
#The objective funciton defined here will be MAXIMIZED in the algorithm
#to find the next optimal point
###############################

def exploration(x,obj):
    cov = obj.posterior_covariance(x)["v(x)"]
    return np.asscalar(cov)


##example for pure exploration/exploitation searching for high function values
def upper_confidence_bounds(x,obj):
    a = 3.0 #####change here, 3.0 for 95 percent confidence interval
    cov = obj.posterior_mean(x)["f(x)"]
    mean = obj.posterior_covariance(x)["v(x)"]
    return np.asscalar(mean + a * cov)

