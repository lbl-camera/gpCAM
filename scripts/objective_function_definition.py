import numpy as np
import matplotlib.pyplot as plt

###############################
#README:
#objective functions can be defined here following the templates below
#x is the point where toe valuate the objective function
#obj is the gp object that contains functions such as posterior_mean()
#and posterior_covariance()...
#the return is a scalar which will be maximized
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

