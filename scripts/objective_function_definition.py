import numpy as np
import matplotlib.pyplot as plt

###############################
#objective functions can be defined here following the templates below
#Note: the objective function returns a scalar whiich will be maximized 
#to find the next optimal point
###############################

def pure_exploration(x,obj):
    res = 
    return abs((np.sqrt(res["posterior covariances"][0,0,0]))-0.0)


##example for pure exploration/exploitation searching for high function values
def upper_confidence_bounds(x,obj):
    a = 3.0 #####change here, 3.0 for 95 percent confidence interval
    res = obj.compute_posterior_fvGP_pdf(np.array([x]), obj.value_positions[-1], compute_posterior_covariances  = True)
    return -(a*np.sqrt(res["posterior covariances"][0,0,0]))+((res["posterior means"][0,0])*np.sqrt(res["posterior covariances"][0,0,0]))

