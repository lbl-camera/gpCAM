import numpy as np
import matplotlib.pyplot as plt



##example for pure exploration
def pure_exploration(x,obj):
    res = obj.compute_posterior_fvGP_pdf(np.array([x]), obj.value_positions[-1], compute_posterior_covariances  = True)
    return -abs((np.sqrt(res["posterior covariances"][0,0,0]))-0.0)


##example for pure exploration/exploitation searching for high function values
def exploitation(x,obj):
    a = 3.0 #####change here, 3.0 for 95 percent confidence interval
    res = obj.compute_posterior_fvGP_pdf(np.array([x]), obj.value_positions[-1], compute_posterior_covariances  = True)
    return -(a*np.sqrt(res["posterior covariances"][0,0,0]))+((res["posterior means"][0,0])*np.sqrt(res["posterior covariances"][0,0,0]))


def shape_finding(x,obj):
    points = []
    #create the stencil
    for i in np.linspace(-50,50,10):
        for j in np.linspace(-50,50,10):
            points.append([x[0]+i,x[1]+j])
    points = np.array(points)
    #plot the stencil
    #call the Gaussian process
    res = obj.compute_posterior_fvGP_pdf(points, obj.value_positions[-1], compute_posterior_covariances  = True)
    return np.var(res["posterior means"])-4.0*np.sqrt(res["posterior covariances"][0,0,0])



###########################################################################
#########################Gradient Mode Functions##(depreciated)############
###########################################################################
def gaussian_bell(x, pos, w):
    return np.exp(-w * (x - pos) ** 2)



def evaluate_gp_gradient(x,obj):
    a = []
    for i in range(len(x)):
        res = obj.compute_posterior_fvGP_pdf_gradient(x, obj.value_positions[-1],direction = i)
        a.append(res["posterior mean gradients"][0,0])
    return np.linalg.norm(np.asarray(a))



def init_population(bounds, number_of_individuals, length_of_genome):
    # function returns a set of randomly chosen points in the Parameter space
    import random
    Population = np.zeros((number_of_individuals, length_of_genome))
    for j in range(number_of_individuals):
        for k in range(length_of_genome):
            Population[j, k] = bounds[k][0] + (
                random.random() * abs(bounds[k][1] - bounds[k][0])
            )
    return Population


def make_distribution(array):
    array = array - min(array)
    if max(array) > 0:
        array = array / max(array)
    if max(array) == 0:
        print(array)
        print("array in make_distribution is 0")

    y = np.zeros((100))
    x = np.linspace(0, 1, 100)
    for i in range(len(array)):
        y = y + (gaussian_bell(x, array[i], 100.0))
    y = y / (sum(y) * 0.01)
    return y



def compute_gradient_weight(bounds, obj):
    gradient_distribution = [0.6, 1.0, 0.01, 0.5]
    set_of_points = init_population(
        bounds, 100, len(bounds)
    )

    IndGrad = np.zeros((len(set_of_points)))
    for i in range(len(set_of_points)):
        IndGrad[i] = np.linalg.norm(evaluate_gp_gradient(obj,set_of_points[i]))
    gdc = gradient_distribution
    d = make_distribution(IndGrad)
    integral = sum(d[int(100 * gdc[0]) : int(100 * gdc[1])]) * 0.01
    if integral < gdc[2]:
        c = 0
    elif integral > gdc[3]:
        c = 0.0
    else:
        delta = abs(gdc[3] - gdc[2])
        c = (integral - gdc[2]) / delta
    return c


def gradient_mode(x,obj):
    gradient = np.linalg.norm(evaluate_gp_gradient(np.array([x]),obj))
    res = obj.compute_posterior_fvGP_pdf(np.array([x]), obj.value_positions[-1], compute_posterior_covariances  = True)
    a = 2.0
    obj_eval = (np.sqrt(abs(res["posterior covariances"][0,0,0])) *a* gradient) + np.sqrt(abs(res["posterior covariances"][0,0,0]))
    return -obj_eval
