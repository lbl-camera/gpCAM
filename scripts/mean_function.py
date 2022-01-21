#####################################
# README:
# The mean function is defined a f = f(x), X --> R
# However, the definitions should work for an array of points
# i.e. x is 2 dimensional
# the return is the evaluation of the mean function at a set of points x
# see example below
####################################


def himmel_blau(gp_obj, x, hyperparameters):
    return (x[:, 0] ** 2 + x[:, 1] - 11.0) ** 2 + (x[:, 0] + x[:, 1] ** 2 - 7.0) ** 2


def example_mean(gp_obj, x, hyperparameters):
    return himmel_blau(x[:, 0])
