import random
import numpy as np
from gpcam import global_config as conf
import math
import os
import re, os.path
import matplotlib.pyplot as plt


def dict2d_2_nparray(a):
    return np.array([list(x.values()) for x in list(a.values())])


def random_number(a, b):
    return min(a, b) + (random.random() * abs(b - a))


def l2_norm(vec1, vec2):
    return np.linalg.norm(np.subtract(vec1, vec2))


def is_element_of(point, space):
    InSpace = True
    for i in range(len(point)):
        if point[i] < space[i][0] or point[i] > space[i][1]:
            InSpace = False
    return InSpace


def point_hit_boundary(point, space):
    space = np.asarray(space)
    for i in range(len(point)):
        if abs(point[i] - space[i, 0]) < 1e-6 or abs(point[i] - space[i, 1]) < 1e-6:
            return True
    return False


def kronecker_delta(a, b):
    if a == b:
        result = 1.0
    if a != b:
        result = 0.0
    return result


def kronecker_delta_3d(a, b, c):
    if a == b and b == c:
        result = 1.0
    else:
        result = 0.0
    return result



def l1_dict_norm(Dict):
    norm = 0
    for entry in Dict:
        norm = norm + abs(Dict[entry])
    return norm


def monte_carlo_integration(LowerBounds, UpperBounds, Func, *argv):
    # print("in monte carlo")
    volume = 1.0
    for i in range(len(LowerBounds)):
        volume = volume * abs(UpperBounds[i] - LowerBounds[i])
    # b = np.zeros((0))
    integral = 1.0
    counter = 0
    mean_1 = 0.0
    s_1 = 0.0
    error = 1e6
    while abs(error / integral) > 0.01:
        a = np.random.uniform(LowerBounds, UpperBounds, len(LowerBounds))
        f_eval = Func(a, *argv)
        counter = counter + 1
        mean = mean_1 + ((f_eval - mean_1) / counter)
        s = s_1 + ((f_eval - mean_1) * (f_eval - mean))
        var = s / counter
        error = volume * np.sqrt(var / counter)
        integral = volume * mean
        if counter > 1e6:
            print(
                "Finding the mc-integral is taking a little longer:",
                integral,
                error,
                abs(error / integral),
                counter,
            )
            break
        mean_1 = mean
        s_1 = s
        if counter < 1000:
            error = 1e6

    # print("Monte Carlo Integration concluded. integral: ",integral,' std: ',std,' percent: ',abs(std/integral),' iterations: ',len(b))
    return integral, error


def kl_divergence(m1, m2, C1, C1_Inv, C2, C2_Inv):
    return 0.5 * (
        np.trace(C1_Inv.dot(C2)) * (m2 - m1).dot(C2_Inv).dot(m2 - m1)
        - len(m1)
        + np.log(np.det(C2) / np.det(C1))
    )

def determine_signal_variance_range(values):
    v = np.var(values)
    a = [v/100.0,v*100.0]
    return a

def delete_files():
    path_new_command = "../data/command/"
    path_new_result = "../data/result/"

    if os.path.isfile(path_new_result + "result.npy"):
        os.remove(path_new_result + "result.npy")
    if os.path.isfile(path_new_command + "command.npy"):
        os.remove(path_new_command + "command.npy")
    mypath = "../data/current_data/"
    for root, dirs, files in os.walk(mypath):
        for file in files:
            if file == ".init": continue
            os.remove(os.path.join(root, file))

def himmel_blau(x):
    return (x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2

def eggholder(x):
    x1 = x[0]
    x2 = x[1]
    term1 = -(x2+47.0) * np.sin(np.sqrt(abs(x2+x1/2.0+47.0)));
    term2 = -x1 * np.sin(np.sqrt(abs(x1-(x2+47.0))));

    y = term1 + term2
    return y
