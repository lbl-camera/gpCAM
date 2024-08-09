import numpy as np
from matplotlib import pyplot as plt
from loguru import logger
from gpcam.gp_kernels import *


import time
import numpy as np

###################################
######gp2Scale GPU kernels#########
###################################
def f_gpu(x,x0, radii, amplts, device):
    b1 = b_gpu(x, x0[0:3], radii[0], amplts[0], device)  ###x0[0] ... D-dim location of bump func 1
    b2 = b_gpu(x, x0[3:6], radii[1], amplts[1], device)  ###x0[1] ... D-dim location of bump func 2
    b3 = b_gpu(x, x0[6:9], radii[2], amplts[2], device)  ###x0[1] ... D-dim location of bump func 2
    b4 = b_gpu(x, x0[9:12],radii[3], amplts[3], device)  ###x0[1] ... D-dim location of bump func 2
    return b1 + b2 + b3 + b4

def g_gpu(x,x0, radii, amplts,device):
    b1 = b_gpu(x, x0[0:3], radii[0], amplts[0], device)  ###x0[0] ... D-dim location of bump func 1
    b2 = b_gpu(x, x0[3:6], radii[1], amplts[1], device)  ###x0[1] ... D-dim location of bump func 2
    b3 = b_gpu(x, x0[6:9], radii[2], amplts[2], device)  ###x0[1] ... D-dim location of bump func 2
    b4 = b_gpu(x, x0[9:12],radii[3], amplts[3], device)  ###x0[1] ... D-dim location of bump func 2
    return b1 + b2 + b3 + b4


def b_gpu(x,x0, r, ampl, device):
    """
    evaluates the bump function
    x ... a point (1d numpy array)
    x0 ... 1d numpy array of location of bump function
    returns the bump function b(x,x0) with radius r
    """
    x_new = x - x0
    d = torch.linalg.norm(x_new, axis = 1)
    a = torch.zeros(d.shape).to(device, dtype = torch.float32)
    a = 1.0 - (d**2/r**2)
    i = torch.where(a > 0.0)
    bump = torch.zeros(a.shape).to(device, dtype = torch.float32)
    e = torch.exp((-1.0/a[i])+1).to(device, dtype = torch.float32)
    bump[i] = ampl * e
    return bump


def get_distance_matrix_gpu(x1,x2,device,hps):
    d = torch.zeros((len(x1),len(x2))).to(device, dtype = torch.float32)
    for i in range(x1.shape[1]):
        d += ((x1[:,i].reshape(-1, 1) - x2[:,i])/hps[i])**2
    return torch.sqrt(d)


def wendland_gpu(x1,x2, radius,device):
    d = get_distance_matrix_gpu(x1,x2,device)
    #d[d == 0.0] = 1e-16
    d[d > radius] = radius
    r = radius
    a = d/r
    kernel = (1.-a)**8 * (35.*a**3 + 25.*a**2 + 8.*a + 1.)
    return kernel


def wendland_cpu(x1,x2, radius,device):
    d = get_distance_matrix(x1,x2)
    d[d > radius] = radius
    r = radius
    a = d/r
    kernel = (1.-a)**8 * (35.*a**3 + 25.*a**2 + 8.*a + 1.)
    return kernel


def ks_gpu_wend_b(x1,x2,hps,cuda_device):
    k1 = torch.outer(f_gpu(x1,hps[0:12],hps[12:16],hps[16:20],cuda_device),
                     f_gpu(x2,hps[0:12],hps[12:16],hps[16:20],cuda_device)) + \
         torch.outer(g_gpu(x1,hps[20:32],hps[32:36],hps[36:40],cuda_device),
                     g_gpu(x2,hps[20:32],hps[32:36],hps[36:40],cuda_device))
    k2 = wendland_gpu(x1,x2, hps[1],cuda_device)
    return k1 + hps[0]*k2

def ks_gpu_wend(x1,x2,hps,cuda_device):
    return hps[0]*wendland_gpu(x1,x2, hps[1],cuda_device)


def kernel_gpu_wend(x1,x2, hps):
    cuda_device = torch.device("cuda:0")
    x1_dev = torch.from_numpy(x1).to(cuda_device, dtype = torch.float32)
    x2_dev = torch.from_numpy(x2).to(cuda_device, dtype = torch.float32)
    hps_dev = torch.from_numpy(hps).to(cuda_device, dtype = torch.float32)
    ksparse = ks_gpu_wend(x1_dev,x2_dev,hps_dev,cuda_device).cpu().numpy()
    return ksparse
###################################
######gp2Scale CPU kernels#########
###################################
def b_cpu(x,x0,r = 0.1, ampl = 1.0):
    """
    evaluates the bump function
    x ... a point (1d numpy array)
    x0 ... 1d numpy array of location of bump function
    returns the bump function b(x,x0) with radius r
    """
    x_new = x - x0
    d = np.linalg.norm(x_new, axis = 1)
    a = np.zeros(d.shape)
    a = 1.0 - (d**2/r**2)
    i = np.where(a > 0.0)
    bump = np.zeros(a.shape)
    bump[i] = ampl * np.exp((-1.0/a[i])+1)
    return bump

def f_cpu(x,x0, radii, amplts):
    b1 = b_cpu(x, x0[0:3],r = radii[0], ampl = amplts[0])  ###x0[0] ... D-dim location of bump func 1
    b2 = b_cpu(x, x0[3:6],r = radii[1], ampl = amplts[1])  ###x0[1] ... D-dim location of bump func 2
    b3 = b_cpu(x, x0[6:9],r = radii[2], ampl = amplts[2])  ###x0[1] ... D-dim location of bump func 2
    b4 = b_cpu(x, x0[9:12],r = radii[3], ampl = amplts[3])  ###x0[1] ... D-dim location of bump func 2
    return b1 + b2 + b3 + b4

def g_cpu(x,x0, radii, amplts):
    b1 = b_cpu(x, x0[0:3],r = radii[0], ampl = amplts[0])  ###x0[0] ... D-dim location of bump func 1
    b2 = b_cpu(x, x0[3:6],r = radii[1], ampl = amplts[1])  ###x0[1] ... D-dim location of bump func 2
    b3 = b_cpu(x, x0[6:9],r = radii[2], ampl = amplts[2])  ###x0[1] ... D-dim location of bump func 2
    b4 = b_cpu(x, x0[9:12],r = radii[3], ampl = amplts[3])  ###x0[1] ... D-dim location of bump func 2
    return b1 + b2 + b3 + b4

def get_distance_matrix_cpu(x1,x2):
    d = np.zeros((len(x1),len(x2)))
    for i in range(x1.shape[1]):
        d += ((x1[:,i].reshape(-1, 1) - x2[:,i]))**2
    return np.sqrt(d)


def wendland_cpu(x1,x2, radius):
    d = get_distance_matrix_cpu(x1,x2)
    #d[d == 0.0] = 1e-16
    d[d > radius] = radius
    r = radius
    a = d/r
    kernel = (1.-a)**8 * (35.*a**3 + 25.*a**2 + 8.*a + 1.)
    return kernel


def kernel_cpu(x1,x2, hps):
    k = np.outer(f_cpu(x1,hps[0:12],hps[12:16],hps[16:20]),
                 f_cpu(x2,hps[0:12],hps[12:16],hps[16:20])) + \
        np.outer(g_cpu(x1,hps[20:32],hps[32:36],hps[36:40]),
                 g_cpu(x2,hps[20:32],hps[32:36],hps[36:40]))
    return k + hps[40] * wendland_cpu(x1,x2, hps[41])

def cory(x1, x2, hps):
    d = get_anisotropic_distance_matrix(x1[:, 0:-1], x2[:, 0:-1], hps[1:7])
    k1 = hps[0] * matern_kernel_diff1(d, 1.)
    l1 = x1[:, -1]
    l2 = x2[:, -1]
    c = hps[7]
    delta = hps[8]
    k2 = c + np.outer(((1. - l1) ** (1. + delta)), ((1. - l2) ** (1. + delta)))
    return k1 * k2
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
#deep kernel
def deep_multi_task_kernel(x1, x2, hps):  # pragma: no cover
    signal_var = hps[0]
    length_scale = hps[1]
    hps_nn = hps[2:]
    w1_indices = np.arange(0, gp_deep_kernel_layer_width * iset_dim)
    last = gp_deep_kernel_layer_width * iset_dim
    w2_indices = np.arange(last, last + gp_deep_kernel_layer_width ** 2)
    last = last + gp_deep_kernel_layer_width ** 2
    w3_indices = np.arange(last, last + gp_deep_kernel_layer_width * iset_dim)
    last = last + gp_deep_kernel_layer_width * iset_dim
    b1_indices = np.arange(last, last + gp_deep_kernel_layer_width)
    last = last + gp_deep_kernel_layer_width
    b2_indices = np.arange(last, last + gp_deep_kernel_layer_width)
    last = last + gp_deep_kernel_layer_width
    b3_indices = np.arange(last, last + iset_dim)

    n.set_weights(hps_nn[w1_indices].reshape(gp_deep_kernel_layer_width, iset_dim),
                       hps_nn[w2_indices].reshape(gp_deep_kernel_layer_width, gp_deep_kernel_layer_width),
                       hps_nn[w3_indices].reshape(iset_dim, gp_deep_kernel_layer_width))
    n.set_biases(hps_nn[b1_indices].reshape(gp_deep_kernel_layer_width),
                      hps_nn[b2_indices].reshape(gp_deep_kernel_layer_width),
                      hps_nn[b3_indices].reshape(iset_dim))
    x1_nn = n.forward(x1)
    x2_nn = n.forward(x2)
    d = get_distance_matrix(x1_nn, x2_nn)
    k = signal_var * matern_kernel_diff1(d, length_scale)
    return k





def kernel_l2_single_task(x1, x2, hyperparameters):
    ################################################################
    ###standard anisotropic kernel in an input space with l2########
    ###########################for single task######################
    """
    x1: 2d numpy array of points
    x2: 2d numpy array of points
    hyperparameters

    Return:
    -------
    Kernel Matrix
    """
    hps = hyperparameters
    distance_matrix = np.zeros((len(x1), len(x2)))
    for i in range(len(x1[0]) - 1):
        distance_matrix += abs(np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
    distance_matrix = np.sqrt(distance_matrix)
    return hps[0] * matern_kernel_diff1(distance_matrix, 1)



def kernel_l1(x1, x2, hp):
    ################################################################
    ###standard anisotropic kernel in an input space with l1########
    ################################################################

    d1 = abs(np.subtract.outer(x1[:, 0], x2[:, 0]))
    d2 = abs(np.subtract.outer(x1[:, 1], x2[:, 1]))
    d3 = abs(np.subtract.outer(x1[:, 2], x2[:, 2]))
    return hp[0] * np.exp(-d1 / hp[1]) * np.exp(-d2 / hp[2]) * np.exp(-d3 / hp[3])


def fvgp_kernel(x1, x2, hps):
    ################################################################
    ###in this kernel we are defining non-stationary length scales##
    ################################################################
    ##UNDER CONSTRUCTION
    ##only 1d so far
    x_center = np.add.outer(x1, x2) / 2.0
    d = abs(np.subtract.outer(x1, x2))
    ##Kernel of the form x1.T @ M @ x2 * k(x1,x2)
    return hps[0] * np.exp(-d ** 2 / l)
    # return hps[0] * np.exp(-d/l)


##################################################
#######non-stationary kernel######################
##################################################
def sig_var(x, hps):
    r = hps[0] + hps[1] * abs(x[0]) + hps[2] * abs(x[1])
    return r


def lamb1(x, hps):
    r = hps[3] + hps[4] * abs(x[0]) + hps[5] * abs(x[1])
    return r


def lamb2(x, hps):
    r = hps[6] + hps[7] * abs(x[0]) + hps[8] * abs(x[1])
    return r


def gamma(x, hps):
    r = hps[9] + hps[10] * x[0] + hps[11] * x[1]
    return r


def non_stat_kernel_2d(x1, x2, hps):
    x1 = x1[:, :-1]
    x2 = x2[:, :-1]
    logger.debug(x1)
    logger.debug(x2)
    C = np.empty((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            s1 = sig_var(x1[i], hps)
            s2 = sig_var(x2[j], hps)
            lambda11 = lamb1(x1[i], hps)
            lambda12 = lamb2(x1[j], hps)
            lambda21 = lamb1(x2[i], hps)
            lambda22 = lamb2(x2[j], hps)
            gamma1 = gamma(x1[i], hps)
            gamma2 = gamma(x2[j], hps)
            L1 = np.array([[lambda11, 0.0], [0.0, lambda12]])
            L2 = np.array([[lambda21, 0.0], [0.0, lambda22]])
            G1 = np.array([[np.cos(gamma1), -np.sin(gamma1)], [np.sin(gamma1), np.cos(gamma1)]])
            G2 = np.array([[np.cos(gamma2), -np.sin(gamma2)], [np.sin(gamma2), np.cos(gamma2)]])
            Sig1 = G1 @ L1 @ G1.T
            Sig2 = G2 @ L2 @ G2.T
            Q = ((x1[i] - x2[j])).T @ np.linalg.inv((Sig1 + Sig2) / 2.0) @ (x1[i] - x2[j])
            M = matern_kernel_diff1(np.sqrt(Q), 1)  ###change base kernel here
            det1 = np.linalg.det(Sig1) ** 0.25
            det2 = np.linalg.det(Sig2) ** 0.25
            det3 = np.sqrt(np.linalg.det((Sig1 + Sig2) / 2.0))

            C[i, j] = s1 * s2 * ((det1 * det2) / (det3)) * M

    return C


##################################################

def symmetric_kernel(x1, x2, hps):
    ################################################################
    ###in this kernel we are enforcing symmetry in the x direction##
    ################################################################
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0
    x1_ = np.array(x1)
    x2_ = np.array(x2)
    x1_[:, 0] = -x1[:, 0]
    x2_[:, 0] = -x2[:, 0]

    for i in range(len(x1[0])):
        d1 += np.abs(np.subtract.outer(x1[:, i], x2[:, i])) ** 2
        d2 += np.abs(np.subtract.outer(x1_[:, i], x2[:, i])) ** 2
        d3 += np.abs(np.subtract.outer(x1[:, i], x2_[:, i])) ** 2
        d4 += np.abs(np.subtract.outer(x1_[:, i], x2_[:, i])) ** 2
    d1 = np.sqrt(d1)
    d2 = np.sqrt(d2)
    d3 = np.sqrt(d3)
    d4 = np.sqrt(d4)
    l = hps[1]
    k1 = np.exp(-np.abs(d1) ** 2 / l)
    k2 = np.exp(-np.abs(d2) ** 2 / l)
    k3 = np.exp(-np.abs(d3) ** 2 / l)
    k4 = np.exp(-np.abs(d4) ** 2 / l)
    k = (k1 + k2 + k3 + k4) / 4.0
    return hps[0] * k


def symmetric_kernel2(x1, x2, hps):
    ######################################################################
    ###in this kernel we are enforcing symmetry in the x and y direction##
    ######################################################################
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0
    d5 = 0
    d6 = 0
    d7 = 0
    d8 = 0
    d9 = 0
    d10 = 0
    d11 = 0
    d12 = 0
    d13 = 0
    d14 = 0
    d15 = 0
    d16 = 0
    x1_0 = np.array(x1)
    x2_0 = np.array(x2)
    x1_1 = np.array(x1)
    x2_1 = np.array(x2)

    x1_0[:, 0] = -x1[:, 0]
    x2_0[:, 0] = -x2[:, 0]
    x1_1[:, 1] = -x1[:, 1]
    x2_1[:, 1] = -x2[:, 1]
    x1_12 = np.array(-x1)
    x2_12 = np.array(-x2)

    for i in range(len(x1[0])):
        d1 += np.abs(np.subtract.outer(x1[:, i], x2[:, i])) ** 2
        d2 += np.abs(np.subtract.outer(x1[:, i], x2_0[:, i])) ** 2
        d3 += np.abs(np.subtract.outer(x1[:, i], x2_1[:, i])) ** 2
        d4 += np.abs(np.subtract.outer(x1[:, i], x2_12[:, i])) ** 2

        d5 += np.abs(np.subtract.outer(x1_0[:, i], x2[:, i])) ** 2
        d6 += np.abs(np.subtract.outer(x1_0[:, i], x2_0[:, i])) ** 2
        d7 += np.abs(np.subtract.outer(x1_0[:, i], x2_1[:, i])) ** 2
        d8 += np.abs(np.subtract.outer(x1_0[:, i], x2_12[:, i])) ** 2

        d9 += np.abs(np.subtract.outer(x1_1[:, i], x2[:, i])) ** 2
        d10 += np.abs(np.subtract.outer(x1_1[:, i], x2_0[:, i])) ** 2
        d11 += np.abs(np.subtract.outer(x1_1[:, i], x2_1[:, i])) ** 2
        d12 += np.abs(np.subtract.outer(x1_1[:, i], x2_12[:, i])) ** 2

        d13 += np.abs(np.subtract.outer(x1_12[:, i], x2[:, i])) ** 2
        d14 += np.abs(np.subtract.outer(x1_12[:, i], x2_0[:, i])) ** 2
        d15 += np.abs(np.subtract.outer(x1_12[:, i], x2_1[:, i])) ** 2
        d16 += np.abs(np.subtract.outer(x1_12[:, i], x2_12[:, i])) ** 2

    d1 = np.sqrt(d1)
    d2 = np.sqrt(d2)
    d3 = np.sqrt(d3)
    d4 = np.sqrt(d4)
    d5 = np.sqrt(d5)
    d6 = np.sqrt(d6)
    d7 = np.sqrt(d7)
    d8 = np.sqrt(d8)
    d9 = np.sqrt(d9)
    d10 = np.sqrt(d10)
    d11 = np.sqrt(d11)
    d12 = np.sqrt(d12)
    d13 = np.sqrt(d13)
    d14 = np.sqrt(d14)
    d15 = np.sqrt(d15)
    d16 = np.sqrt(d16)
    l = hps[1]
    k1 = np.exp(-np.abs(d1) ** 2 / l)
    k2 = np.exp(-np.abs(d2) ** 2 / l)
    k3 = np.exp(-np.abs(d3) ** 2 / l)
    k4 = np.exp(-np.abs(d4) ** 2 / l)
    k5 = np.exp(-np.abs(d5) ** 2 / l)
    k6 = np.exp(-np.abs(d6) ** 2 / l)
    k7 = np.exp(-np.abs(d7) ** 2 / l)
    k8 = np.exp(-np.abs(d8) ** 2 / l)
    k9 = np.exp(-np.abs(d9) ** 2 / l)
    k10 = np.exp(-np.abs(d10) ** 2 / l)
    k11 = np.exp(-np.abs(d11) ** 2 / l)
    k12 = np.exp(-np.abs(d12) ** 2 / l)
    k13 = np.exp(-np.abs(d13) ** 2 / l)
    k14 = np.exp(-np.abs(d14) ** 2 / l)
    k15 = np.exp(-np.abs(d15) ** 2 / l)
    k16 = np.exp(-np.abs(d16) ** 2 / l)
    k = (k1 + k2 + k3 + k4 + k5 + k6 + k7 + k8 + k9 + k10 + k11 + k12 + k13 + k14 + k15 + k16) / 16.0
    return hps[0] * k


def periodic_kernel_2d(x1, x2, hps):
    ####
    ####change depending on periodicity in x or y direction
    ####this kernel need 4 hps [sigma, l1,l2,p]
    c = (x1[:, 0] + x2[:, 0]) / 2.0
    offset = 2.0
    p = (hps[-1] * c) + offset
    # print(p)
    # p = 2.0 * np.pi
    x1_newp = np.array(x1)
    x1_newm = np.array(x1)
    x2_newp = np.array(x2)
    x2_newm = np.array(x2)
    ##change here for different direction
    x1_newp[:, 1] = x1_newp[:, 1] + p
    x2_newp[:, 1] = x2_newp[:, 1] + p
    x1_newm[:, 1] = x1_newm[:, 1] - p
    x2_newm[:, 1] = x2_newm[:, 1] - p
    #####################################
    k = kernel_l2_single_task

    k1 = k(x1, x2, hps[:-1])
    k2 = k(x1, x2_newp, hps[:-1])
    k3 = k(x1, x2_newm, hps[:-1])
    k4 = k(x1_newp, x2, hps[:-1])
    k5 = k(x1_newm, x2, hps[:-1])
    k6 = k(x1_newp, x2_newp, hps[:-1])
    k7 = k(x1_newp, x2_newm, hps[:-1])
    k8 = k(x1_newm, x2_newp, hps[:-1])
    k9 = k(x1_newm, x2_newm, hps[:-1])
    return (1.0 / 9.0) * (k1 + k2 + k3 + k4 + k5 + k6 + k7 + k8 + k9)


###########################################################
#############important psd proofs##########################
###########################################################
def psd_kernel_test1(func, hps):
    for i in range(100):
        x = np.random.rand(1000)
        K = func(x, x, hps)
        logger.debug("check if it is 0 or larger (allow for machine precision zero): {}", np.min(np.real(np.linalg.eig(K)[0])))


def psd_proof(N, kernel):
    a = (np.random.rand(N) * 10.0) - 5.0
    x1 = (np.random.rand(N) * 10.0) - 5.0
    s = 0
    for i in range(N):
        for j in range(N):
            s += a[i] * a[j] * kernel(x1[i], x1[j])
    return s


def fft_kernel_ckeck():
    N = 1024
    x = np.arange(-10, 10, 20. / (2.0 * N))
    y = np.exp(-x ** 2)
    y_fft = np.fft.fftshift(np.abs(np.fft.fft(y))) / np.sqrt(len(y))
    plt.plot(x, y)
    plt.plot(x, y_fft)
    plt.show()
