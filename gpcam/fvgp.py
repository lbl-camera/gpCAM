#!/usr/bin/env python

import dask.distributed as distributed
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

import itertools
import time
import torch
import numba as nb
from functools import partial

class FVGP:
    """
    GP class: Finds hyper-parameters and therefore the mean
    and covariance of a (multi-output) Gaussian process

    symbols:
        N: Number of points in the data set
        n: number of return values
        dim1: number of dimension of the input space
        dim2: number of dimension of the output space

    Attributes:
        input_space_dim (int):         dim1
        output_space_dim (int):        dim2
        output_number (int):           n
        points (N x dim1 numpy array): array of points.
        values (N x n numpy array):    array of values.
    Optional Attributes:
        value_positions (N x dim1 x dim2 numpy array): the positions of the outputs in the output space, default = [0,1,2,...]
        variances (N x n numpy array):                  variances of the values, default = [0,0,...]
        compute_device:                                 cpu/gpu, default = cpu
        gp_kernel_function(func):                       None/function defining the kernel def name(x1,x2,hyperparameters,self), default = None
        gp_mean_function(func):                         None/a function def name(x, self), default = None
        init_hyperparameters (1d list):                default: list of [1,1,...]
        sparse (bool):                                  default = False

    Example:
        obj = FVGP(
            input_space_dim = 3,
            output_space_dim = 1,
            output_number = 2,
            points = np.array([[1,2,3],
                                4,5,6]),
            values = np.array([[2,3],
                            [13,27.2]]),
            value_positions = np.array([[[0]],[[1]]]),
            variances = np.array([[0.001,0.01],
                                [0.1,2]]),
            gp_kernel_function = kernel_function,
            init_hyperparameters = [2,3,4,5],
            gp_mean_function = some_mean_function
        )
    ---------------------------------------------
    """
    def __init__(
        self,
        input_space_dim,
        output_space_dim,
        output_number,
        points,
        values,
        init_hyperparameters,
        value_positions = None,
        variances = None,
        compute_device = "cpu",
        gp_kernel_function = None,
        gp_mean_function = None,
        sparse = False,
        compute_inverse = False
        ):

        """
        The constructor for the fvgp class.
        type help(FVGP) for more information about attributes, methods and their parameters
        """
        ##########################################
        #######check if dimensions match##########
        ##########################################
        if input_space_dim != len(points[0]):
            raise ValueError("input space dimensions are not in agreement with the point positions given")
        ##########################################
        if output_number != len(values[0]):
            raise ValueError("the output number is not in agreement with the data values given")
        ##########################################
        self.input_dim = input_space_dim
        self.output_dim = output_space_dim
        self.output_num = output_number
        self.data_x = points
        self.point_number = len(self.data_x)
        self.data_y = values
        self.compute_device = compute_device
        self.sparse = sparse
        self.compute_inverse = compute_inverse
        ##########################################
        #######prepare value positions############
        ##########################################
        if self.output_dim == 1 and isinstance(value_positions, np.ndarray) == False:
            self.value_positions = self.compute_standard_value_positions()
        elif self.output_dim > 1 and isinstance(value_positions, np.ndarray) == False:
            raise ValueError(
                "If the dimensionality of the output space is > 1, the value positions have to be given to the FVGP class"
            )
        else:
            self.value_positions = value_positions

        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            self.variances = np.ones((values.shape)) * abs(np.mean(self.data_y[0]) / 100.0)
            print("CAUTION: you have not provided data variances, they will set to be 1 percent of the |mean| of the data values!")
        else:
            self.variances = variances
        ##########################################
        #######define kernel and mean function####
        ##########################################
        if gp_kernel_function is None:
            self.kernel = self.default_kernel
        else:
            self.kernel = gp_kernel_function
        self.d_kernel_dx = self.d_gp_kernel_dx
        self.gp_mean_function = gp_mean_function
        if gp_mean_function is None:
            self.mean_function = self.default_mean_function
        else:
            self.mean_function = gp_mean_function

        ##########################################
        #######prepare hyper parameters###########
        ##########################################
        self.hyperparameters = init_hyperparameters
        ##########################################
        #transform index set and elements#########
        ##########################################
        self.iset_dim = self.input_dim + self.output_dim
        self.data_x, self.data_y, self.variances = self.transform_index_set()
        self.point_number = len(self.data_x)
        #########################################
        ####compute prior########################
        #########################################
        self.compute_prior_fvGP_pdf()
######################################################################
######################################################################
######################################################################
    def update_gp_data(
        self,
        points,
        values,
        value_positions = None,
        variances = None,
        ):

        """
        This function updates the data in the gp_class.


        Attributes:
            points (N x dim1 numpy array): An array of points.
            values (N x n):                An array of values.

        optional attributes:
            values_positions (N x dim1 x dim2 numpy array): the positions of the outputs in the output space
            variances (N x n):                              variances of the values
            """
        self.data_x = points
        self.point_number = len(self.data_x)
        self.data_y = values
        ##########################################
        #######prepare value positions############
        ##########################################
        if self.output_dim == 1 and isinstance(value_positions, np.ndarray) == False:
            self.value_positions = self.compute_standard_value_positions()
        elif self.output_dim > 1 and isinstance(value_positions, np.ndarray) == False:
            raise ValueError(
                "If the dimensionality of the output space is > 1, the value positions have to be given to the FVGP class. EXIT"
            )
        else:
            self.value_positions = value_positions
        ##########################################
        #######prepare variances##################
        ##########################################
        if variances is None:
            self.variances = np.zeros((values.shape))
        else:
            self.variances = variances
        ######################################
        #####transform to index set###########
        ######################################
        self.data_x, self.data_y, self.variances = self.transform_index_set()
        self.point_number = len(self.data_x)
        self.compute_prior_fvGP_pdf()
    ###################################################################################
    ###################################################################################
    ###################################################################################
    ###################################################################################
    ###################################################################################
    #################TRAINING##########################################################
    ###################################################################################
    def stop_training(self):
        print("Cancelling asynchronous training.")
        try: self.opt.cancel_tasks()
        except:
            print("No asynchronous training to be cancelled, no training is running.")
    ###################################################################################
    def train(self,
        hyperparameter_bounds,
        init_hyperparameters = None,
        optimization_method = "global",
        optimization_dict = None,
        optimization_pop_size = 20,
        optimization_tolerance = 0.1,
        optimization_max_iter = 120,
        compute_inverse = None,
        dask_client = False):
        """
        This function finds the maximum of the log_likelihood and therefore trains the fvGP.
        This can be done on a remote cluster/computer by giving a dask client

        inputs:
            hyperparameter_bounds (2d list)
        optional inputs:
            init_hyperparameters (list):  default = None
            optimization_method : default = "global","global"/"local"/"hgdl"/callable f(obj,optimization_dict)
            optimization_dict: if optimization is callable, the this will be passed as dict
            optimization_pop_size: default = 20,
            optimization_tolerance: default = 0.1,
            optimization_max_iter: default = 120,
            dask_client: True/False/dask client, default = False

        output:
            None, just updates the class with new hyperparameters
        """
        ############################################
        if dask_client is True: dask_client = distributed.Client()
        self.hyperparameter_optimization_bounds = hyperparameter_bounds
        if init_hyperparameters is None:
            init_hyperparameters = self.hyperparameters
        if compute_inverse is not None: self.compute_inverse = compute_inverse
        print("fvGP training started with ",len(self.data_x)," data points")
        ######################
        #####TRAINING#########
        ######################
        self.hyperparameters = list(self.optimize_log_likelihood(
            init_hyperparameters,
            self.hyperparameter_optimization_bounds,
            optimization_method,
            optimization_dict,
            optimization_max_iter,
            optimization_pop_size,
            optimization_tolerance,
            dask_client
        ))

        self.compute_prior_fvGP_pdf()
        ######################
        ######################
        ######################
    ##################################################################################
    def update_hyperparameters(self):
        try:
            res = self.opt.get_latest(1)
            self.hyperparameters = res["x"][0]
            self.compute_prior_fvGP_pdf()
            print("Async hyper-parameter update successful")
            print("Latest hyper-parameters: ", self.hyperparameters)
        except:
            print("Async Hyper-parameter update not successful. I am keeping the old ones.")
            print("That probbaly means you are not optimizing them asynchronously")
            print("hyper-parameters: ", self.hyperparameters)
    ##################################################################################
    def optimize_log_likelihood(
        self,
        starting_hps,
        hp_bounds,
        optimization_method,
        optimization_dict,
        optimization_max_iter,
        likelihood_pop_size,
        optimization_tolerance,
        dask_client):

        epsilon = np.inf
        step_size = 1.0
        print(
            "Hyper-parameter tuning in progress. Old hyper-parameters: ",
            starting_hps, " with old log likelihood: ",
            self.log_likelihood(starting_hps))

        ############################
        ####start of optimization:##
        ############################
        if optimization_method == "global":
            print("I am performing a global differential evolution algorithm to find the optimal hyper-parameters.")
            res = differential_evolution(
                self.log_likelihood,
                hp_bounds,
                disp=True,
                maxiter=optimization_max_iter,
                popsize = likelihood_pop_size,
                tol = optimization_tolerance,
            )
            hyperparameters1 = np.array(res["x"])
            Eval1 = self.log_likelihood(hyperparameters1)
            hyperparameters = np.array(hyperparameters1)
            print("I found hyper-parameters ",hyperparameters," with likelihood ",
                self.log_likelihood(hyperparameters))
        elif optimization_method == "local":
            hyperparameters = np.array(starting_hps)
            print("Performing a local update of the hyper parameters.")
            print("starting hyper-parameters: ", hyperparameters)
            print("Attempting a BFGS optimization.")
            print("maximum number of iterations: ", optimization_max_iter)
            print("termination tolerance: ", optimization_tolerance)
            print("bounds: ", hp_bounds)
            OptimumEvaluation = minimize(
                self.log_likelihood,
                hyperparameters,
                method="L-BFGS-B",
                jac=self.log_likelihood_gradient,
                bounds = hp_bounds,
                tol = optimization_tolerance,
                callback = None,
                options = {"maxiter": optimization_max_iter,
                    #"iprint" : 101,
                    }#,
                           #"ftol": optimization_tolerance}
                )

            if OptimumEvaluation["success"] == True:
                print(
                    "Local optimization successfully concluded with result: ",
                    OptimumEvaluation["fun"],
                )
                hyperparameters = OptimumEvaluation["x"]
            else:
                print("Optimization not successful.")
        elif optimization_method == "hgdl":
            print("HGDL optimization submitted")
            print('bounds are',hp_bounds)
            from hgdl.hgdl import HGDL
            try:
                res = self.opt.get_latest(10)
                x0 = res["x"][0:min(len(res["x"])-1,likelihood_pop_size)]
                print("starting with points from the last iteration")
            except Exception as err:
                print("starting with random points because")
                print(str(err))
                x0 = None
            self.opt = HGDL(self.log_likelihood,
                       self.log_likelihood_gradient,
                       self.log_likelihood_hessian,
                       hp_bounds,
                       number_of_walkers = likelihood_pop_size, 
                       maxEpochs = optimization_max_iter, verbose = False)
            self.opt.optimize(dask_client = dask_client, x0 = x0)
            res = self.opt.get_latest(10)
            while res["success"] == False:
                time.sleep(0.1)
                res = self.opt.get_latest(10)
            hyperparameters = res["x"][0]
        elif callable(optimization_method):
            hyperparameters = optimization_method(self,optimization_dict)
        else:
            raise ValueError("no optimization mode specified")
        ###################################################
        print("New hyper-parameters: ",
            hyperparameters,
            "with log likelihood: ",
            self.log_likelihood(hyperparameters))

        return hyperparameters
    ##################################################################################
    def log_likelihood(self,hyperparameters):
        """
        computes the marginal log-likelihood
        input:
            hyper parameters
        output:
            negative marginal log-likelihood (scalar)
        """
        mean = self.mean_function(self,self.data_x,hyperparameters)
        x,K = self._compute_covariance_value_product(hyperparameters,self.data_y, self.variances, mean)
        y = self.data_y - mean
        sign, logdet = self.slogdet(K)
        n = len(y)
        if sign == 0.0: return (0.5 * (y.T @ x)) + (0.5 * n * np.log(2.0*np.pi))
        return (0.5 * (y.T @ x)) + (0.5 * sign * logdet) + (0.5 * n * np.log(2.0*np.pi))
    ##################################################################################
    @staticmethod
    @nb.njit
    def numba_dL_dH(y, a, b, length):
        dL_dH = np.empty((length))
        for i in range(length):
                dL_dH[i] = 0.5 * ((y.T @ (a[i] @ b)) - (np.trace(a[i])))
        return dL_dH
    ##################################################################################
    def log_likelihood_gradient(self, hyperparameters):
        """
        computes the gradient of the negative marginal log-likelihood
        input:
            hyper parameters
        output:
            gradient of the negative marginal log-likelihood (vector)
        """
        mean = self.mean_function(self,self.data_x,hyperparameters)
        b,K = self._compute_covariance_value_product(hyperparameters,self.data_y, self.variances, mean)
        y = self.data_y - mean
        dK_dH = self.gradient_gp_kernel(self.data_x,self.data_x, hyperparameters)
        K = np.array([K,] * len(hyperparameters))
        a = self.solve(K,dK_dH)
        y = np.ascontiguousarray(y, dtype=np.float64)
        a = np.ascontiguousarray(a, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        #dL_dH = self.numba_dL_dH(y, a, b, len(hyperparameters))
        dL_dH = np.empty((len(hyperparameters)))
        for i in range(len(hyperparameters)):
            dL_dH[i] = 0.5 * ((y.T @ a[i] @ b) - (np.trace(a[i])))
        return -dL_dH
    ##################################################################################
    @staticmethod
    @nb.njit
    def numba_d2L_dH2(x, y, s, ss):
        len_hyperparameters = s.shape[0]
        d2L_dH2 = np.empty((len_hyperparameters,len_hyperparameters))
        for i in range(len_hyperparameters):
            x1 = s[i]
            for j in range(i+1):
                x2 = s[j]
                x3 = ss[i,j]
                f = 0.5 * ((y.T @ (-x2 @ x1 @ x - x1 @ x2 @ x + x3 @ x)) - np.trace(-x2 @ x1 + x3))
                d2L_dH2[i,j] = d2L_dH2[j,i] = f
        return d2L_dH2
    ##################################################################################
    def log_likelihood_hessian(self, hyperparameters):
        """
        computes the hessian of the negative  marginal  log-likelihood
        input:
            hyper parameters
        output:
            hessian of the negative marginal log-likelihood (matrix)
        """
        mean = self.mean_function(self,self.data_x,hyperparameters)
        x,K = self._compute_covariance_value_product(hyperparameters,self.data_y, self.variances, mean)
        y = self.data_y - mean
        dK_dH = self.gradient_gp_kernel(self.data_x,self.data_x, hyperparameters)
        d2K_dH2 = self.hessian_gp_kernel(self.data_x,self.data_x, hyperparameters)
        K = np.array([K,] * len(hyperparameters))
        s = self.solve(K,dK_dH)
        ss = self.solve(K,d2K_dH2)
        # make contiguous 
        K = np.ascontiguousarray(K, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        s = np.ascontiguousarray(s, dtype=np.float64)
        ss = np.ascontiguousarray(ss, dtype=np.float64)
        #d2L_dH2 = self.numba_d2L_dH2(x, y, s, ss)
        len_hyperparameters = s.shape[0]
        d2L_dH2 = np.empty((len_hyperparameters,len_hyperparameters))
        for i in range(len_hyperparameters):
            x1 = s[i]
            for j in range(i+1):
                x2 = s[j]
                x3 = ss[i,j]
                f = 0.5 * ((y.T @ (-x2 @ x1 @ x - x1 @ x2 @ x + x3 @ x)) - np.trace(-x2 @ x1 + x3))
                d2L_dH2[i,j] = d2L_dH2[j,i] = f

        return -d2L_dH2


    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################Compute#Covariance#Matrix###################################
    ##################################################################################
    ##################################################################################
    def compute_prior_fvGP_pdf(self):
        """
        This function computes the important entities, namely the prior covariance and
        its product with the (values - prior_mean) and returns them and the prior mean
        input:
            none
        return:
            prior mean
            prior covariance
            covariance value product
        """
        self.prior_mean_vec = self.mean_function(self,self.data_x,self.hyperparameters)
        cov_y,K = self._compute_covariance_value_product(
                self.hyperparameters,
                self.data_y,
                self.variances,
                self.prior_mean_vec)
        self.prior_covariance = K
        if self.compute_inverse is True: self.prior_covariance_inverse = np.linalg.inv(K)
        self.covariance_value_prod = cov_y
    ##################################################################################
    def _compute_covariance_value_product(self, hyperparameters,values, variances, mean):
        K = self.compute_covariance(hyperparameters, variances)
        y = values - mean
        x = self.solve(K, y)
        #Jacobi preconditioned system:
        #P = np.identity(len(K)) / K
        #yy = np.linalg.solve(K@P,y)
        #x = np.linalg.solve(P,yy)
        return x[:,0],K
        #return x,K
    ##################################################################################
    def compute_covariance(self, hyperparameters, variances):
        """computes the covariance matrix from the kernel"""
        CoVariance = self.kernel(
            self.data_x, self.data_x, hyperparameters, self)
        self.add_to_diag(CoVariance, variances)
        return CoVariance

    def slogdet(self, A):
        """
        fvGPs slogdet method based on torch
        """
        #s,l = np.linalg.slogdet(A)
        #return s,l
        if self.compute_device == "cpu":
            A = torch.from_numpy(A)
            sign, logdet = torch.slogdet(A)
            sign = sign.numpy()
            logdet = logdet.numpy()
            logdet = np.nan_to_num(logdet)
            return sign, logdet
        elif self.compute_device == "gpu" or self.compute_device == "multi-gpu":
            A = torch.from_numpy(A).cuda()
            sign, logdet = torch.slogdet(A)
            sign = sign.cpu().numpy()
            #i = np.where(sign == -1)
            #sign[i] = 1
            logdet = logdet.cpu().numpy()
            logdet = np.nan_to_num(logdet)
            return sign, logdet

    def solve(self, A, b):
        """
        fvGPs slogdet method based on torch
        """
        #x = np.linalg.solve(A,b)
        #return x
        if b.ndim == 1: b = np.expand_dims(b,axis = 1)
        if self.compute_device == "cpu":
            #####for sparsity:
            if self.sparse == True:
                zero_indices = np.where(A < 1e-16)
                A[zero_indices] = 0.0
                if self.is_sparse(A):
                    try:
                        A = scipy.sparse.csr_matrix(A)
                        x = scipy.sparse.spsolve(A,b)
                        return x
                    except Exceprion as e:
                        print("Sparse solve did not work out.")
                        print("reason: ", str(e))
            ##################
            A = torch.from_numpy(A)
            b = torch.from_numpy(b)
            try:
                x, lu = torch.solve(b,A)
                #x = np.linalg.solve(A,b)
                return x.numpy()
            except Exception as e:
                try:
                    print("except statement invoked: torch.solve() on cpu did not work")
                    print("reason: ", str(e))
                    x, qr = torch.lstsq(b,A)
                except Exception as e:
                    print("except statement 2 invoked: torch.solve() and torch.lstsq() on cpu did not work")
                    print("falling back to numpy.lstsq()")
                    print("reason: ", str(e))
                    x,res,rank,s = np.linalg.lstsq(A.numpy(),b.numpy())
                    return x
            return x.numpy()
        elif self.compute_device == "gpu" or A.ndim < 3:
            A = torch.from_numpy(A).cuda()
            b = torch.from_numpy(b).cuda()
            try:
                x, lu = torch.solve(b,A)
            except Exception as e:
                print("except statement invoked: torch.solve() on gpu did not work")
                print("reason: ", str(e))
                try:
                    x, qr = torch.lstsq(b,A)
                except Exception as e:
                    print("except statement 2 invoked: torch.solve() and torch.lstsq() on gpu did not work")
                    print("falling back to numpy.lstsq()")
                    print("reason: ", str(e))
                    x,res,rank,s = np.linalg.lstsq(A.numpy(),b.numpy())
                    return x
            return x.cpu().numpy()
        elif self.compute_device == "multi-gpu":
            n = min(len(A), torch.cuda.device_count())
            split_A = np.array_split(A,n)
            split_b = np.array_split(b,n)
            results = []
            for i, (tmp_A,tmp_b) in enumerate(zip(split_A,split_b)):
                cur_device = torch.device("cuda:"+str(i))
                tmp_A = torch.from_numpy(tmp_A).cuda(cur_device)
                tmp_b = torch.from_numpy(tmp_b).cuda(cur_device)
                results.append(torch.solve(tmp_b,tmp_A)[0])
            total = results[0].cpu().numpy()
            for i in range(1,len(results)):
                total = np.append(total, results[i].cpu().numpy(), 0)
            return total
    ##################################################################################
    def add_to_diag(self,Matrix, Vector):
        d = np.einsum("ii->i", Matrix)
        d += Vector
        return Matrix
    def is_sparse(self,A):
        if float(np.count_nonzero(A))/float(len(A)**2) < 0.01: return True
        else: return False
    def how_sparse_is(self,A):
        return float(np.count_nonzero(A))/float(len(A)**2)

    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###############################gp prediction###############################
    ###########################################################################
    ###########################################################################
    def posterior_mean(self, x_iset):
        """
        function to compute the posterior mean
        input:
        ------
            x_iset: 2d numpy array of points, note, these are elements of the 
            index set which results from a cartesian product of input and output space
        output:
        -------
            {"x":    the input points,
             "f(x)": the posterior mean vector (1d numpy array)}
        """
        p = np.array(x_iset)
        if x_iset.ndim < 2: print("x_iset has to be given as a 2d numpy array: [[x1],[x2],...]"); input()
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])
        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        kk = self.kernel(p, p,self.hyperparameters,self)
        A = k.T @ self.covariance_value_prod
        posterior_mean = self.mean_function(self,p,self.hyperparameters) + A
        return {"x": p,
                "f(x)": posterior_mean}
    ###########################################################################
    def posterior_covariance(self, x_iset):
        """
        function to compute the posterior covariance
        input:
        ------
            x_iset: 2d numpy array of points, note, these are elements of the 
            index set which results from a cartesian product of input and output space
        output:
        -------
            {"x":    the index set points,
             "v(x)": the posterior variances (1d numpy array) for each input point,
             "S":    covariance matrix, v(x) = diag(S)}
        """
        p = np.array(x_iset)
        if x_iset.ndim < 2: print("x_iset has to be given as a 2d numpy array: [[x1],[x2],...]")
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        kk = self.kernel(p, p,self.hyperparameters,self)
        if self.compute_inverse is True:
            a = kk - (k.T @ self.prior_covariance_inverse @ k)
        else:
            k_cov_prod = self.solve(self.prior_covariance,k)
            a = kk - (k_cov_prod.T @ k)
        diag = np.diag(a)
        diag = np.where(diag<0.0,0.0,diag)
        if any([x < -0.001 for x in np.diag(a)]):
            print("CAUTION, negative variances encountered. That normally means that the model is unstable.")
            print("Rethink the kernel definitions, add more noise to the data,")
            print("or double check the hyper-parameter optimization bounds. This will not ")
            print("terminate the algorithm, but expect anomalies.")
            print("diagonal of the posterior covariance: ",np.diag(a))
            p = np.block([[self.prior_covariance, k],[k.T, kk]])
            print("eigenvalues of the prior: ", np.linalg.eig(p)[0])

        np.fill_diagonal(a,diag)
        return {"x": p,
                "v(x)": np.diag(a),
                "S": a}
    ###########################################################################
    def gp_prior(self, x_iset):
        """
        function to compute the data-informed prior
        input:
        ------
            x_iset: 2d numpy array of points, note, these are elements of the 
            index set which results from a cartesian product of input and output space
        output:
        -------
            {"x": the index set points,
             "K": covariance matrix between data points
             "k": covariance between data and requested poins,
             "kappa": covariance matrix between requested points,
             "prior mean": the mean of the prior
             "S:": joint prior covariance}
        """
        p = np.array(x_iset)
        if x_iset.ndim < 2: print("x_iset has to be given as a 2d numpy array: [[x1],[x2],...]")
        if len(p[0]) != len(self.data_x[0]): p = np.column_stack([p,np.zeros((len(p)))])

        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        kk = self.kernel(p, p,self.hyperparameters,self)
        post_mean = np.empty((len(x_iset)))
        post_mean = self.mean_function(self,x_iset, self.hyperparameters)
        full_gp_prior_mean = np.append(self.prior_mean_vec, post_mean)
        return  {"x": p,
                 "K": self.prior_covariance,
                 "k": k,
                 "kappa": kk,
                 "prior mean": full_gp_prior_mean,
                 "S": np.block([[self.prior_covariance, k],[k.T, kk]])}

    ###########################################################################
    def entropy(self, S):
        """
        function comuting the entropy of a normal distribution
        res = entropy(S); S is a 2d numpy array, matrix has to be non-singular
        """
        dim  = len(S[0])
        s, logdet = self.slogdet(S)
        return (float(dim)/2.0) +  ((float(dim)/2.0) * np.log(2.0 * np.pi)) + (0.5 * s * logdet)
    ###########################################################################
    def gp_entropy(self, x_iset):
        """
        function comuting the entropy of a normal distribution
        a = entropy(S); S is a 2d numpy array, matrix has to be non-singular
        """
        priors = self.gp_prior(x_iset)
        S = priors["S"]
        dim  = len(S[0])
        s, logdet = self.slogdet(S)
        return (float(dim)/2.0) +  ((float(dim)/2.0) * np.log(2.0 * np.pi)) + (0.5 * s * logdet)
    ###########################################################################

    def kl_div(self,mu1, mu2, S1, S2):
        """
        function comuting the KL divergence between two normal distributions
        a = kl_div(mu1, mu2, S1, S2); S1, S2 are a 2d numpy arrays, matrices has to be non-singular
        mu1, mu2 are mean vectors, given as 2d arrays
        """
        s1, logdet1 = self.slogdet(S1)
        s2, logdet2 = self.slogdet(S2)
        x1 = self.solve(S2,S1)
        mu = np.subtract(mu2,mu1)
        x2 = self.solve(S2,mu)
        dim = len(mu)
        kld = 0.5 * (np.trace(x1) + (x2.T @ mu) - dim + ((s2*logdet2)-(s1*logdet1)))
        if kld < -1e-4: print("negative KL divergence encountered")
        return kld
    ###########################################################################
    def gp_kl_div(self, x_iset, comp_mean, comp_cov):
        """
        function to compute the kl divergence of a posterior at given points
        input:
        ------
            x_iset: 2d numpy array of points, note, these are elements of the 
            index set which results from a cartesian product of input and output space
        output:
        -------
            {"x": the index set points,
             "gp posterior mean": ,
             "gp posterior covariance":  ,
             "given mean": the user-provided mean vector,
             "given covariance":  the use_provided covariance,
             "kl-div:": the kl div between gp pdf and given pdf}
        """
        res = self.posterior_mean(x_iset)
        gp_mean = res["f(x)"]
        gp_cov = self.posterior_covariance(x_iset)["S"]

        return {"x": x_iset,
                "gp posterior mean" : gp_mean,
                "gp posterior covariance": gp_cov,
                "given mean": comp_mean,
                "given covariance": comp_cov,
                "kl-div": self.kl_div(gp_mean, comp_mean, gp_cov, comp_cov)}
    ###########################################################################
    def shannon_information_gain(self, x_iset):
        """
        function to compute the shannon-information gain of data
        input:
        ------
            x_iset: 2d numpy array of points, note, these are elements of the 
            index set which results from a cartesian product of input and output space
        output:
        -------
            {"x": the index set points,
             "prior entropy": prior entropy
             "posterior entropy": posterior entropy
             "sig:" shannon_information gain}
        """
        p = np.array(x_iset)
        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        kk = self.kernel(p, p,self.hyperparameters,self)


        full_gp_covariances = \
                np.asarray([np.block([[self.prior_covariance,k],\
                            [k.T,kk]])])

        e1 = self.entropy(self.prior_covariance)
        e2 = self.entropy(full_gp_covariances)
        sig = (e2 - e1)
        return {"x": p,
                "prior entropy" : e1,
                "posterior entropy": e2,
                "sig":sig}
    ###########################################################################
    def posterior_probability(self, x_iset, comp_mean, comp_cov):
        """
        function to compute the probability of an uncertain feature given the gp posterior
        input:
        ------
            x_iset: 2d numpy array of points, note, these are elements of the 
            index set which results from a cartesian product of input and output space
            comp_mean: a vector of mean values, same length as x_iset
            comp_cov: covarianve matrix, \in R^{len(x_iset)xlen(x_iset)}

        output:
        -------
            {"x": the index set points,
             "gp posterior mean": ,
             "gp posterior covariance":  ,
             "given mean": the user-provided mean vector,
             "given covariance":  the use_provided covariance,
             "kl-div:": the kl div between gp pdf and given pdf}
        """

        res = self.posterior_mean(x_iset)
        gp_mean = res["f(x)"]
        gp_cov = self.posterior_covariance(x_iset)["S"]
        gp_cov_inv = np.linalg.inv(gp_cov)
        comp_cov_inv = np.linalg.inv(comp_cov)
        cov = np.linalg.inv(gp_cov_inv + comp_cov_inv)
        mu =  cov @ gp_cov_inv @ gp_mean + cov @ comp_cov_inv @ comp_mean
        s1, logdet1 = self.slogdet(cov)
        s2, logdet2 = self.slogdet(gp_cov)
        s3, logdet3 = self.slogdet(comp_cov)
        dim  = len(mu)
        C = 0.5*(((gp_mean.T @ gp_cov_inv + comp_mean.T @ comp_cov_inv).T \
               @ cov @ (gp_cov_inv @ gp_mean + comp_cov_inv @ comp_mean))\
               -(gp_mean.T @ gp_cov_inv @ gp_mean + comp_mean.T @ comp_cov_inv @ comp_mean)).squeeze()
        ln_p = (C + 0.5 * logdet1) - (np.log((2.0*np.pi)**(dim/2.0)) + (0.5*(logdet2 + logdet3)))
        #print("posterior gp mean: ", gp_mean)
        #print("given mean: ", comp_mean)
        #print("posterior gp cov : ", gp_cov)
        #print("given cov        : ", comp_cov)
        #print("===========")
        #print("combined mean: ", mu)
        #print("combined cov:  ", cov)
        #print("integral gp g: ", self._int_gauss(gp_cov))
        #print("integral co g: ", self._int_gauss(comp_cov))
        #print("integral res g: ", self._int_gauss(cov))
        return {"mu": mu,
                "covariance": cov,
                "probability": 
                np.exp(ln_p)
                }
    ###########################################################################
    def _int_gauss(self,S):
        return ((2.0*np.pi)**(len(S)/2.0))*np.sqrt(np.linalg.det(S))


    def compute_posterior_fvGP_pdf(self, x_input, x_output=np.array([[0.0]]), mode = 'cartesian product',
            compute_entropies = False, compute_prior_covariances = False,
            compute_posterior_covariances = False, compute_means = True):
        """
        Function to compute the variances, covariances and means of the (joint) posterior.
        gp.compute_posterior_fvGP_pdf(x_input, x_output=None, mode = 'cartesian product',compute_entropy = True, compute_covariance = True, compute_mean = True)
        Parameters:
        -----------
            x_input (2d numpy array): a set of elements of the input set, these are coordinates in the input (parameter) space
        Optional:
        ---------
            x_output(2d numpy array):             a set of elements of the output set, these are coordinates in the output space, default: None
            mode (str):                           'cartesian product' or 'stack', tell the algorithm how input and output coordinates will be combined, if 'stack' row number of x_input and x_output has to be the same
            compute_entropies (bool):             True/False, can be turned off if not required for better computational performance, default: True
            compute_prior_covariances (bool):     True/False, can be turned off if not required for better computational performance, default: True
            compute_posterior_covariances (bool): True/False, can be turned off if not required for better computational performance, default: True
            compute_mean(bool):                   True/False, can be turned off if not required for better computational performance, default: True
        Returns:
        --------
        res = {"input points": p,
               "posterior means": mean,
               "posterior covariances": covariance,
               "prior covariances": full_gp_covariances,
               "prior means": None,
               "entropies": entropies
               }
        """
        ########################################
        ####computation we need for all returns:
        ########################################
        if len(x_input.shape) == 1: 
            raise ValueError("x_input is not a 2d numpy array, please see docstring for correct usage of gp.compute_posterior_fvGP_pdf() function, thank you")
        n_orig = len(x_input)
        tasks = len(x_output)
        if mode == 'cartesian product':
            new_points = np.zeros((len(x_input) * len(x_output), len(x_input[0]) + len(x_output[0])))
            counter = 0
            for element in itertools.product(x_input, x_output):
                new_points[counter] = np.concatenate([element[0], element[1]], axis=0)
                counter += 1   ###can't we append?
        elif mode == 'stack':
            new_points = np.column_stack([x_input,x_output])
        p = np.array(new_points)
        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        kk = self.kernel(p, p,self.hyperparameters,self)
        if compute_prior_covariances == True: 
            full_gp_covariances = \
                    np.asarray([np.block([[self.prior_covariance,k[:,i*tasks:(i+1)*tasks]],\
                             [k[:,i*tasks:(i+1)*tasks].T,kk[i*tasks:(i+1)*tasks,i*tasks:(i+1)*tasks]]])\
                    for i in range(n_orig)])
        else: full_gp_covariances = None
        if compute_entropies == True:
            entropies = []
            for i in range(n_orig):
                sgn, logdet = self.slogdet(np.block([[self.prior_covariance, 
                                              k[:,i*tasks:(i+1)*tasks]],[k[:,i*tasks:(i+1)*tasks].T, 
                                              kk[i*tasks:(i+1)*tasks,i*tasks:(i+1)*tasks]]]))
                if sgn == 0.0: entropies.append(0.0); print("entropy is zero, that should never happen. Double check your input!")
                else:entropies.append(sgn*logdet)
            entropies = np.asarray(entropies)
        else: entropies = None

        if compute_means == True:
            A = k.T @ self.covariance_value_prod
            posterior_mean = np.reshape(self.mean_function(self,p,self.hyperparameters) + A, (n_orig, len(x_output)))
        else:
            posterior_mean = None
        if compute_posterior_covariances == True:
            k_cov_prod = self.solve(self.prior_covariance,k)
            if self.compute_inverse is True:
                a = kk - (k.T @ self.prior_covariance_inverse @ k)
            else:
                k_cov_prod = self.solve(self.prior_covariance,k)
                a = kk - (k_cov_prod.T @ k)
            diag = np.diag(a)
            diag = np.where(diag<0.0,0.0,diag)
            if any([x < -0.001 for x in np.diag(a)]):
                print("CAUTION, negative variances encountered. That normally means that the model is unstable.")
                print("Rethink the kernel definitions, add more noise to the data,")
                print("or double check the hyper-parameter optimization bounds. This will not ")
                print("terminate the algorithm, but expect anomalies.")
                print("diagonal of the posterior covariance: ",np.diag(a))
                prior = np.block([[self.prior_covariance, k],[k.T, kk]])
                print("eigenvalues of the prior: ", np.linalg.eig(prior)[0])
                print("data points:")
                print(self.data_x)
                print("prediction points:")
                print(p)
                print("hyper parameters")
                print(self.hyperparameters)
                print(np.linalg.matrix_rank(prior), len(prior))
                print("===============")
                input()

            np.fill_diagonal(a,diag)
            covariance = np.asarray([
                a[i * tasks : (i + 1) * tasks, i * tasks : (i + 1) * tasks]
                for i in range(int(a.shape[0] / tasks))
            ])
        else:
            covariance = None
        res = {"input points": p,
               "posterior means": posterior_mean,
               "posterior covariances": covariance,
               "prior covariances": full_gp_covariances,
               "prior means": None,
               "entropies": entropies
               }
        return res
    ###########################################################################
    def compute_posterior_fvGP_pdf_gradient(self, x_input, x_output=np.array([[0.0]]), direction = 0, mode = 'cartesian product',
            compute_entropies = False, compute_prior_covariances = False,
            compute_posterior_covariances = False, compute_means = True):
        """
        Function to compute the variances, covariances and means of the (joint) posterior.
        gp.compute_posterior_fvGP_pdf_gradient(x_input, x_output=None, direction = 0, mode = 'cartesian product',compute_entropy = True, compute_covariance = True, compute_mean = True)
        Parameters:
        -----------
            x_input (2d numpy array): a set of elements of the input set, these are coordinates in the input (parameter) space
        Optional:
        ---------
            x_output(2d numpy array): a set of elements of the output set, these are coordinates in the output space, default: None
            direction (int): index of the derivative direction
            mode (str): 'cartesian product' or 'stack', tell the algorithm how input and output coordinates will be combined, if 'stack' row number of x_input and x_output has to be the same
            compute_entropies (bool):             True/False, can be turned off if not required for better computational performance, default: True
            compute_prior_covariances (bool):     True/False, can be turned off if not required for better computational performance, default: True
            compute_posterior_covariances (bool): True/False, can be turned off if not required for better computational performance, default: True
            compute_mean(bool):                   True/False, can be turned off if not required for better computational performance, default: True
        Returns:
        --------
        res = {"input points": p,
               "posterior mean gradients": mean,
               "posterior covariance gradients": covariance,
               "prior covariance gradients": full_gp_covariances,
               "prior mean gradients": None,
               "entropy gradients": entropies
               }
        """
        ####computation we need for all returns:
        if len(x_input.shape) == 1: raise ValueError("x_input is not a 2d numpy array, please see docstring for correct usage of gp.fvGP function, thank you")
        n_orig = len(x_input)
        tasks = len(x_output)
        if mode == 'cartesian product':
            new_points = np.zeros((len(x_input) * len(x_output), len(x_input[0]) + len(x_output[0])))
            counter = 0
            for element in itertools.product(x_input, x_output):
                new_points[counter] = np.concatenate([element[0], element[1]], axis=0)
                counter += 1   ###can't we append?
        elif mode == 'stack':
            new_points = np.column_stack([x_input,x_output])
        p = np.array(new_points)
        k = self.kernel(self.data_x,p,self.hyperparameters,self)
        k_g = self.d_kernel_dx(p,self.data_x, direction,self.hyperparameters).T
        kk =  self.kernel(p, p,self.hyperparameters,self)
        kk_g =  self.d_kernel_dx(p, p,direction,self.hyperparameters)
        if compute_prior_covariances == True: 
            full_gp_covariances = np.asarray([
                np.block([[self.prior_covariance, k_g[:,i*tasks:(i+1)*tasks]],\
                          [k_g[:,i*tasks:(i+1)*tasks].T, kk_g[i*tasks:(i+1)*tasks,i*tasks:(i+1)*tasks]]])\
                          for i in range(n_orig)])
        else: full_gp_covariances = None

        if compute_entropies == True:
            entropies = []
            for i in range(n_orig):
                Sigma = np.block([[self.prior_covariance, k[:,i*tasks:(i+1)*tasks]],  [\
                                k[:,i*tasks:(i+1)*tasks].T,   kk[i*tasks:(i+1)*tasks,i*tasks:(i+1)*tasks]]])
                Sigma_d = np.block([[self.prior_covariance, k_g[:,i*tasks:(i+1)*tasks]],\
                                    [k_g[:,i*tasks:(i+1)*tasks].T, kk_g[i*tasks:(i+1)*tasks,i*tasks:(i+1)*tasks]]])
                entropies.append(np.trace(self.solve(Sigma,Sigma_d)))
            entropies = np.asarray(entropies)
        else: entropies = None

        if compute_means == True:
            A = k_g.T @ self.covariance_value_prod
            mean = np.reshape(A, (n_orig, len(x_output)))
        else:
            mean = None
        if compute_posterior_covariance == True:
            if self.compute_inverse is True:
                a = kk - ((k.T @ self.prior_covariance_inverse @ k)+(k_g.T @ self.prior_covariance_inverse @ k_g))
            else:
                k_covariance_prod = self.solve(self.prior_covariance,k)
                kg_covariance_prod = self.solve(self.prior_covariance,k_g)
                a = kk_g - ((k_covariance_prod.T @ k) + (k_g_covariance_prod.T @ k))

            covariance = [
                a[i * tasks : (i + 1) * tasks, i * tasks : (i + 1) * tasks]
                for i in range(int(a.shape[0] / tasks))
                ]
        else:
            covariance = None

        res = {"input points": p,
               "posterior mean gradients": mean,
               "posterior covariance gradients": covariance,
               "prior covariance gradients": full_gp_covariances,
               "prior mean gradients": None,
               "entropy gradients": entropies
               }
        return res

    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################Kernels#####################################################
    ##################################################################################
    ##################################################################################
    def squared_exponential_kernel(self, distance, length):
        """
        function for the squared exponential kernel

        Parameters:
        -----------
            distance (float): scalar or numpy array with distances
            length (float):   length scale (note: set to one if length scales are accounted for by the distance matrix)
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """
        kernel = np.exp(-(distance ** 2) / (2.0 * (length ** 2)))
        return kernel

    def exponential_kernel(self, distance, length):
        """
        function for the exponential kernel

        Parameters:
        -----------
            distance (float): scalar or numpy array with distances
            length (float):   length scale (note: set to one if length scales are accounted for by the distance matrix)
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """

        kernel = np.exp(-(distance) / (length))
        return kernel

    def matern_kernel_diff1(self, distance, length):
        """
        function for the matern kernel  1. order differentiability

        Parameters:
        -----------
            distance (float): scalar or numpy array with distances
            length (float):   length scale (note: set to one if length scales are accounted for by the distance matrix)
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """

        kernel = (1.0 + ((np.sqrt(3.0) * distance) / (length))) * np.exp(
            -(np.sqrt(3.0) * distance) / length
        )
        return kernel

    def matern_kernel_diff2(self, distance, length):
        """
        function for the matern kernel  2. order differentiability

        Parameters:
        -----------
            distance (float): scalar or numpy array with distances
            length (float):   length scale (note: set to one if length scales are accounted for by the distance matrix)
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """

        kernel = (
            1.0
            + ((np.sqrt(5.0) * distance) / (length))
            + ((5.0 * distance ** 2) / (3.0 * length ** 2))
        ) * np.exp(-(np.sqrt(5.0) * distance) / length)
        return kernel

    def sparse_kernel(self, distance, radius):
        """
        function for the sparse kernel
        this kernel is compactly supported, which makes the covariance matrix sparse

        Parameters:
        -----------
            distance (float): scalar or numpy array with distances
            radius (float):   length scale (note: set to one if length scales are accounted for by the distance matrix)
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """

        d = np.array(distance)
        d[d == 0.0] = 10e-6
        d[d > radius] = radius
        kernel = (np.sqrt(2.0)/(3.0*np.sqrt(np.pi)))*\
        ((3.0*(d/radius)**2*np.log((d/radius)/(1+np.sqrt(1.0 - (d/radius)**2))))+\
        ((2.0*(d/radius)**2+1.0)*np.sqrt(1.0-(d/radius)**2)))
        return kernel

    def periodic_kernel(self, distance, length, p):
        """periodic kernel
        Parameters:
        -----------
            distance: float or array containing distances
            length (float): the length scale
            p (float): period of the oscillation
        Return:
        -------
            a structure of the she shape of the distance input parameter
        """
        kernel = np.exp(-(2.0/length**2)*(np.sin(np.pi*distance/p)**2))
        return kernel

    def linear_kernel(self, x1,x2, hp1,hp2,hp3):
        """
        function for the linear kernel in 1d

        Parameters:
        -----------
            x1 (float):  scalar position of point 1
            x2 (float):  scalar position of point 2
            hp1 (float): vertical offset of the linear kernel
            hp2 (float): slope of the linear kernel
            hp3 (float): horizontal offset of the linear kernel
        Return:
        -------
            scalar
        """
        kernel = hp1 + (hp2*(x1-hp3)*(x2-hp3))
        return kernel

    def dot_product_kernel(self, x1,x2,hp,matrix):
        """
        function for the dot-product kernel

        Parameters:
        -----------
            x1 (2d numpy array of points):  scalar position of point 1
            x2 (2d numpy array of points):  scalar position of point 2
            hp (float):                     vertical offset
            matrix (2d array of len(x1)):   a metric tensor to define the dot product
        Return:
        -------
            numpy array of shape len(x1) x len(x2)
        """
        kernel = hp + x1.T @ matrix @ x2
        return kernel

    def polynomial_kernel(self, x1, x2, p):
        """
        function for the polynomial kernel

        Parameters:
        -----------
            x1 (2d numpy array of points):  scalar position of point 1
            x2 (2d numpy array of points):  scalar position of point 2
            p (float):                      exponent
        Return:
        -------
            numpy array of shape len(x1) x len(x2)
        """
        kernel = (1.0+x1.T @ x2)**p
        return p
    def default_kernel(self,x1,x2,hyperparameters,obj):
        ################################################################
        ###standard anisotropic kernel in an input space with l2########
        ################################################################
        """
        x1: 2d numpy array of points
        x2: 2d numpy array of points
        obj: object containing kernel definition

        Return:
        -------
        Kernel Matrix
        """
        hps = hyperparameters
        distance_matrix = np.zeros((len(x1),len(x2)))
        for i in range(len(hps)-1):
            distance_matrix += abs(np.subtract.outer(x1[:,i],x2[:,i])/hps[1+i])**2
        distance_matrix = np.sqrt(distance_matrix)
        return   hps[0] *  obj.exponential_kernel(distance_matrix,1)

    def _compute_distance_matrix_l2(self,points1,points2,hp_list):
        """computes the distance matrix for the l2 norm"""
        distance_matrix = np.zeros((len(points2), len(points1)))
        for i in range(len(points1[0])):
            distance_matrix += (
            np.abs(
            np.subtract.outer(points2[:, i], points1[:, i]) ** 2
            )/hp_list[i])
        return np.sqrt(distance_matrix)
    
    def _compute_distance_matrix_l1(self,points1,points2):
        """computes the distance matrix for the l1 norm"""
        distance_matrix = (
        np.abs(
        np.subtract.outer(points2, points1)
        ))
        return distance_matrix

    def d_gp_kernel_dx(self, points1, points2, direction, hyperparameters):
        new_points = np.array(points1)
        epsilon = 1e-6
        new_points[:,direction] += epsilon
        a = self.kernel(new_points, points2, hyperparameters,self)
        b = self.kernel(points1,    points2, hyperparameters,self)
        derivative = ( a - b )/epsilon
        return derivative

    def d_gp_kernel_dh(self, points1, points2, direction, hyperparameters):
        new_hyperparameters1 = np.array(hyperparameters)
        new_hyperparameters2 = np.array(hyperparameters)
        epsilon = 1e-6
        new_hyperparameters1[direction] += epsilon
        new_hyperparameters2[direction] -= epsilon
        a = self.kernel(points1, points2, new_hyperparameters1,self)
        b = self.kernel(points1, points2, new_hyperparameters2,self)
        derivative = ( a - b )/(2.0*epsilon)
        return derivative

    def gradient_gp_kernel(self, points1, points2, hyperparameters):
        gradient = np.empty((len(hyperparameters), len(points1),len(points2)))
        for direction in range(len(hyperparameters)):
            gradient[direction] = self.d_gp_kernel_dh(points1, points2, direction, hyperparameters)
        return gradient

    def d2_gp_kernel_dh2(self, points1, points2, direction1, direction2, hyperparameters):
        ###things to consider then things go south with the Hessian:
        ###make sure the epsilon is appropriate, not too large, not too small, 1e-3 seems alright
        epsilon = 1e-3
        new_hyperparameters1 = np.array(hyperparameters)
        new_hyperparameters2 = np.array(hyperparameters)
        new_hyperparameters3 = np.array(hyperparameters)
        new_hyperparameters4 = np.array(hyperparameters)

        new_hyperparameters1[direction1] = new_hyperparameters1[direction1] + epsilon
        new_hyperparameters1[direction2] = new_hyperparameters1[direction2] + epsilon

        new_hyperparameters2[direction1] = new_hyperparameters2[direction1] + epsilon
        new_hyperparameters2[direction2] = new_hyperparameters2[direction2] - epsilon

        new_hyperparameters3[direction1] = new_hyperparameters3[direction1] - epsilon
        new_hyperparameters3[direction2] = new_hyperparameters3[direction2] + epsilon

        new_hyperparameters4[direction1] = new_hyperparameters4[direction1] - epsilon
        new_hyperparameters4[direction2] = new_hyperparameters4[direction2] - epsilon

        return (self.kernel(points1,points2,new_hyperparameters1,self) \
              - self.kernel(points1,points2,new_hyperparameters2,self) \
              - self.kernel(points1,points2,new_hyperparameters3,self) \
              + self.kernel(points1,points2,new_hyperparameters4,self))\
              / (4.0*(epsilon**2))
#    @profile
    def hessian_gp_kernel(self, points1, points2, hyperparameters):
        hessian = np.zeros((len(hyperparameters),len(hyperparameters), len(points1),len(points2)))
        for i in range(len(hyperparameters)):
            for j in range(i+1):
                hessian[i,j] = hessian[j,i] = self.d2_gp_kernel_dh2(points1, points2, i,j, hyperparameters)
        return hessian

    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ######################OTHER#######################################################
    ##################################################################################
    ##################################################################################

    def compute_standard_value_positions(self):
        value_pos = np.zeros((self.point_number, self.output_num, self.output_dim))
        for j in range(self.output_num):
            value_pos[:, j, :] = j
        return value_pos

    def transform_index_set(self):
        new_points = np.zeros((self.point_number * self.output_num, self.iset_dim))
        new_values = np.zeros((self.point_number * self.output_num))
        new_variances = np.zeros((self.point_number * self.output_num))
        for i in range(self.output_num):
            new_points[i * self.point_number : (i + 1) * self.point_number] = \
            np.column_stack([self.data_x, self.value_positions[:, i, :]])
            new_values[i * self.point_number : (i + 1) * self.point_number] = \
            self.data_y[:, i]
            new_variances[i * self.point_number : (i + 1) * self.point_number] = \
            self.variances[:, i]
        return new_points, new_values, new_variances

    def default_mean_function(self,gp_obj,x,hyperparameters):
        """evaluates the gp mean function at the data points """
        mean = np.zeros((len(x)))
        mean[:] = np.mean(self.data_y)
        return mean
###########################################################################
###########################################################################
###########################################################################
#######################################Testing#############################
###########################################################################
###########################################################################
    def test_derivatives(self):
        print("====================================")
        print("====================================")
        print("====================================")
        res = self.compute_posterior_fvGP_pdf(np.array([[0.20,0.15]]),self.value_positions[-1])
        a1 = res["means"]
        b1 = res["covariances"]

        res = self.compute_posterior_fvGP_pdf(np.array([[0.200001,0.15]]),self.value_positions[-1])
        a2 = res["means"]
        b2 = res["covariances"]

        print("====================================")
        res = self.compute_posterior_fvGP_pdf_gradient(np.array([[0.2,0.15]]),self.value_positions[-1],direction = 0)
        print("function values: ",a1,a2)
        print("gradient:")
        print((a2-a1)/0.000001,res["means grad"])
        print("variances:")
        print((b2-b1)/0.000001,res["covariances grad"])
        print("compare values and proceed with enter")
        input()

###########################################################################
#######################################MISC################################
###########################################################################
    def gradient(self, function, point, epsilon = 10e-3,*args):
        """
        This function calculates the gradient of a function by using finite differences

        Extended description of function.

        Parameters:
        function (function object): the function the gradient should be computed of
        point (numpy array 1d): point at which the gradient should be computed

        optional:
        epsilon (float): the distance used for the evaluation of the function

        Returns:
        numpy array of gradient

        """
        gradient = np.zeros((len(point)))
        for i in range(len(point)):
            new_point = np.array(point)
            new_point[i] = new_point[i] + epsilon
            gradient[i] = (function(new_point,args) - function(point,args))/ epsilon
        return gradient
    def hessian(self, function, point, epsilon = 10e-3, *args):
        """
        This function calculates the hessian of a function by using finite differences

        Extended description of function.

        Parameters:
        function (function object): the function, the hessian should be computed of
        point (numpy array 1d): point at which the gradient should be computed

        optional:
        epsilon (float): the distance used for the evaluation of the function

        Returns:
        numpy array of hessian

        """
        hessian = np.zeros((len(point),len(point)))
        for i in range(len(point)):
            for j in range(len(point)):
                new_point1 = np.array(point)
                new_point2 = np.array(point)
                new_point3 = np.array(point)
                new_point4 = np.array(point)

                new_point1[i] = new_point1[i] + epsilon
                new_point1[j] = new_point1[j] + epsilon

                new_point2[i] = new_point2[i] + epsilon
                new_point2[j] = new_point2[j] - epsilon

                new_point3[i] = new_point3[i] - epsilon
                new_point3[j] = new_point3[j] + epsilon

                new_point4[i] = new_point4[i] - epsilon
                new_point4[j] = new_point4[j] - epsilon

                hessian[i,j] = \
                (function(new_point1,args) - function(new_point2,args) - function(new_point3,args) +  function(new_point4,args))\
                / (4.0*(epsilon**2))
        return hessian

###########################################################################
###################################END#####################################
###########################################################################
