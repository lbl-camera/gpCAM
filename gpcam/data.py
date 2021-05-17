import time
from time import sleep
import random
import numpy as np
from random import seed
from random import randint
import itertools
import math
import uuid
import time
import datetime

class gpData:
    def __init__(self, dim, parameter_bounds, function, init_dataset_size = None, output_number = None, output_dim = None, append = False):
        self.function = function
        self.dim = dim
        self.parameter_bounds = parameter_bounds
        self.init_dataset_size = init_dataset_size
        self.append = append
        self.dataset = []
        self.output_number = output_number
        self.output_dim =output_dim

    ###############################################################
    #either create random data or commubicate a data set (use translate2data for numpy arrays)
    def create_random_init_dataset(self):
        self.x = self._create_random_points()
        self.point_number = len(self.x)
        self.x, self.y, self.v = self.add_data_points(self.x)

    def comm_init_dataset(self,data):
        self.dataset = data
        self.point_number = len(data)
        self.x = self.extract_points_from_data()
        self.y = self.extract_values_from_data()
        self.v = self.extract_variances_from_data()
        self.times = self.extract_times_from_data()
        self.measurement_costs = self.extract_costs_from_data()

    ###############################################################
    def add_data_points(self, new_points, post_var = None, hps = None):
        """
        adds points to data and asks the instrument to get values and variances
        """
        new_data_list = self.translate2data(new_points,post_var = post_var, hps = hps)
        self._get_data_from_instrument(new_data_list, self.append)
        self.dataset = self.clean_data_NaN(self.dataset)
        self.x = self.extract_points_from_data()
        self.y = self.extract_values_from_data()
        self.v = self.extract_variances_from_data()
        self.times = self.extract_times_from_data()
        self.measurement_costs = self.extract_costs_from_data()
        return self.x, self.y, self.v   
    ###############################################################
    def translate2data(self, x, y = None, v = None, post_var = None ,hps = None):
        """
        translates numpy arrays to the data format
        """
        data = []
        for i in range(len(x)):
            data.append(self.npy2data(x[i],post_var,hps))
            if y is not None: data[i]["value"] = y[i]
            if v is not None: data[i]["variance"] = v[i]
        return data
    ###############################################################
    def npy2data(self, x,post_var = None, hps = None, cost = None):
        """
        parameters:
        -----------
        x ... 1d numpy array
        y ... float
        v ... float
        post_var ... float
        """
        d = {}
        d["position"] = x
        d["value"] = None
        d["variance"] = None
        d["cost"] = cost
        d["id"] = str(uuid.uuid4())
        d["time stamp"] = time.time()
        d["date time"] = datetime.datetime.now().strftime("%d/%m/%Y_%H:%M%S")
        d["measured"] = False
        d["posterior variance"] = post_var
        d["hyperparameters"] = hps
        return d
    ###############################################################
    def _get_data_from_instrument(self, new_data, append = False):
        if append:
            self.dataset = self.dataset + new_data
            self.dataset = self.function(self.dataset)
        else:
            new_data = self.function(new_data)
            self.dataset = self.dataset + new_data
    ###############################################################
    #def data2npy(self,data_entry):
    #    """
    #    takes an entry in the data list and returns point, val,var
    #    """
    #    x = data_entry["position"]
    #    y = data_entry["value"]
    #    v = data_entry["variance"]
    #    return x,y,v
    ###############################################################
    ###Printing####################################################
    ###############################################################
    def print_data(self, data):
        np.set_printoptions(precision=5)
        for idx in range(len(data)):
            print(idx, " ==> ")
            for key in list(data[idx]):
                if isinstance(data[idx][key], dict):
                    print("     ", key, " :")
                    for key2 in data[idx][key]:
                        print("          ", key2, " : ", data[idx][key][key2])
                else:
                    print("     ", key, " : ", data[idx][key])
    ################################################################
    ########Extracting##############################################
    ################################################################
    def extract_points_from_data(self):
        P = np.zeros((self.point_number, self.dim))
        for idx_data in range(self.point_number):
            P[idx_data] = self.dataset[idx_data]["position"]
        return P

    def extract_values_from_data(self):
        M= np.zeros((self.point_number))
        for idx_data in range(self.point_number):
            M[idx_data] = self.dataset[idx_data]["value"]
        return M

    def extract_variances_from_data(self):
        Variance = np.zeros((self.point_number))
        for idx_data in range(self.point_number):
            if self.dataset[idx_data]["variance"] is None: return None
            Variance[idx_data] = self.dataset[idx_data]["variance"]
        return Variance

    def extract_costs_from_data(self):
        Costs = []
        for idx in range(self.point_number):
            Costs.append(self.dataset[idx]["cost"])
        return Costs

    def extract_times_from_data(self):
        times = np.zeros((self.point_number))
        for idx_data in range(self.point_number):
            times[idx_data] = self.dataset[idx_data]["time stamp"]
        return times

    ###############################################################
    #######Creating################################################
    ###############################################################
    def _create_random_x(self):
        return np.random.uniform(low = self.parameter_bounds[:,0],
                                 high =self.parameter_bounds[:,1],
                                 size = self.dim)
    def _create_random_points(self):
        x = np.empty((self.init_dataset_size,self.dim))
        for i in range(self.init_dataset_size):
            x[i,:] = self._create_random_x()
        return x
    ###############################################################
    #########Cleaning##############################################
    ###############################################################
    def clean_data_NaN(self,data):
        for entry in data:
            if self._nan_in_dict(entry):
                print("CAUTION, NaN detected in data")
                data.remove(entry)
        return data

    def _nan_in_dict(self,dictionary):
        is_nan = False
        try:
            for key in dictionary:
                if type(dictionary[key]) is dict:
                    is_nan = self._nan_in_dict(dictionary[key])
                elif type(dictionary[key]) is float and math.isnan(dictionary[key]):
                    is_nan = True
                elif type(dictionary[key]) is np.ndarray and any(np.isnan(dictionary[key])):
                    is_nan = True
                elif type(dictionary[key]) is list and any(np.isnan(np.asarray(dictionary[key]))):
                    is_nan = True
                else:
                    is_nan = False
        except:
            pass
        return is_nan
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################

class fvgpData(gpData):
    def create_random_init_dataset(self):
        if self.output_number is None or self.output_dim is None:
            raise Exception("When initializing the data class for a multi-output GP, please provide output_number AND an output_dim parameters.")

        self.x = self._create_random_points()
        self.point_number = len(self.x)
        self.x, self.y,self.v, self.vp = self.add_data_points(self.x)

    def comm_init_dataset(self,data):
        if self.output_number is None or self.output_dim is None:
            raise Exception("When initializing the data class for a multi-output GP, please provide output_number AND an output_dim parameters.")

        self.dataset = data
        self.point_number = len(data)
        self.output_number = len(data[0]["variances"])
        self.vp = self.extract_value_positions_from_data()
        self.x = self.extract_points_from_data()
        self.y = self.extract_values_from_data()
        self.v = self.extract_variances_from_data()
        self.times = self.extract_times_from_data()
        self.measurement_costs = self.extract_costs_from_data()
        self.output_number = len(self.v[0])
        self.output_dim = len(self.vp[0,0])

    def add_data_points(self, new_points, post_var = None, hps = None):
        """
        adds points to data and asks the instrument to get values, variances and value positions
        """
        new_data_list = self.translate2data(new_points,post_var = post_var, hps = hps)
        self._get_data_from_instrument(new_data_list, self.append)
        self.dataset = self.clean_data_NaN(self.dataset)
        self.x = self.extract_points_from_data()
        self.y = self.extract_values_from_data()
        self.v = self.extract_variances_from_data()
        self.times = self.extract_times_from_data()
        self.measurement_costs = self.extract_costs_from_data()
        self.vp = self.extract_value_positions_from_data()
        return self.x, self.y, self.v, self.vp

    def translate2data(self, x, y = None, v = None, vp = None,post_var = None ,hps = None):
        """
        translates numpy arrays to the data format
        """
        data = []
        for i in range(len(x)):
            data.append(self.npy2data(x[i],post_var,hps))
            data[i]["value positions"] = None
            if y is not None: data[i]["values"] = y[i]
            if v is not None: data[i]["variances"] = v[i]
            if vp is not None:data[i]["value positions"] = vp[i]
        return data

    def npy2data(self, x,post_var = None, hps = None, cost = None):
        """
        parameters:
        -----------
        x ... 1d numpy array
        y ... float
        v ... float
        post_var ... float
        """
        d = {}
        d["position"] = x
        d["values"] = None
        d["variances"] = None
        d["value positions"] = None
        d["cost"] = cost
        d["id"] = str(uuid.uuid4())
        d["time stamp"] = time.time()
        d["date time"] = datetime.datetime.now().strftime("%d/%m/%Y_%H:%M%S")
        d["measured"] = False
        d["posterior variances"] = post_var
        d["hyperparameters"] = hps
        return d

    ################################################################
    ################################################################
    ################################################################
    def extract_value_positions_from_data(self):
        VP = np.zeros((self.point_number, self.output_number, self.output_dim))
        for idx_data in range(self.point_number):
            if ("value positions" in self.dataset[idx_data]):
                VP[idx_data] = self.dataset[idx_data]["value positions"]
            else:
                VP[idx_data] = self.dataset[idx_data - 1]["value positions"]
        return VP

    def extract_values_from_data(self):
        M= np.zeros((self.point_number,self.output_number))
        for idx_data in range(self.point_number):
            M[idx_data] = self.dataset[idx_data]["values"]
        return M

    def extract_variances_from_data(self):
        Variance = np.zeros((self.point_number,self.output_number))
        for idx_data in range(self.point_number):
            if self.dataset[idx_data]["variances"] is None: return None
            Variance[idx_data] = self.dataset[idx_data]["variances"]
        return Variance


