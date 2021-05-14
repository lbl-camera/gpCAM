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



class fvgpData(gpData):
    def create_random_init_dataset(self):
        self.x = self._create_random_points()
        self.point_number = len(self.x)
        self.x, self.y,self.v, self.vp = self.add_data_points(self.x)
        self.output_number = len(self.v[0])

    def comm_init_dataset(self,data):
        self.dataset = data
        self.point_number = len(data)
        self.output_number = len(data[0]["variances"])
        self.vp = self.extract_value_positions_from_data()
        self.x = self.extract_points_from_data()
        self.y = self.extract_values_from_data()
        self.v = self.extract_variances_from_data()
        self.times = self.extract_times_from_data()
        self.measurement_costs = self.extract_costs_from_data()
        self.oput_number = len(self.vp[0])

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
        tranlates numpy arrays to the data format
        """
        data = []
        for i in range(len(x)):
            data.append(self.npy2data(x[i],post_var,hps))
            data[i]["value position"] = None
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
        VP = np.zeros((self.point_number, self.oput_num, self.oput_dim))
        for idx_data in range(self.point_number):
            if ("value positions" in self.data_set[idx_data]):
                VP[idx_data] = self.data_set[idx_data]["value positions"]
            else:
                VP[idx_data] = self.data_set[idx_data - 1]["value positions"]
        return VP
    def extract_values_from_data(self):
        M= np.zeros((self.point_number,self.ouput_number))
        for idx_data in range(self.point_number):
            M[idx_data] = self.dataset[idx_data]["values"]
        return M

    def extract_variances_from_data(self):
        Variance = np.zeros((self.point_number,self.ouput_number))
        for idx_data in range(self.point_number):
            if self.dataset[idx_data]["variances"] is None: return None
            Variance[idx_data] = self.dataset[idx_data]["variances"]
        return Variance


