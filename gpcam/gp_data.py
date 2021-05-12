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
    def __init__(self, dim, function):
        self.function = function

    def update_data(self, new_points, post_var, hps):
        new_data_list = self.compress_into_data(new_points,post_var, hps)
        self.data_set = self.update_data_set(new_data_list)
        self.data_set = self.clean_data_NaN(self.data_set)
        self.points = self.extract_points_from_data()
        self.values = self.extract_values_from_data()
        self.variances = self.extract_variances_from_data()
        self.times = self.extract_times_from_data()
        return self.point, self.values,self.variances

    ################################################################
    ################################################################
    ################################################################
    def extract_points_from_data(self):
        P = np.zeros((self.point_number, self.iput_dim))
        for idx_data in range(self.point_number):
            index = 0
            for idx_parameter in self.conf.parameters:
                P[idx_data, index] = self.data_set[idx_data]["position"][idx_parameter]
                index = index + 1
        return P

    def extract_values_from_data(self):
        M= np.zeros((self.point_number, self.oput_num))
        for idx_data in range(self.point_number):
            M[idx_data] = self.data_set[idx_data]["measurement values"]["values"]
        return M

    def extract_variances_from_data(self):
        Variance = np.zeros((self.point_number, self.oput_num))
        for idx_data in range(self.point_number):
            Variance[idx_data] = self.data_set[idx_data][
                "measurement values"
            ]["variances"]
        return Variance

    def extract_costs_from_data(self):
        Costs = []
        for idx in range(self.point_number):
            Costs.append(self.data_set[idx]["cost"])
        return Costs

    def extract_times_from_data(self):
        times = np.zeros((self.point_number))
        for idx_data in range(self.point_number):
            times[idx_data] = self.data_set[idx_data]["time stamp"]
        return times


    def compress_into_data(self, new_points, 
            post_var,
            hps):
        data = []
        for i in range(len(new_points)):
            data.append({})
            data[i] = {}
            index = 0
            data[i]["position"] = new_points[i]
            data[i]["values"] = np.zeros((self.oput_num))
            data[i]["variances"] = np.zeros((self.oput_num))

            data[i]["cost"] = None #new_measurement_costs[i]
            data[i]["id"] = str(uuid.uuid4())
            data[i]["time stamp"] = time.time()
            data[i]["date time"] = datetime.datetime.now().strftime("%d/%m/%Y_%H:%M%S")
            data[i]["measured"] = False
            data[i]["posterior variance"] = post_var
            data[i]["hyperparameters"] = hps
        return data
    ###############################################################
    def update_data_set(self, new_data, append):
        if append:
            self.data_set = self.data_set + new_data
            self.data_set = self.function(self.data_set)
        else:
            new_data = self.function(new_data)
            self.data_set = self.data_set + new_data
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
