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



class Data:
    def __init__(self, gp_idx, function, conf, Data = None):
        self.conf = conf
        self.function = function
        self.gp_idx = gp_idx
        self.oput_dim = \
            conf.gaussian_processes[gp_idx]["dimensionality of return"]
        self.oput_num = conf.gaussian_processes[gp_idx]["number of returns"]
        if Data is not None:
            self.data_set = Data
            self.point_number = len(self.data_set)
        elif Data is None:
            self.point_number = conf.initial_data_set_size
            self.data_set = self.initialize_data()
        else:
            print("no data specified")
            exit()

        self.data_set = self.clean_data_NaN(self.data_set)
        self.variance_optimization_bounds = self.extract_variance_optimization_bounds()
        self.point_number = len(self.data_set)
        self.iput_dim = len(self.conf.parameters)
        self.points = self.extract_points_from_data()
        self.values = self.extract_values_from_data()
        self.value_positions = self.extract_value_positions_from_data()
        self.variances = self.extract_variances_from_data()
        if conf.gaussian_processes[gp_idx]["cost function"] is not None: self.measurement_costs = self.extract_costs_from_data()
        else: self.measurement_costs = None
        self.times = self.extract_times_from_data()
        self.conf = conf

    def update_data(self, new_points, 
            #new_values, 
            #new_variances, new_value_positions, 
            #new_measurement_costs, 
            obj_func_value,
            hps):
        new_data_list = self.compress_into_data(new_points, 
                #new_values, 
                #new_variances, new_value_positions, 
                #new_measurement_costs, 
                obj_func_value,
                hps)
        self.update_data_set(new_data_list)
        self.data_set = self.clean_data_NaN(self.data_set)
        self.point_number = len(self.data_set)
        self.points = self.extract_points_from_data()
        self.values = self.extract_values_from_data()
        self.variances = self.extract_variances_from_data()
        self.value_positions = self.extract_value_positions_from_data()
        if self.conf.gaussian_processes[self.gp_idx]["cost function"] is not None: 
            self.measurement_costs = self.extract_costs_from_data()
        else: self.measurement_costs = None
        self.times = self.extract_times_from_data()

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

    def extract_value_positions_from_data(self):
        VP = np.zeros(
            (self.point_number, self.oput_num, self.oput_dim)
        )
        for idx_data in range(self.point_number):
            if (
                "value positions"
                in self.data_set[idx_data]["measurement values"]
            ):
                VP[idx_data] = self.data_set[idx_data]["measurement values"]["value positions"]
            else:
                VP[idx_data] = self.data_set[idx_data - 1]["measurement values"
                ][self.gp_idx]["value positions"]
        return VP

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
            #new_values,
            #new_variances, new_value_positions,
            #new_measurement_costs, 
            obj_func_eval,
            hps):
        data = []
        for i in range(len(new_points)):
            data.append({})
            data[i] = {}
            data[i]["position"] = {}
            data[i]["measurement values"] = {}
            data[i]["measurement values"]["values"] = {}
            data[i]["measurement values"]["variances"] = {}
            data[i]["measurement values"]["value positions"] = {}
            data[i]["function name"] = self.gp_idx
            index = 0
            for idx in self.conf.parameters.keys():
                data[i]["position"][idx] = new_points[i][index]
                index = index + 1
            data[i]["measurement values"]["values"] = np.zeros((self.oput_num))
            data[i]["measurement values"]["variances"] = np.zeros((self.oput_num))
            data[i]["measurement values"]["value positions"] = np.zeros((self.oput_num, self.oput_dim))

            data[i]["cost"] = None #new_measurement_costs[i]
            data[i]["id"] = str(uuid.uuid4())
            data[i]["time stamp"] = time.time()
            data[i]["date time"] = datetime.datetime.now().strftime("%d/%m/%Y_%H:%M%S")
            data[i]["measured"] = False
            data[i]["objective function evaluation"] = obj_func_eval
            data[i]["hyperparameters"] = hps
        return data


    def extract_variance_optimization_bounds(self):
        Bounds = []
        for name in list(self.conf.parameters):
            Bounds.append(self.conf.parameters[name]["element interval"])
        return Bounds

    def update_data_set(self, new_data):
        communicate_entire_data_set = True
        if communicate_entire_data_set == True:
            self.data_set = self.data_set + new_data
            self.data_set = self.function(self.data_set)
        else:
            new_data = self.function(new_data)
            self.data_set = self.data_set + new_data
        return 0

    def initialize_data(self):
        data = self.create_random_data()
        return data

    def read_data_from_file(self):
        if self.conf.ask_for_file == False:
            print("Read initial data from ", self.conf.data_file)
            data = np.load(self.conf.data_file).item()
        else:
            while FilesExist == False and counter < 100:
                counter = counter + 1
                try:
                    File1 = input("What is the path to the data file?")
                    data = np.load(File1).item()
                    FilesExist = True
                except:
                    FilesExist = False
                    print("Paths not correctly given. Try again!")
        return data

    def create_random_data(self):
        data = []
        point_index = 0
        while len(data) < self.point_number:
            data.append({})
            data[point_index]["position"] = {}
            for para_name in self.conf.parameters:
                lower_limit = self.conf.parameters[para_name]["element interval"][0]
                upper_limit = self.conf.parameters[para_name]["element interval"][1]
                data[point_index]["position"][para_name] = random.uniform(
                    lower_limit, upper_limit
                )
            if self.conf.gaussian_processes[self.gp_idx]["cost function"] is not None:
                data[point_index]["cost"] = {"origin": [0],"point":[0],"cost": 0}
            else:
                data[point_index]["cost"] = None

            data[point_index]["measurement values"] = {}
            s = self.oput_dim
            data[point_index]["measurement values"] = {}
            data[point_index]["measurement values"]["values"] = np.zeros((self.oput_num))
            data[point_index]["measurement values"]["variances"] = np.zeros((self.oput_num))
            data[point_index]["measurement values"]["value positions"] = np.zeros((self.oput_num, self.oput_dim))
            data[point_index]["time stamp"] = time.time()
            data[point_index]["date time"] = datetime.datetime.now().strftime("%d/%m/%Y_%H:%M%S")
            data[point_index]["measured"] = False
            data[point_index]["function name"] = self.gp_idx
            data[point_index]["metadata"] = None
            data[point_index]["id"] = str(uuid.uuid4())
            point_index = len(data)
        data = self.function(data)
        self.print_data(data)
        return data

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


    def clean_data_NaN(self,data):
        for entry in data:
            if self.nan_in_dict(entry):
                print("CAUTION, NaN detected in data")
                data.remove(entry)
        return data

    def nan_in_dict(self,dictionary):
        is_nan = False
        try:
            for key in dictionary:
                if type(dictionary[key]) is dict:
                    is_nan = self.nan_in_dict(dictionary[key])
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


###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
