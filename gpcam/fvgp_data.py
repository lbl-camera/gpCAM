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
    def __init__(self, dim, parameter_bounds, function, init_dataset_size = None, append = False):
        gpData__init__(self, dim, parameter_bounds, function,
                init_dataset_size = init_dataset_size, append = append)

    def create_random_init_dataset(self):
        self.x = self._create_random_points()
        self.point_number = len(self.x)
        self.x, self.y, self.v, self.vp = self.add_data_points(self.x)


    def update_data(self, new_points,post_var,hps):
        new_data_list = self.compress_into_data(new_points, 
                post_var,
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
    def extract_value_positions_from_data(self):
        VP = np.zeros((self.point_number, self.oput_num, self.oput_dim))
        for idx_data in range(self.point_number):
            if ("value positions" in self.data_set[idx_data]):
                VP[idx_data] = self.data_set[idx_data]["value positions"]
            else:
                VP[idx_data] = self.data_set[idx_data - 1]["value positions"]
        return VP
