import datetime
import math
import time
import uuid

import numpy as np


class gpData:
    """
    Data Class

    dataset is a list of dictionaries
    data arrays are numpy arrays
    """

    def __init__(self, dim, parameter_bounds, output_number=None, output_dim=None):
        self.dim = dim
        self.parameter_bounds = parameter_bounds
        self.dataset = []
        self.output_number = output_number
        self.output_dim = output_dim

    ###############################################################
    # either create random data or communicate a data set (use translate2data for numpy arrays)
    def create_random_dataset(self, length):
        """
        creates random data of "length" and creates a dataset
        """
        self.x = self._create_random_points(length)
        self.point_number = len(self.x)
        self.dataset = self.inject_arrays(self.x)

    def inject_dataset(self, dataset):
        """
        initializes a previously-collected dataset
        !!!for intitiation only!!! just use the "+" operator to update the existing dataset
        """
        self.point_number = len(self.dataset)
        self.dataset = dataset

    def inject_arrays(self, x, y=None, v=None, info=None):
        """
        translates numpy arrays to the data format
        """
        if np.ndim(x) != 2: raise Exception("'inject_arrays' called with dim(x) != 2")
        if np.ndim(y) == 2: y = y[:, 0]
        if np.ndim(v) == 2: v = v[:, 0]
        data = []
        for i in range(len(x)):
            val = None
            var = None
            if y is not None: val = y[i]
            if v is not None: var = v[i]
            data.append(self.npy2dataset_entry(x[i], val, var))
            if info is not None: data[i].update(info[i])
        return data

    ###############################################################
    def npy2dataset_entry(self, x, y=None, v=None):
        """
        parameters:
        -----------
            x ... 1d numpy array
        """
        d = {}
        d["position"] = x
        d["value"] = y
        d["variance"] = v
        d["cost"] = None
        d["id"] = str(uuid.uuid4())
        d["time stamp"] = time.time()
        d["date time"] = datetime.datetime.now().strftime("%d/%m/%Y_%H:%M%S")
        d["measured"] = False
        # d["posterior variance"] = None #post_var
        # d["hyperparameters"] = None #hps
        return d

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
    def extract_data(self):
        x = self.extract_points_from_data()
        y = self.extract_values_from_data()
        v = self.extract_variances_from_data()
        t = self.extract_times_from_data()
        c = self.extract_costs_from_data()
        return x, y, v, t, c

    def extract_points_from_data(self):
        self.point_number = len(self.dataset)
        P = np.zeros((self.point_number, self.dim))
        for idx_data in range(self.point_number):
            P[idx_data] = self.dataset[idx_data]["position"]
        return P

    def extract_values_from_data(self):
        self.point_number = len(self.dataset)
        M = np.zeros((self.point_number))
        for idx_data in range(self.point_number):
            M[idx_data] = self.dataset[idx_data]["value"]
        return M

    def extract_variances_from_data(self):
        self.point_number = len(self.dataset)
        Variance = np.zeros((self.point_number))
        for idx_data in range(self.point_number):
            if self.dataset[idx_data]["variance"] is None: return None
            Variance[idx_data] = self.dataset[idx_data]["variance"]
        return Variance

    def extract_costs_from_data(self):
        self.point_number = len(self.dataset)
        Costs = []
        for idx in range(self.point_number):
            Costs.append(self.dataset[idx]["cost"])
        return Costs

    def extract_times_from_data(self):
        self.point_number = len(self.dataset)
        times = np.zeros((self.point_number))
        for idx_data in range(self.point_number):
            times[idx_data] = self.dataset[idx_data]["time stamp"]
        return times

    ###############################################################
    #######Creating################################################
    ###############################################################
    def _create_random_x(self):
        return np.random.uniform(low=self.parameter_bounds[:, 0],
                                 high=self.parameter_bounds[:, 1],
                                 size=self.dim)

    def _create_random_points(self, length):
        x = np.empty((length, self.dim))
        for i in range(length):
            x[i, :] = self._create_random_x()
        return x

    ###############################################################
    #########Cleaning##############################################
    ###############################################################
    def check_incoming_data(self):
        try:
            for entry in self.dataset:
                if entry["value"] is None:
                    raise Exception("Entry with no specified value in communicated list of data dictionaries")
                if entry["position"] is None:
                    raise Exception("Entry with no specified position in communicated list of data dictionaries")
        except:
            raise Exception(
                "Checking the incoming data could not be accomplished. This normally means that wrong formats were "
                "communicated")

    def clean_data_NaN(self):
        for entry in self.dataset:
            if self._nan_in_dict(entry):
                print("CAUTION, NaN detected in data")
                self.dataset.remove(entry)
        self.point_number = len(self.data_set)

    def nan_in_dataset(self):
        for entry in self.dataset:
            if self._nan_in_dict(entry):
                return True
        return False

    def _nan_in_dict(self, dictionary):
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
    def __init__(self, dim, parameter_bounds, output_number, output_dim):
        if output_number is None or output_dim is None:
            raise Exception("When initializing the data class for a multi-output GP, \
                    please provide output_number AND output_dim parameters.")
        super(fvgpData, self).__init__(dim, parameter_bounds, output_number, output_dim)

    def create_random_dataset(self, length):
        self.x = self._create_random_points(length)
        self.point_number = len(self.x)
        self.dataset = self.inject_arrays(self.x)

    def inject_dataset(self, dataset):
        """
        initializes a previously-collected dataset
        !!!for intitiation only!!! just use the "+" operator to update the existing dataset
        """
        self.point_number = len(self.dataset)
        self.dataset = dataset

    def inject_arrays(self, x, y=None, v=None, vp=None, info=None):
        """
        translates numpy arrays to the data format
        """
        if np.ndim(x) != 2: raise Exception("'inject_arrays' called with dim(x) != 2")
        if np.ndim(y) != 2 and y is not None: raise Exception("'inject_arrays' called with dim(y) != 2")
        if np.ndim(v) != 2 and v is not None: raise Exception("'inject_arrays' called with dim(v) != 2")
        if np.ndim(vp) != 3 and vp is not None: raise Exception("'inject_arrays' called with dim(vp)!= 3")

        data = []
        for i in range(len(x)):
            val = None
            var = None
            valp = None
            if y is not None: val = y[i]
            if v is not None: var = v[i]
            if vp is not None: valp = vp[i]

            data.append(self.npy2dataset_entry(x[i], val, var, valp))
            if info is not None: data[i].update(info[i])
        return data

    def npy2dataset_entry(self, x, y=None, v=None, vp=None):
        """
        parameters:
        -----------
        x ... 1d numpy array
        """
        d = {}
        d["position"] = x
        d["values"] = y
        d["variances"] = v
        d["value positions"] = vp
        d["cost"] = None
        d["id"] = str(uuid.uuid4())
        d["time stamp"] = time.time()
        d["date time"] = datetime.datetime.now().strftime("%d/%m/%Y_%H:%M%S")
        d["measured"] = False
        # d["posterior variances"] = None
        # d["hyperparameters"] = None
        return d

    ################################################################
    ################################################################
    ################################################################
    def extract_data(self):
        x = self.extract_points_from_data()
        y = self.extract_values_from_data()
        v = self.extract_variances_from_data()
        t = self.extract_times_from_data()
        c = self.extract_costs_from_data()
        vp = self.extract_value_positions_from_data()
        return x, y, v, t, c, vp

    def extract_value_positions_from_data(self):
        self.point_number = len(self.dataset)
        VP = np.zeros((self.point_number, self.output_number, self.output_dim))
        for idx_data in range(self.point_number):
            if ("value positions" in self.dataset[idx_data]):
                VP[idx_data] = self.dataset[idx_data]["value positions"]
            else:
                VP[idx_data] = self.dataset[idx_data - 1]["value positions"]
        return VP

    def extract_values_from_data(self):
        self.point_number = len(self.dataset)
        M = np.zeros((self.point_number, self.output_number))
        for idx_data in range(self.point_number):
            M[idx_data] = self.dataset[idx_data]["values"]
        return M

    def extract_variances_from_data(self):
        self.point_number = len(self.dataset)
        Variance = np.zeros((self.point_number, self.output_number))
        for idx_data in range(self.point_number):
            if self.dataset[idx_data]["variances"] is None: return None
            Variance[idx_data] = self.dataset[idx_data]["variances"]
        return Variance

    def check_incoming_data(self):
        try:
            for entry in self.dataset:
                if entry["values"] is None:
                    raise Exception("Entry with no specified value in communicated list of data dictionaries")
                if entry["position"] is None:
                    raise Exception("Entry with no specified position in communicated list of data dictionaries")
                if entry["value positions"] is None:
                    raise Exception("Entry with no specified position in communicated list of data dictionaries")
        except:
            raise Exception(
                "Checking the incoming data could not be accomplished. This normally means that wrong formats were communicated")
