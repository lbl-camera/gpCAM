import datetime
import math
import time
import uuid
import warnings

import numpy as np


class gpData:
    """
    Data Class

    dataset is a list of dictionaries
    data arrays are numpy arrays
    """

    def __init__(self, dim, parameter_bounds, output_number=None):
        self.dim = dim
        self.parameter_bounds = parameter_bounds
        self.dataset = []
        self.output_number = output_number
        self.point_number = None
        self.x = None

    ###############################################################

    def update_dataset(self, dataset):
        self.dataset = dataset
        self.point_number = len(dataset)

    # either create random data or communicate a data set (use translate2data for numpy arrays)
    def create_random_dataset(self, length):
        """
        creates random data of "length" and creates a dataset
        """
        self.x = self._create_random_points(length)
        self.point_number = len(self.x)
        self.dataset = self.arrays2data(self.x)

    def inject_dataset(self, dataset):
        """
        initializes a previously-collected dataset
        !!!for initialization only!!! just use the "+" operator to update the existing dataset
        """
        self.point_number = len(self.dataset)
        self.dataset = dataset

    def arrays2data(self, x, y=None, v=None, info=None):
        """
        translates numpy arrays to the data format
        """
        assert np.ndim(x) == 2
        if y is not None: assert np.ndim(y) == 1
        if v is not None: assert np.ndim(v) == 1
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
        d = {"x_data": x, "y_data": y, "noise variance": v, "cost": None, "id": str(uuid.uuid4()),
             "time stamp": time.time(), "date time": datetime.datetime.now().strftime("%d/%m/%Y_%H:%M:%S"),
             "measured": False}
        # d["posterior variance"] = None #post_var
        # d["hyperparameters"] = None #hps
        return d

    ################################################################
    ########Extracting##############################################
    ################################################################
    def extract_data(self):
        x = self.extract_points_from_data()
        y = self.extract_y_data_from_data()
        v = self.extract_variances_from_data()
        t = self.extract_times_from_data()
        c = self.extract_costs_from_data()
        return x, y, v, t, c

    def extract_points_from_data(self):
        P = np.zeros((self.point_number, self.dim))
        for idx_data in range(self.point_number):
            P[idx_data] = self.dataset[idx_data]["x_data"]
        return P

    def extract_y_data_from_data(self):
        M = np.zeros(self.point_number)
        for idx_data in range(self.point_number):
            M[idx_data] = self.dataset[idx_data]["y_data"]
        return M

    def extract_variances_from_data(self):
        Variance = np.zeros(self.point_number)
        for idx_data in range(self.point_number):
            if self.dataset[idx_data]["noise variance"] is None: return None
            Variance[idx_data] = self.dataset[idx_data]["noise variance"]
        return Variance

    def extract_costs_from_data(self):
        Costs = []
        for idx in range(self.point_number):
            Costs.append(self.dataset[idx]["cost"])
        return Costs

    def extract_times_from_data(self):
        times = np.zeros(self.point_number)
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
                if entry["y_data"] is None:
                    raise Exception("Entry with no specified y_data in communicated list of data dictionaries")
                if entry["x_data"] is None:
                    raise Exception("Entry with no specified x_data in communicated list of data dictionaries")
                if entry["noise variance"] is None:
                    raise Exception("Entry with no specified noise variance in communicated list of data dictionaries")
        except:
            raise Exception(
                "Checking the incoming data could not be accomplished. This normally means that wrong formats were "
                "communicated")

    def clean_data_NaN(self):
        for entry in self.dataset:
            if self._nan_in_dict(entry):
                warnings.warn("CAUTION, NaN detected in data")
                self.dataset.remove(entry)
        self.point_number = len(self.dataset)

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
    def __init__(self, dim, parameter_bounds, output_number):
        if output_number is None:
            raise Exception("When initializing the data class for a multi-output GP, "
                            "please provide the output_number.")
        super(fvgpData, self).__init__(dim, parameter_bounds, output_number)

    def create_random_dataset(self, length):
        self.x = self._create_random_points(length)
        self.point_number = len(self.x)
        self.dataset = self.arrays2data(self.x)

    def inject_dataset(self, dataset):
        """
        initializes a previously-collected dataset
        !!!for initiation only!!! just use the "+" operator to update the existing dataset
        """
        self.point_number = len(self.dataset)
        self.dataset = dataset

    def arrays2data(self, x, y=None, v=None, vp=None, info=None):
        """
        translates numpy arrays to the data format
        """
        if np.ndim(x) != 2: raise Exception("'arrays2data' called with dim(x) != 2")
        if np.ndim(y) != 2 and y is not None: raise Exception("'arrays2data' called with dim(y) != 2")
        if np.ndim(v) != 2 and v is not None: raise Exception("'arrays2data' called with dim(v) != 2")
        if np.ndim(vp) != 2 and vp is not None: raise Exception("'arrays2data' called with dim(vp)!= 2")

        data = []
        for i in range(len(x)):
            val = None
            var = None
            valp = None
            if y is not None: val = y[i]
            if v is not None: var = v[i]
            if vp is not None: valp = vp[i]
            if y is not None and vp is None: valp = np.array([np.array([float(i)]) for i in range(len(y[0]))])

            data.append(self.npy2dataset_entry(x[i], val, var, valp))
            if info is not None: data[i].update(info[i])
        return data

    def npy2dataset_entry(self, x, y=None, v=None, vp=None):
        """
        parameters:
        -----------
        x ... 1d numpy array
        """
        d = {"x_data": x, "y_data": y, "noise variances": v, "output positions": vp, "cost": None,
             "id": str(uuid.uuid4()), "time stamp": time.time(),
             "date time": datetime.datetime.now().strftime("%d/%m/%Y_%H:%M:%S"), "measured": False}
        # d["posterior variances"] = None
        # d["hyperparameters"] = None
        return d

    ################################################################
    ################################################################
    ################################################################
    def extract_data(self):
        x = self.extract_points_from_data()
        y = self.extract_y_data_from_data()
        v = self.extract_variances_from_data()
        t = self.extract_times_from_data()
        c = self.extract_costs_from_data()
        vp = self.extract_output_positions_from_data()
        return x, y, v, t, c, vp

    def extract_output_positions_from_data(self):
        VP = np.zeros((self.point_number, self.output_number))
        for idx_data in range(self.point_number):
            if "output positions" in self.dataset[idx_data]:
                VP[idx_data] = self.dataset[idx_data]["output positions"]
            else:
                VP[idx_data] = self.dataset[idx_data - 1]["output positions"]
        return VP

    def extract_y_data_from_data(self):
        M = np.zeros((self.point_number, self.output_number))
        for idx_data in range(self.point_number):
            M[idx_data] = self.dataset[idx_data]["y_data"]
        return M

    def extract_variances_from_data(self):
        Variance = np.zeros((self.point_number, self.output_number))
        for idx_data in range(self.point_number):
            if self.dataset[idx_data]["noise variances"] is None: return None
            Variance[idx_data] = self.dataset[idx_data]["noise variances"]
        return Variance

    def check_incoming_data(self):
        try:
            for entry in self.dataset:
                if entry["y_data"] is None:
                    raise Exception("Entry with no specified y_data in communicated list of data dictionaries")
                if entry["x_data"] is None:
                    raise Exception("Entry with no specified x_data in communicated list of data dictionaries")
                if entry["output positions"] is None:
                    raise Exception(
                        "Entry with no specified output positions in communicated list of data dictionaries")
                if entry["noise variances"] is None:
                    raise Exception("Entry with no specified noise variances in communicated list of data dictionaries")
        except Exception as e:
            raise Exception(
                "Checking the incoming data could not be accomplished. "
                "This normally means that wrong formats were communicated: ", e)
