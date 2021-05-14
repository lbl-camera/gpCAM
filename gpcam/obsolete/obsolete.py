    def evaluate_acquisition_function(
        self, x, acquisition_function="covariance", cost_function = None,
        origin=None):
        """Evaluates the acquisition function.

        Parameters:
        -----------
        x: 1d numpy array.

        Optional Parameters:
        --------------------
        acquisition_function : default = "covariance",
                               "covariance","shannon_ig" ,..., or callable, use the same you use
                               in ask(). (The default is "covariance").
        origin:                default = None, only important for cost considerations

        Returns
        -------
        float or numpy array
        """

        if self.gp_initialized is False:
            raise Exception(
                "Initialize GP before evaluating the acquisition function. "
                "See help(gp_init)."
            )
        x = np.array(x)
        try:
            return sm.evaluate_acquisition_function(
                x, self, acquisition_function, origin, self.cost_function,
                self.cost_function_parameters
            )
        except Exception as a:
            print("Evaluating the acquisition function was not successful.")
            print("Error Message:")
            print(str(a))
##############################################################
    def train_gp_async(self, hyperparameter_bounds,
            likelihood_optimization_pop_size = 20,
            likelihood_optimization_tolerance = 1e-6,
            likelihood_optimization_max_iter = 10000,
            dask_client = None):
        """
        Function to start fvGP asynchronous training.
        Parameters:
        -----------
            hyperparameter_bounds:                  2d np.array of bounds for the hyperparameters
        Optional Parameters:
        --------------------
            likelihood_optimization_pop_size:       number of walkers in the optimization, default = 20
            likelihood_optimization_tolerance:      tolerance for termination, default = 1e-6
            likelihood_optimization_max_iter:       maximum number of iterations, default = 10000
            dask_client:                            a DASK client, see dask package docs for explanation
        """
        if self.gp_initialized is False:
            raise Exception("No GP to be trained. Please call init_gp(...) before training.")
        self.train(
                hyperparameter_bounds,
                init_hyperparameters = self.hyperparameters,
                optimization_pop_size = likelihood_optimization_pop_size,
                optimization_tolerance = likelihood_optimization_tolerance,
                optimization_max_iter = likelihood_optimization_max_iter,
                dask_client = dask_client
                )
        return self.hyperparameters

##############################################################
    def train_gp(self,hyperparameter_bounds,
            likelihood_optimization_method = "global",likelihood_optimization_pop_size = 20,
            optimization_dict = None,likelihood_optimization_tolerance = 1e-6,
            likelihood_optimization_max_iter = 120):
        """
        Function to perform fvGP training.
        Parameters:
        -----------
            hyperparameter_bounds:                  2d np.array of bounds for the hyperparameters
        Optional Parameters:
        --------------------
            likelihood_optimization_method:         "hgdl"/"global"/"local", default = "global"
            likelihood_optimization_pop_size:       number of walkers in the optimization, default = 20
            optimization_dict:                      default = None
            likelihood_optimization_tolerance:      tolerance for termination, default = 1e-6
            likelihood_optimization_max_iter:       maximum number of iterations, default = 120
        """

        if self.gp_initialized is False:
            raise Exception("No GP to be trained. Please call init_gp(...) before training.")
        self.train(
                hyperparameter_bounds,
                init_hyperparameters = self.hyperparameters,
                optimization_method = likelihood_optimization_method,
                optimization_dict = optimization_dict,
                optimization_pop_size = likelihood_optimization_pop_size,
                optimization_tolerance = likelihood_optimization_tolerance,
                optimization_max_iter = likelihood_optimization_max_iter
                )
        return self.hyperparameters

##############################################################
    def stop_async_train(self):
        """
        function to stop vfGP async training
        Parameters:
        -----------
            no input parameters
        """
        try: self.stop_training()
        except: pass

##############################################################
    def update_hyperparameters(self):
        self.update_hyperparameters()
        return self.hyperparameters

##############################################################
    def init_cost(self,cost_function,cost_function_parameters,
            cost_update_function = None, cost_function_optimization_bounds = None):
        """
        This function initializes the costs. If used, the acquisition function will be augmented by the costs
        which leads to different suggestions

        Parameters:
        -----------
            cost_function: callable
            cost_function_parameters: arbitrary, are passed to the user defined cost function

        Optional Parameters:
        --------------------
            cost_update_function: a function that updates the cost_fucntion_parameters, default = None
            cost_function_optimization_bounds: optimization bounds for the update, default = None

        Return:
        -------
            cost function that can be injected into ask()
        """

        self.cost_function = cost_function
        self.cost_function_parameters = cost_function_parameters
        self.cost_function_optimization_bounds = cost_function_optimization_bounds
        self.cost_update_function = cost_update_function
        self.consider_costs = True
        print("Costs successfully initialized")
        return self.cost_function

##############################################################
    def update_cost_function(self,measurement_costs):
        """
        This function updates the parameters for the cost function
        It essentially calls the user-given cost_update_function which
        should return the new parameters how they are used by the user defined
        cost function
        Parameters:
        -----------
            measurement_costs:    an arbitrary structure that describes 
                                  the costs when moving in the parameter space
            cost_update_function: a user-defined function 
                                  def name(measurement_costs,
                                  cost_fucntion_optimization_bounds,cost_function_parameters)
                                  which returns the new parameters
            cost_function_optimization_bounds: see above
        Optional Parameters:
        --------------------
            cost_function_parameters, default = None
        """

        print("Performing cost function update...")
        if self.cost_function_parameters is None: raise Exception("No cost function parameters specified. Please call init_cost() first.")
        self.cost_function_parameters = \
        self.cost_update_function(measurement_costs,
        self.cost_function_optimization_bounds,
        self.cost_function_parameters)
        print("cost parameters changed to: ", self.cost_function_parameters)
##############################################################
    def ask(self, position = None, n = 1,
            acquisition_function = "covariance",
            cost_function = None,
            optimization_bounds = None,
            optimization_method = "global",
            optimization_pop_size = 20,
            optimization_max_iter = 20,
            optimization_tol = 10e-6,
            optimization_x0 = None,
            dask_client = False):
        """
        Given that the acquisition device is at "position", the function ask() s for
        "n" new optimal points within certain "bounds" and using the optimization setup:
        "acquisition_function_pop_size", "max_iter" and "tol"
        Parameters:
        -----------

        Optional Parameters:
        --------------------
            position (numpy array):            last measured point, default = None
            n (int):                           how many new measurements are requested, default = 1
            acquisition_function:              default = None, means that the class acquisition function will be used
            cost_function:                     default = None, otherwise cost objective received from init_cost, or callable
            optimization_bounds (2d list/None):             default = None
            optimization_method:                            default = "global", "global"/"hgdl"
            optimization_pop_size (int):                    default = 20
            optimization_max_iter (int):                    default = 20
            optimization_tol (float):                       default = 10e-6
            optimization_x0:                                default = None, starting positions for optimizer
            dask_client:                                    default = False
        """
        print("aks() initiated with hyperparameters:",self.hyperparameters)
        print("optimization method: ", optimization_method)
        print("bounds: ",optimization_bounds)
        if optimization_bounds is None: optimization_bounds = self.index_set_bounds
        maxima,func_evals = sm.find_acquisition_function_maxima(
                self,
                acquisition_function,
                position,n, optimization_bounds,
                optimization_method = optimization_method,
                optimization_pop_size = optimization_pop_size,
                optimization_max_iter = optimization_max_iter,
                optimization_tol = optimization_tol,
                cost_function = cost_function,
                cost_function_parameters = self.cost_function_parameters,
                dask_client = dask_client)
        return {'x':np.array(maxima), "f(x)" : np.array(func_evals)}

def send_data_as_files(data):
    path_new_command = "../data/command/"
    path_new_result = "../data/result/"
    while os.path.isfile(path_new_command + "command.npy"):
        time.sleep(1)
        print("Waiting for experiment device to read and subsequently delete last command.")

    write_success = False
    read_success = False
    while write_success == False:
        try:
            np.save(path_new_command + "command", data)
            np.save(path_new_command + "command_bak", data)
            write_success = True
            print("Successfully send data set of length ",len(data)," to experiment device")
        except:
            time.sleep(1)
            print("Saving new experiment command file not successful, trying again...")
            write_success = False
    while read_success == False:
        try:
            new_data = np.load(
                path_new_result + "result.npy", encoding="ASCII", allow_pickle=True
            )
            read_success = True
            print("Successfully received data set of length ",len(new_data)," from experiment device")
            copyfile(path_new_result + "result.npy",path_new_result + "result_bak.npy")
            os.remove(path_new_result + "result.npy")
        except:
            print("New measurement values have not been written yet.")
            print("exception: ", sys.exc_info()[0])
            time.sleep(1)
            read_success = False
    return list(new_data)



#################################################
############interpolate existing data############
#################################################
def interpolate_experiment_data(data):
    space_dim = 4
    File = ""
    method = ""
    interpolate_data_array = np.load(File)
    p = interpolate_data_array[:, 0 : space_dim]
    m = interpolate_data_array[:,space_dim]
    asked_points = np.zeros((len(data),space_dim))
    for idx_data in range(len(data)):
        index = 0
