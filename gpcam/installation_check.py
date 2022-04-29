import numpy as np
from gpcam.autonomous_experimenter import AutonomousExperimenterGP
def instrument(data, instrument_dict = {}):
    for entry in data:
        entry["value"] = np.sin(np.linalg.norm(entry["position"]))
        entry["variance"] = 0.1
    return data

def check():
    try:
        parameters = np.array([[3.0,45.8],
                                [4.0,47.0]])

        ##set up some hyperparameters, if you have no idea, set them to 1 and make the training bounds large
        init_hyperparameters = np.array([1,1,1])
        hyperparameter_bounds =  np.array([[0.01,100],[0.01,100.0],[0.01,100]])

        ##let's initialize the autonomous experimenter ...
        my_ae = AutonomousExperimenterGP(parameters, init_hyperparameters,
                                        hyperparameter_bounds,instrument_func = instrument,
                                        init_dataset_size=10)
        print("Installation test successfully concluded. gpCAM is installed and ready to go.")
    except: raise Exception("Installation test not successful")


