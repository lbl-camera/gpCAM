import numpy as np

from gpcam.autonomous_experimenter import AutonomousExperimenterGP


def test_basic_smoke():
    def instrument(data, instrument_dict={}):
        for entry in data:
            entry["value"] = np.sin(np.linalg.norm(entry["position"]))
        return data

    ##set up your parameter space
    parameters = np.array([[3.0, 45.8],
                           [4.0, 47.0]])

    ##set up some hyperparameters, if you have no idea, set them to 1 and make the training bounds large
    init_hyperparameters = np.array([1, 1, 1])
    hyperparameter_bounds = np.array([[0.01, 100], [0.01, 100.0], [0.01, 100]])

    ##let's initialize the autonomous experimenter ...
    my_ae = AutonomousExperimenterGP(parameters, init_hyperparameters,
                                     hyperparameter_bounds, instrument_func=instrument,
                                     init_dataset_size=10)
    # ...train...
    my_ae.train()

    # ...and run. That's it. You successfully executed an autonomous experiment.
    my_ae.go(N=15, retrain_locally_at=(11,), retrain_globally_at=(12,), retrain_async_at=(13,))
