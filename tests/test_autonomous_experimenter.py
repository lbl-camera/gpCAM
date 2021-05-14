#/usr/bin/env python
import numpy as np
from gpcam.autonomous_experimenter import AutonomousExperimenterGP

def instrument(data):
    for entry in data:
        entry["value"] = np.sin(np.linalg.norm(entry["position"]))
    return data

my_ae = AutonomousExperimenterGP(np.array([[0,10],[0,10]]),instrument,np.ones((3)),np.array([[0.001,100],[0.001,100],[0.001,100]]),init_dataset_size= 10)
my_ae.train()
input("training done")
my_ae.go(N = 200)

