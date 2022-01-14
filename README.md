# gpCAM

[![PyPI](https://img.shields.io/pypi/v/gpCAM)](https://pypi.org/project/gpcam/)
[![Documentation Status](https://readthedocs.org/projects/gpcam/badge/?version=latest)](https://gpcam.readthedocs.io/en/latest/?badge=latest)
[![gpCAM CI](https://github.com/lbl-camera/gpCAM/actions/workflows/gpCAM-CI.yml/badge.svg)](https://github.com/lbl-camera/gpCAM/actions/workflows/gpCAM-CI.yml)
[![Codecov](https://img.shields.io/codecov/c/github/lbl-camera/gpCAM)](https://app.codecov.io/gh/lbl-camera/gpCAM)
[![PyPI - License](https://img.shields.io/pypi/l/gpCAM)](https://pypi.org/project/gpcam/)
[<img src="https://img.shields.io/badge/slack-@gpCAM-purple.svg?logo=slack">](https://gpCAM.slack.com/)

[comment]: <> ([![Maintainability]&#40;https://api.codeclimate.com/v1/badges/29b04c3f69e2b515dac6/maintainability&#41;]&#40;https://codeclimate.com/github/lbl-camera/gpCAM/maintainability&#41;)
[comment]: <> (Hiding maintainibility score while starting to address issues)


gpCAM is an API and software designed to make autonomous data acquisition and analysis for experiments and simulations faster, simpler and more widely available. The tool is based on a flexible and powerful Gaussian process regression at the core. The flexibility stems from the modular design of gpCAM which allows the user to implement and import their own Python functions to customize and control almost every aspect of the software. That makes it possible to easily tune the algorithm to account for various kinds of physics and other domain knowledge, and to identify and find interesting features. A specialized function optimizer in gpCAM can take advantage of HPC architectures for fast analysis time and reactive autonomous data acquisition.


## Usage

The following demonstrates a simple usage of the gpCAM API (see [interactive demo](https://colab.research.google.com/drive/1FU4iKW626XiLqluDXQH-gzPYHyCf9N76?usp=sharing)). 

```python
from gpcam.autonomous_experimenter import AutonomousExperimenterGP
import numpy as np

def instrument(data, instrument_dict = {}):
    for entry in data:
        entry["value"] = np.sin(np.linalg.norm(entry["position"]))
    return data

##set up your parameter space
parameters = np.array([[3.0,45.8],
                       [4.0,47.0]])

##set up some hyperparameters, if you have no idea, set them to 1 and make the training bounds large
init_hyperparameters = np.array([1,1,1])
hyperparameter_bounds =  np.array([[0.01,100],[0.01,100.0],[0.01,100]])

##let's initialize the autonomous experimenter ...
my_ae = AutonomousExperimenterGP(parameters, init_hyperparameters,
                                 hyperparameter_bounds,instrument_func = instrument,  
                                 init_dataset_size=10)
#...train...
my_ae.train()

#...and run. That's it. You successfully executed an autonomous experiment.
my_ae.go(N = 100)
```


## Credits

Main Developer: Marcus Noack ([MarcusNoack@lbl.gov](mailto:MarcusNoack@lbl.gov))
Several people from across the DOE national labs have given insights
that led the code in it's current form.
See [AUTHORS](AUTHORS.rst) for more details on that.


