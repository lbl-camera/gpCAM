# gpCAM

[![PyPI](https://img.shields.io/pypi/v/gpCAM)](https://pypi.org/project/gpcam/)
[![Documentation Status](https://readthedocs.org/projects/gpcam/badge/?version=latest)](https://gpcam.readthedocs.io/en/latest/?badge=latest)
[![gpCAM CI](https://github.com/lbl-camera/gpCAM/actions/workflows/gpCAM-CI.yml/badge.svg)](https://github.com/lbl-camera/gpCAM/actions/workflows/gpCAM-CI.yml)
[![Codecov](https://img.shields.io/codecov/c/github/lbl-camera/gpCAM)](https://app.codecov.io/gh/lbl-camera/gpCAM)
[![PyPI - License](https://img.shields.io/badge/license-GPL%20v3-lightgrey)](https://pypi.org/project/gpcam/)
[<img src="https://img.shields.io/badge/slack-@gpCAM-purple.svg?logo=slack">](https://gpCAM.slack.com/)
[![DOI](https://zenodo.org/badge/434768487.svg)](https://zenodo.org/badge/latestdoi/434768487)
![Downloads](https://img.shields.io/pypi/dm/gpcam)


[comment]: <> ([![Maintainability]&#40;https://api.codeclimate.com/v1/badges/29b04c3f69e2b515dac6/maintainability&#41;]&#40;https://codeclimate.com/github/lbl-camera/gpCAM/maintainability&#41;)
[comment]: <> (Hiding maintainibility score while starting to address issues)


gpCAM [(gpcam.lbl.gov)](https://gpcam.lbl.gov/home) is an API and software designed to make advanced Gaussian Process function approximation and autonomous data acquisition/Bayesian Optimization for experiments and simulations more accurate, faster, simpler, and more widely available. The tool is based on a flexible and powerful Gaussian process regression at the core. The flexibility stems from the modular design of gpCAM which allows the user to implement and import their own Python functions to customize and control almost every aspect of the software. That makes it possible to easily tune the algorithm to account for various kinds of physics and other domain knowledge and to identify and find interesting features, in Euclidean and non-Euclidean spaces. A specialized function optimizer in gpCAM can take advantage of HPC architectures for fast analysis time and reactive autonomous data acquisition. gpCAM broke a 2019 record for the largest exact GP ever run! Below you can see a simple example of how to set up an autonomous experimentation loop.


## Usage

The following demonstrates a simple usage of the gpCAM API (see [interactive demo](https://colab.research.google.com/drive/1FU4iKW626XiLqluDXQH-gzPYHyCf9N76?usp=sharing)). 

```python
!pip install gpcam

from gpcam.autonomous_experimenter import AutonomousExperimenterGP
import numpy as np

#define an instrument function, this is how the autonomous experimenter interacts with the world.
def instrument(data):
    for entry in data:
        print("I want to know the y_data at: ", entry["x_data"])
        ##always fill in y_data and noise variances
        entry["y_data"] = 0.001 * np.linalg.norm(entry["x_data"])**2
        entry["noise variance"] = 0.01
        print("I received ",entry["y_data"])
        print("")
    return data

##set up your parameter space
parameters = np.array([[3.0,45.8],
                       [4.0,47.0]])

##set up some hyperparameters, if you have no idea, set them to 1 and make the training bounds large
init_hyperparameters = np.array([ 1.,1.,1.])
hyperparameter_bounds =  np.array([[0.01,100],[0.01,1000.0],[0.01,1000]])

##let's initialize the autonomous experimenter ...
my_ae = AutonomousExperimenterGP(parameters, init_hyperparameters,
                                 hyperparameter_bounds,instrument_function = instrument, online=True, calc_inv=True, 
                                 init_dataset_size=20)
#...train...
my_ae.train(max_iter=2)


#...and run. That's it. You successfully executed an autonomous experiment.
st = time.time()
my_ae.go(N = 100, retrain_globally_at=[], retrain_locally_at=[])
print("Exec time: ", time.time() - st)
```


## Credits

Main Developer: Marcus Noack ([MarcusNoack@lbl.gov](mailto:MarcusNoack@lbl.gov))
Many people from across the DOE national labs (especially BNL) have given insights
that led to the code in it's current form.
See [AUTHORS](AUTHORS.rst) for more details on that.


