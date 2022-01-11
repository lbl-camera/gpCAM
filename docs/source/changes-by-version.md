---
banner: _static/path.jpg
---

# Changes by Version

## 7.3.2 --> 7.4.0 (NOT AVAILABLE FROM PIPY YET)

* [autonomous experimenter](api/autonomous-experimenter.md),
  [gp_optimizer](api/gpOptimizer.md), and 
  [fvgp](api/fvgpOptimizer.md) (+ all multi-task variants):
  optional ram_economy mode, can compute the gradient and Hessian without extra RAM usage and is only slightly slower

* [fvgp](api/fvgpOptimizer.md): offers robust kernels in which 1/l is replaces by b**2, should be used with local and hybrid optimizers, makes the domain closed and convex. This change is adopted by all downstream classes

* [autonomous experimenter](api/autonomous-experimenter.md),
  [gp_optimizer](api/gpOptimizer.md), and
  [fvgp](api/fvgpOptimizer.md) (+ all multi-task variants):
  The user can communicate derivatives dk/dh and dm/dh which can lead to significant speedups compared to the native implementation since finite differencing can be completely avoided.

## 7.3.2 --> 7.3.3

* [autonomous experimenter](api/autonomous-experimenter.md):
  instrument_func for GP and fvGP is now optional, in case gpCAM is used for analysis of an existing dataset.

## 7.2.5 --> 7.3.0

* [autonomous experimenter](api/autonomous-experimenter.md):
  append_data_after_send --> communicate_full_dataset; with the opposite effect. default: communicate_full_dataset = False

* [autonomous experimenter](api/autonomous-experimenter.md):
  new attribute: use_inv; allows the user to opt for covariance inverse based calculation of the posterior covariance; can lead to speedup (same for fvgp 3.2.0, gp_optimizer and fvgp_optimizer), Explanation: it is now possible to direct gpCAM and fvgp to compute and store the inverse of the covariance whenever it is updated. That can lead to speedup when computing time is dominated by computing the uncertainty.

* The default acquisition function is now "variance" instead of covariance. The variance only needs the diagonal entries of the posterior covariance matrix which (combined with the stored inverse) can be computed very efficiently.

* fvgp: posterior_covariance(x_iset, variance_only = False)

* Other changes: more informative error messages when wrong data formats are communicated, speedups by only computing variance, not covariance

* the instrument function now gets an extra input, to communicate some settings def instrument(data) --> def instrument(data, instrument_dict)

* the instrument_dict can be injected into the initialization of the autonomous experimenter

## 7.0.0 --> 7.2.2 - 7.2.5

* Minor changes to asynchronous training in the [autonomous experimenter](api/autonomous-experimenter.md) and
  [gp_optimizer](api/gpOptimizer.md) classes. Have a look at the docs.

## 6.0.5 --> 7.0.0

In version 7, the autonomous loop is included in the API, so does not have to be implemented by the user using the gp_optimizer class.

**New**:

* class [AutonomousExperimenterGP](autonomous-experimenter.md (see the documentation for more explanation)

* class [AutonomousExperimenterFvGP](autonomous-experimenter.md (see the documentation for more explanation)

* overall there is more distinction between single-task and multi-task (fv)GPs
  ([gp_optimizer](api/gpOptimizer.md) and [fvgp](api/fvgpOptimizer.md))

**changes to gp_optimizer (and therefore fvgp_optimizer)**:

* reminder: async_train_gp() --> train_gp_async()

* added parameter: ask(optimization_x0); gives the option to provide starting positions for local and hgdl optimization

**Additional Notes**:

The AutonomousExperimenter receives an optional lambda function to decide what method to use to ask() for new data

ask() has to receive the cost function for it to be used

tell() does not append data anymore but only receives full datasets

Communicating an old dataset can now be done via numpy arrays or by providing the data structure from a previous run. In the latter case, only one filename has to be supplied, the hyperparameters are defined in the data. 

## 6.0.4 --> 6.0.5

async_train_gp() --> train_gp_async()