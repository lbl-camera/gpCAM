---
banner: ../_static/green-surface-plot.png
banner_brightness: .5
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: gpcam
  language: python
  name: python3
---

# Example for the Advanced User

Here we show the test that is included in the package in `./examples/advanced_test.ipynb`

Again, the API is changing frequently, always use `help()`.

This test is a great way to get started.
Work through the test step by step, by the end, you will have a firm handle on gpCAM.

The data can be downloaded
[here](https://drive.google.com/file/d/1jlBi6hwA1jGfenKib6xA7AVJoYf09Ze7/view()).

Try out the code 
[here](https://colab.research.google.com/drive/1-A6O3OyM7vjX6a5nhX-fJeY3h6TtnhbR?usp=sharing())
with google colab.


+++ {"id": "WExbsL_ITfK2"}

## gpCAM Test Notebook
In this notebook we will go through many features of gpCAM. Work through it 
and you are ready for your own autonomous experiment. 

```{code-cell} ipython3
:id: i1G7qc_ETfK8

####install gpcam here if you do not have already done so
#!pip install gpcam
```

+++ {"id": "djqaiOZuTfK_"}

## This first cell has nothing to do with gpCAM, it's just a function to plot some results later

```{code-cell} ipython3
:id: zob6HIlHTfLA

import plotly.graph_objects as go
import numpy as np
def plot(x,y,z,data = None):
    fig = go.Figure()
    fig.add_trace(go.Surface(x = x, y = y,z=z))
    if data is not None: 
        fig.add_trace(go.Scatter3d(x=data[:,0], y=data[:,1], z=data[:,2],
                                   mode='markers'))

    fig.update_layout(title='Posterior Mean', autosize=True,
                  width=800, height=800,
                  margin=dict(l=65, r=50, b=65, t=90))


    fig.show()
```

+++ {"id": "t8A9KF26TfLC"}

## Here we want to define some points at which we will predict, still has nothing to do with gpCAM 

```{code-cell} ipython3
:id: frkUYlSGTfLC

x_pred = np.zeros((10000,2))
x = np.linspace(0,10,100)
y = np.linspace(0,10,100)
x,y = np.meshgrid(x,y)
counter = 0
for i in  range(100):
    for j in range(100):
        x_pred[counter] = np.array([x[i,j],y[i,j]])
        counter += 1
```

+++ {"id": "50XXhzWpTfLD"}

## Let's get after it by setting up a Single-Task GP Autonomous Data Acquisition Run
### The following function are optional and already show you some advanced features

```{code-cell} ipython3
:id: INCNOuO7TfLD

def optional_acq_func(x,obj):
    #this acquisition function makes the autonomous experiment a Bayesian optimization
    a = 3.0 #3.0 for 95 percent confidence interval
    mean = obj.posterior_mean(x)["f(x)"]
    cov = obj.posterior_covariance(x)["v(x)"]
    return mean + a * cov

def optional_mean_func(gp_obj,x,hyperparameters):
    return ((x[:,0] ** 2 + x[:,1] - 11.0) ** 2 + (x[:,0] + x[:,1] ** 2 - 7.0) ** 2) * hyperparameters[-1]

def optional_cost_function(origin,x,arguments = None):
    #cost pf l1 motion in the input space
    offset = arguments["offset"]
    slope = arguments["slope"]
    d = np.abs(np.subtract(origin,x))
    c = (d * slope) + offset
    n = np.sum(c)
    return n
def optional_cost_update_function(costs, parameters):
    ###defining a cost update function might look tricky but just need a bit
    ###of tenacity. And remember, this is optional, if you have a great guess for your costs you
    ###don't need to update the costs. Also, if you don't account for costs, this funciton is not needed
    from scipy.optimize import differential_evolution as devo
    print("Cost adjustment in progress...")
    print("old cost parameters: ",parameters)
    bounds = np.array([[0.001,10],[0.0001,10]])
    ###remove outliers:
    origins = []
    points = []
    motions = []
    c   = []
    cost_per_motion = []
    for i in range(len(costs)):
        origins.append(costs[i][0])
        points.append(costs[i][1])
        motions.append(abs(costs[i][0] - costs[i][1]))
        c.append(costs[i][2])
        cost_per_motion.append(costs[i][2]/optional_cost_function(costs[i][0],costs[i][1], parameters))
    mean_costs_per_distance = np.mean(np.asarray(cost_per_motion))
    sd = np.std(np.asarray(cost_per_motion))
    for element in cost_per_motion:
        if (
            element >= mean_costs_per_distance - 2.0 * sd
            and element <= mean_costs_per_distance + 2.0 * sd
        ):
            continue
        else:
            motions.pop(cost_per_motion.index(element))
            c.pop(cost_per_motion.index(element))
            origins.pop(cost_per_motion.index(element))
            points.pop(cost_per_motion.index(element))
            cost_per_motion.pop(cost_per_motion.index(element))
    def compute_l1_cost_misfit(params, origins,points, costs):
        parameters = {"offset": params[0], "slope": params[1]}
        sum1 = 0.0
        for idx in range(len(points)):
            sum1 = sum1 + (
                (optional_cost_function(origins[idx],points[idx],parameters) - costs[idx]) ** 2)
        return sum1
    res = devo(compute_l1_cost_misfit, bounds, args = (origins, points,c), tol=1e-6, disp=True, maxiter=300, popsize=20,polish=False)
    arguments = {"offset": res["x"][0],"slope": res["x"][1:]}
    print("New cost parameters: ", arguments)
    return arguments
```

```{code-cell} ipython3
:id: GK2QxXoRTfLG

import time
from gpcam.autonomous_experimenter import AutonomousExperimenterGP

def instrument(data, instrument_dict = {}):
    print("This is the current length of the data received by gpCAM: ", len(data))
    print(instrument_dict)
    for entry in data:
        entry["value"] = np.sin(np.linalg.norm(entry["position"]))
        #entry["cost"]  = [np.array([0,0]),entry["position"],np.sum(entry["position"])]
    return data

#initialization
#feel free to try different acquisition functions, e.g. optional_acq_func, "covariance", "shannon_ig"
#note how costs are defined in for the autonomous experimenter
my_ae = AutonomousExperimenterGP(np.array([[0,10],[0,10]]),
                                 np.ones((3)),np.array([[0.001,100],[0.001,100],[0.001,100]]),
                                 init_dataset_size= 20, instrument_func = instrument,
                                 instrument_dict = {"something": 3},
                                 acq_func = "variance", #optional_acq_func, 
                                 #cost_func = optional_cost_function, 
                                 #cost_update_func = optional_cost_update_function,
                                 cost_func_params={"offset": 5.0,"slope":10.0},
                                 kernel_func = None, use_inv = True,
                                 communicate_full_dataset = False, ram_economy = True)
                                 #, prior_mean_func = optional_mean_func)


print("length of the dataset: ",len(my_ae.x))


#my_ae.train_async()                 #train asynchronously
my_ae.train(method = "global")       #or not, or both, choose between "global","local" and "hgdl"
```

```{code-cell} ipython3
:id: pfJ4iSS1TfLI

#update hyperparameters in case they are optimized asynchronously
my_ae.update_hps()
```

```{code-cell} ipython3
:id: pMqdCCZITfLI

#training and client can be killed if desired and in case they are optimized asynchronously
my_ae.kill_training()
```

+++ {"id": "LBFuHxv2TfLJ"}

## Let's see what our initial model looks like

```{code-cell} ipython3
:id: kMFtOYf8TfLJ

f = my_ae.gp_optimizer.posterior_mean(x_pred)["f(x)"]
f_re = f.reshape(100,100)

plot(x,y,f_re, data = np.column_stack([my_ae.x,my_ae.y]))
```

+++ {"id": "NAyhaFIlTfLJ"}

## Let's run the autonomus loop to 100 points

```{code-cell} ipython3
:id: Gq6_xogJTfLJ

#here we see how python's help function is used to get info about a function
help(my_ae.go)
```

```{code-cell} ipython3
:id: chPCvsBaTfLK

#run the autonomous loop
my_ae.go(N = 100, 
            retrain_async_at=[25, 30,40],
            retrain_globally_at = [],
            retrain_locally_at = [],
            acq_func_opt_setting = lambda number: "global" if number % 2 == 0 else "local",
            training_opt_callable = None,
            training_opt_max_iter = 20,
            training_opt_pop_size = 10,
            training_opt_tol      = 1e-6,
            acq_func_opt_max_iter = 20,
            acq_func_opt_pop_size = 20,
            acq_func_opt_tol      = 1e-6,
            number_of_suggested_measurements = 1,
            acq_func_opt_tol_adjust = [True,0.1])
```

+++ {"id": "caQJQv29TfLK"}






## Now let's plot the posterior mean after the experiment has concluded

```{code-cell} ipython3
:id: QS2WHmigTfLL

res = my_ae.gp_optimizer.posterior_mean(x_pred)
f = res["f(x)"]
f = f.reshape(100,100)

plot(x,y,f, data = np.column_stack([my_ae.x,my_ae.y]))
```

+++ {"id": "SITFwcEDTfLM"}

## Running a Multi-Task GP Autonomous Data Acquisition
This example uses 21 (!) dim robot data and 7 tasks, which you can all use or pick a subset of them

```{code-cell} ipython3
:id: F3sy4rMpTfLM

##prepare some data
import numpy as np
from scipy.interpolate import griddata
data = np.load("sarcos.npy")
print(data.shape)
x = data[:,0:21]
y = data[:,21:23]
```

```{code-cell} ipython3
:id: TB5mfakVTfLM


from gpcam.autonomous_experimenter import AutonomousExperimenterFvGP

def instrument(data, instrument_dict = {}):
    print("Suggested by gpCAM: ", data)
    print("")
    for entry in data:
        entry["values"] = griddata(x,y,entry["position"],method = "nearest", fill_value = 0)[0]
        entry["value positions"] = np.array([[0],[1]])
    return data

def recommended_acq_func(x,obj):
    #multi-tast autonomous experiments should make use of a user-defined acquisition function to
    #make full use of the surrogate and the uncertainty in all tasks.
    a = 3.0 #3.0 for ~95 percent confidence interval
    x = np.block([[x,np.zeros((len(x))).reshape(-1,1)],[x,np.ones((len(x))).reshape(-1,1)]]) #for task 0 and 1
    mean = obj.posterior_mean(x)["f(x)"]
    cov = obj.posterior_covariance(x)["v(x)"]
    #it takes a little bit of wiggling to get the tasks seperated and then merged again...
    task0index = np.where(x[:,21] == 0.)[0]
    task1index = np.where(x[:,21] == 1.)[0]
    mean_task0 = mean[task0index]
    mean_task1 = mean[task1index]
    cov_task0 = cov[task0index]
    cov_task1 = cov[task1index]
    mean = np.column_stack([mean_task0,mean_task1])
    cov  = np.column_stack([cov_task0 ,cov_task1 ])
    #and now we are interested in the l2 norm of the mean and variance at each input location.
    return np.linalg.norm(mean, axis = 1) + a * np.linalg.norm(cov,axis = 1)


input_s = np.array([np.array([np.min(x[:,i]),np.max(x[:,i])]) for i in range(len(x[0]))])
print("index set (input space) bounds:")
print(input_s)
print("hps bounds:")
hps_bounds = np.empty((22,2))
hps_bounds[:,0] = 0.0001
hps_bounds[:,1] = 100.0
hps_bounds[0] = np.array([0.0001, 10000])
print(hps_bounds)
print("shape of y: ")
print(y.shape)

my_fvae = AutonomousExperimenterFvGP(input_s,2,1,np.ones((22)), hps_bounds,
                                     init_dataset_size= 10, instrument_func = instrument, acq_func=recommended_acq_func)
my_fvae.train()
my_fvae.go(N = 50)

```

+++ {"id": "GvXb7A8BTfLN"}

## Plotting the 0th task in a 2d slice

```{code-cell} ipython3
:id: n2giEi4mTfLN

x_pred = np.zeros((10000,21))
x = np.linspace(input_s[0,0],input_s[0,1],100)
y = np.linspace(input_s[1,0],input_s[1,1],100)
x,y = np.meshgrid(x,y)
counter = 0
for i in  range(100):
    for j in range(100):
        x_pred[counter] = np.zeros((21))
        x_pred[counter,[0,1]] = np.array([x[i,j],y[i,j]])
        counter += 1
res = my_fvae.gp_optimizer.posterior_mean(x_pred)
f = res["f(x)"]
f = f.reshape(100,100)

plot(x,y,f)
```

+++ {"id": "_XUwvB9mTfLN"}

## Back to a single task: using the GPOptimizer class directly gives you some more flexibility
We will show more soon!

```{code-cell} ipython3
:id: 4iC90_hVTfLO

#/usr/bin/env python
import numpy as np
from gpcam.gp_optimizer import GPOptimizer

#initialize some data
x_data = np.random.uniform(size = (100,1))
y_data = np.sin(x_data)[:,0]


#initialize the GPOptimizer
my_gpo = GPOptimizer(1,np.array([[0,1]]))
#tell() it some data
my_gpo.tell(x_data,y_data)
#initialize a GP ...
my_gpo.init_gp(np.ones(2))
#and train it
my_gpo.train_gp(np.array([[0.001,100],[0.001,100]]))

#let's make a prediction
print(my_gpo.posterior_mean(np.array([0.44])))

#now we can ask for a new point
r = my_gpo.ask()
print(r)
#putting the ask() in a loop and updating the data will
#give you all you need for your autonomous experiment
```

```{code-cell} ipython3
:id: mU3Bq_75TfLP


```
