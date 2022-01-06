---
banner: ../_static/topography.png
banner_brightness: .8
---

# US Topography
The data for this script can be
[downloaded here](https://drive.google.com/file/d/1BMNsdv168PoxNCHsNWR_znpDswjdFxXI/view?usp=sharing)
This script uses version 7 of gpCAM
```python
import numpy as np
from gpcam.gp_optimizer import GPOptimizer
import matplotlib.pyplot as plt
from numpy.random import default_rng

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

a = np.load("us_topo.npy")
rng = default_rng()
ind = rng.choice(len(a)-1, size=3000, replace=False)
points = a[ind,0:2]
values = a[ind,2:3]
print("x_min ", np.min(points[:,0])," x_max ",np.max(points[:,0]))
print("y_min ", np.min(points[:,1])," y_max ",np.max(points[:,1]))
print("length of data set: ", len(points))

index_set_bounds = np.array([[0,99],[0,248]])
hyperparameter_bounds = np.array([[0.001,1e9],[1,1000],[1,1000]])
hps_guess = np.array([4.71907062e+06, 4.07439017e+02, 3.59068120e+02])

###################################################################################
gp = GPOptimizer(2, index_set_bounds)
gp.tell(points,values)
gp.init_gp(hps_guess)
gp.train_gp(hyperparameter_bounds,pop_size = 20,tolerance = 1e-6,max_iter = 2)

x_pred = np.empty((10000,2))
counter = 0
x = np.linspace(0,99,100)
y = np.linspace(0,248,100)

for i in x:
 for j in y:
   x_pred[counter] = np.array([i,j])
   counter += 1

res1 = gp.posterior_mean(x_pred)
res2 = gp.posterior_covariance(x_pred)
#res3 = gp.gp.shannon_information_gain(x_pred)
X,Y = np.meshgrid(x,y)

PM = np.reshape(res1["f(x)"],(100,100))
PV = np.reshape(res2["v(x)"],(100,100))
plot(X,Y,PM)
plot(X,Y,PV)

next = gp.ask(position = None, n = 1, acquisition_function = "covariance", bounds = None,
             method = "global", pop_size = 50, max_iter = 20,
             tol = 10e-6, dask_client = False)
print(next)
```
:::{figure-md} us-topography-example-plot
<img src='_static/us-topography-example-plot.png' alt='gpCAM with US Topography data''>
:::
