# Advanced Use of gpCAM

The advanced use of gpCAM is about communicating domain knowledge in the form of kernel, acquisition and mean functions, and optimization constraints.

## Prior-Mean Functions to Communicate Trends 

Often times an overall trend of the model is known in absolute terms or in parametric form. In that case, the user may define their own prior mean function following the example below.

```python
def himmel_blau(gp_obj,x,hyperparameters):
    return (x[:,0] ** 2 + x[:,1] - 11.0) ** 2 + (x[:,0] + x[:,1] ** 2 - 7.0) ** 2
```

## Tailored Acquisition Functions for Feature Finding 

The acquisition function uses the output of a Gaussian process to steer the experiment or simulation to high-value regions of the search space. You can find an example below.

```python
def upper_confidence_bounds(x,obj):
    a = 3.0 #3.0 for 95 percent confidence interval
    mean = obj.posterior_mean(x)["f(x)"]
    cov = obj.posterior_covariance(x)["v(x)"]
    return mean + a * np.sqrt(cov)   ##which is 1-d numpy array
```

## Tailored Kernel Functions for Hard Constraints on the Posterior Mean

Kernel functions are a tremendously powerful tool to communicate hard constraints to the Gaussian process. Examples include the order of differentiability, periodicity, and symmetry of the model function. The kernel can be defined in the way presented below. 

```python
def kernel_l2_single_task(x1,x2,hyperparameters,obj):
    hps = hyperparameters
    distance_matrix = np.zeros((len(x1),len(x2)))
    
    for i in range(len(x1[0])-1):
           distance_matrix += abs(np.subtract.outer(x1[:,i],x2[:,i])/hps[1+i])**2
        
    distance_matrix = np.sqrt(distance_matrix)
    
    return   hps[0] *  obj.matern_kernel_diff1(distance_matrix,1)
```

## Tailored Cost Functions for Optimizing Data Acquisition when Costs are Present

Cost functions are very useful when the main effort of exploration does not come from the data acquisition itself but from the motion through the search space. gpCAM can use cost and cost update functions. You can find examples for both below. If costs are recorded during data acquisition, gpCAM can use them to update the cost function repeatedly.

```python
def l2_cost(origin,x,arguments = None):
    offset = arguments["offset"]
    slope = arguments["slope"]
    return slope*np.linalg.norm(np.abs(np.subtract(origin,x)), axis = 1)+offset
```

```python
def update_l2_cost_function(costs, bounds, parameters):
    print("Cost adjustment in progress...")
    print("old cost parameters: ",parameters)
    ###remove outliers:
    origins = []
    points = []
    motions = []
    c   = []
    cost_per_motion = []
    
    for i in range(len(costs)):
        origins.append(costs[i]["origin"])
        points.append(costs[i]["point"])
        motions.append(abs(costs[i]["origin"] - costs[i]["point"]))
        c.append(costs[i]["cost"])
        cost_per_motion.append(costs[i]["cost"]/l2_cost(costs[i]["origin"],costs[i]["point"], parameters))
        
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
            
    res = devo(compute_l2_cost_misfit, bounds, args = (origins, points,c), tol=1e-6, disp=True, maxiter=300, popsize=20, polish=False)
    arguments = {"offset": res["x"][0],"slope": res["x"][1]}
    print("New cost parameters: ", arguments)
    return arguments
```

## Constrained Optimization

It is now possible to create hgdl.constraints.NonLinearConstraint object instances and communicate them to gp_optimizer.train and gp_optimizer.train_async().
Setting this up is a little tricky but potentially very beneficial.
