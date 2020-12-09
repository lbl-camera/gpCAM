import numpy as np
import matplotlib.pyplot as plt


def plot_function(gp_optimizer_obj):
    #define ranges
    plot_iput_dim = [[-5,5],[-5,5]]
    #define resolution
    resolution = [100,100]
    #defie labels
    xlabel = "x_label"
    ylabel = "y_label"
    zlabel = "z_label"
    print("plotting dims:", plot_iput_dim)
    ##plot the current model
    x = np.linspace(plot_iput_dim[0][0],plot_iput_dim[0][1],resolution[0])
    y = np.linspace(plot_iput_dim[1][0],plot_iput_dim[1][1],resolution[0])
    x,y = np.meshgrid(x,y)
    mean = np.zeros((x.shape))
    var = np.zeros((x.shape))
    obj = np.zeros((x.shape))

    points = []
    for i in range(len(x)):
        for j in range(len(y)):
            p = np.array([x[i,j],y[i,j]])
            points.append(p)
            mean[i,j] = gp_optimizer_obj.gp.posterior_mean(p)["f(x)"]
            var [i,j] = gp_optimizer_obj.gp.posterior_covariance(p)["v(x)"]
            obj[i,j] = gp_optimizer_obj.evaluate_objective_function(x, objective_function = "shannon_ig")
    points = np.asarray(points)
    fig = plt.figure(1)
    hb = plt.pcolor(x, y,mean, cmap='inferno')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("gp mean model function")
    cb = fig.colorbar(hb)
    cb.set_label(zlabel)

    fig = plt.figure(2)
    hb = plt.pcolor(x, y,var, cmap='inferno')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(gp_optimizer_obj.points[:,0], gp_optimizer_obj.points[:,1])
    plt.title("gp variance function")
    cb = fig.colorbar(hb)

    fig = plt.figure(2)
    hb = plt.pcolor(x, y,obj, cmap='inferno')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(gp_optimizer_obj.points[:,0], gp_optimizer_obj.points[:,1])
    plt.title("gp objective function")
    cb = fig.colorbar(hb)

    plt.show()
