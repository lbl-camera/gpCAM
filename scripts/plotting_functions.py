import numpy as np
import matplotlib.pyplot as plt


def plot_output_function(gp_optimizer_obj):
    import matplotlib.pyplot as plt
    x = np.linspace(plot_oput_dim[0][0],plot_oput_dim[0][1],resolution[0])
    x = list(x)
    for i in range(len(x)): x[i] = [x[i]]
    x =np.asarray(x)
    model = np.zeros((x.shape))
    variance = np.zeros((x.shape))

    res= gp_optimizer_obj.gp.compute_posterior_fvGP_pdf(x_input = np.array([plot_iput_dim[0][0],plot_iput_dim[1][0]]),x_output = x, 
            compute_posterior_covariances = True)
    model = res["posterior means"][0]
    variance = res["posterior covariances"][:,0,0]

    plt.plot(x,model, linewidth = 2, label = 'mean function')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()



def plot_1d_function(gp_optimizer_obj):
    import matplotlib.pyplot as plt
    x = np.linspace(plot_iput_dim[plot_indices[0]][0],plot_iput_dim[plot_indices[0]][1],resolution[0])
    model = np.zeros((x.shape))
    variance = np.zeros((x.shape))

    for i in range(len(x)):
        res= gp_optimizer_obj.gp.compute_posterior_fvGP_pdf(np.array([x[i]]),
                compute_posterior_covariances = True)
        model[i] = res["posterior means"][0,0]
        variance[i] = res["posterior covariances"][0]

    plt.plot(x,model, linewidth = 2, label = 'mean function')
    plt.fill_between(x, model-3.0*np.sqrt(variance), model+3.0*np.sqrt(variance), color = 'green', alpha = 0.5, label = "95% confidence interval")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(gp_optimizer_obj.points, gp_optimizer_obj.values, color = 'black')
    plt.legend()
    plt.show()

def plot_2d_function(gp_optimizer_obj):
    plot_iput_dim = [[-5,5],[-5,5]]
    resolution = [100,100]
    xlabel = "x_label"
    ylabel = "y_label"
    zlabel = "z_label"
    gp_mean = True
    gp_var = False
    gp_mean_grad = False,
    objective_function = False
    costs = False
    entropy = False
    print("plotting dims:", plot_iput_dim)
    l = [len(plot_iput_dim[i]) for i in range(len(plot_iput_dim))]
    plot_indices = [i for i, x in enumerate(l) if x == 2]
    slice_indices = [i for i, x in enumerate(l) if x == 1]


    ##plot the current model
    import matplotlib.pyplot as plt
    x = np.linspace(plot_iput_dim[0][0],plot_iput_dim[0][1],resolution[0])
    y = np.linspace(plot_iput_dim[1][0],plot_iput_dim[1][1],resolution[0])
    x,y = np.meshgrid(x,y)
    model = np.zeros((x.shape))
    variance = np.zeros((x.shape))
    obj = np.zeros((x.shape))
    cost = np.zeros((x.shape))
    gp_grad = np.zeros((x.shape))
    entropy_array = np.zeros((x.shape))

    points = []
    for i in range(len(x)):
        for j in range(len(y)):
            point = np.zeros((len(plot_iput_dim)))
            point[plot_indices[0]] = x[i,j]
            point[plot_indices[1]] = y[i,j]
            for k in range(len(slice_indices)):
                point[slice_indices[k]] = plot_iput_dim[slice_indices[k]][0]
            points.append(point)

    points = np.asarray(points)
    index = 0
    for i in range(len(x)):
        for j in range(len(y)):
            res = gp_optimizer_obj.gp.compute_posterior_fvGP_pdf(np.array([points[index]]), np.array([[0]]),
            compute_posterior_covariances = True)
            aa = res["posterior means"][0]
            bb = res["posterior covariances"][0]
            model[i,j] = aa
            variance[i,j] = bb
            index += 1
    fig = plt.figure(1)
    hb = plt.pcolor(x, y,model, cmap='inferno')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("gp mean model function")
    cb = fig.colorbar(hb)
    cb.set_label(zlabel)

    fig = plt.figure(2)
    hb = plt.pcolor(x, y,variance, cmap='inferno')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(gp_optimizer_obj.points[:,plot_indices[0]], gp_optimizer_obj.points[:,plot_indices[1]])
    plt.title("gp variance function")
    cb = fig.colorbar(hb)
    plt.show()

def plot_series_curve(gp_optimizer_obj):
    number_of_modeled_curves = 1000
    alpha_exponent = 3.0

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    point = []
    model = []

    low = [plot_iput_dim[i][0] for i in range(len(plot_iput_dim))]
    high= [plot_iput_dim[i][1] for i in range(len(plot_iput_dim))]
    N = number_of_modeled_curves
    res = np.zeros((N))
    for i in range(N):
        a = np.random.uniform(low,high,len(plot_iput_dim))
        point.append(a)
        model.append(gp_optimizer_obj.gp.compute_posterior_fvGP_pdf(a)['means'][0,0])

    model = np.asarray(model)
    point = np.asarray(point)
    model_min = np.min(model)
    model_max = np.max(model)
    norm = mpl.colors.Normalize(vmin=model_min, vmax = model_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])
    for i in range(N):
        plt.plot(point[i],c = cmap.to_rgba(model[i]),alpha = ((model[i]-model_min)/(model_max-model_min))**alpha_exponent)
    plt.colorbar(cmap, ticks=[np.min(model),np.max(model)],label = zlabel)
    plt.show()

