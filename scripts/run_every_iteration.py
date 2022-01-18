import numpy as np


def write_vtk_file(gp_optimizer_obj):
    print("This will be printed in every iteration of gpCAM")
    plot_dim = [[0,28],[0,41]]
    resolution = [100,100]
    print("plotting dims:", plot_dim)
    l = [len(plot_dim[i]) for i in range(len(plot_dim))]
    plot_indices = [i for i, x in enumerate(l) if x == 2]
    slice_indices = [i for i, x in enumerate(l) if x == 1]
    file_name = "vid.csv."+str(len(gp_optimizer_obj.points)).zfill(5)
    file_name_p = "vid_p.csv."+str(len(gp_optimizer_obj.points)).zfill(5)
    print("writing csv file", file_name)
    print(plot_dim[plot_indices[0]][0],plot_dim[plot_indices[0]][1],resolution[0])
    print(plot_dim[plot_indices[1]][0],plot_dim[plot_indices[1]][1],resolution[1])


    x = np.linspace(plot_dim[plot_indices[0]][0],plot_dim[plot_indices[0]][1],resolution[0])
    y = np.linspace(plot_dim[plot_indices[1]][0],plot_dim[plot_indices[1]][1],resolution[1])
    mean = np.zeros((len(x)*len(y)))
    points = np.zeros((len(x)*len(y),2))
    print("plot indices:", plot_indices)
    print("slice indices:", slice_indices)
    index = 0
    for i in range(len(x)):
        for j in range(len(y)):
            point = np.array([x[i],y[j]])
            points[index,0] = point[0]
            points[index,1] = point[1]
            mean[index] = gp_optimizer_obj.posterio_mean(point)["f()"]
            index += 1
    l = np.array([points[:,0],points[:,1],model])
    np.savetxt(file_name,l.T, delimiter = ",",header = 'x coord, y coord, scalar')
    np.savetxt(file_name_p,gp_optimizer_obj.points, delimiter = ",",header = 'x coord, y coord, z coord')
