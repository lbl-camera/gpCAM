from dask.distributed import Client
import socket
import time
import numpy
import numpy as np
import time
import argparse
import datetime
import time
import sys
from dask.distributed import performance_report
import torch
from fvgp.gp import GP
import numpy as np
import scipy.sparse

def normalize(v):
    v = v - np.min(v)
    v = v/np.max(v)
    return v


"""
This run script is called by the HPC_jobscript.sl that is provided in the examples folder.
"""



def main():
    start_time = time.time()
    print("inputs to the run script: ",sys.argv, flush = True)
    print("port: ", str(sys.argv[1]), flush = True)
    client = Client(str(sys.argv[1]))
    client.wait_for_workers(int(sys.argv[2]))
    print("Client is ready", flush = True)
    print(datetime.datetime.now().isoformat())
    print("client received: ", client, flush = True)

    print("Everything is ready to call gp2Scale after ", time.time() - start_time, flush = True)
    print("Number of GPUs per Node: ", torch.cuda.device_count())



    input_dim = 3
    station_locations = np.load("station_coord.npy")
    temperatures = np.load("data.npy")
    N = len(station_locations) * len(temperatures)
    x_data = np.zeros((N,3))
    y_data = np.zeros((N))
    count  = 0
    for i in range(len(temperatures)):
        for j in range(len(temperatures[0])):
            x_data[count] = np.array([station_locations[j,0],station_locations[j,1],float(i)])
            y_data[count] = temperatures[i,j]
            count += 1

    non_nan_indices = np.where(y_data == y_data)  ###nans in data
    x_data = x_data[non_nan_indices]
    y_data = y_data[non_nan_indices]
    x_data = x_data[::100]  ##1000: about 50 000 points; 100: 500 000; 10: 5 million
    y_data = y_data[::100]

    x_data[:,0] = normalize(x_data[:,0])
    x_data[:,1] = normalize(x_data[:,1])
    x_data[:,2] = normalize(x_data[:,2]) 
    print(np.min(x_data[:,0]),np.max(x_data[:,0]))
    print(np.min(x_data[:,1]),np.max(x_data[:,1]))
    print(np.min(x_data[:,2]),np.max(x_data[:,2]))

    N = len(x_data)
    hps_n = 42
    hps_bounds = np.array([[0.1,10.],    ##signal var of stat kernel
                           [0.001,0.05],    ##length scale for stat kernel
                           [0.001,0.05],     ##length scale for stat kernel
                           [0.001,0.05],     ##length scale for stat kernel
                           ])

    init_hps = np.random.uniform(size = len(hps_bounds), low = hps_bounds[:,0], high = hps_bounds[:,1])

    print(init_hps)
    print(hps_bounds)
    st = time.time()

    my_gp2S = GP(3, x_data, y_data, init_hps,
            gp2Scale = True, gp2Scale_batch_size= 10000, info = True, gp2Scale_dask_client = client)



    print("Initialization done after: ",time.time() - st," seconds")
    print("===============")
    print("===============")
    print("===============")

    my_gp2S.train(hps_bounds, max_iter = 20)


if __name__ == '__main__':
    main()


