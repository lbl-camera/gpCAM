###############################################
##File Containing Data Acquisition Functions###
###############################################


import numpy as np
import os
import random
import time
from scipy.interpolate import griddata
import sys
from shutil import copyfile
import zmq
from zmq import ssh
import pickle
import zlib

#for zmq communication, when gpCAM runs on a remote server##
#port = "5555"
#context = zmq.Context()
#socket = context.socket(zmq.PAIR)
#socket.bind("tcp://*:%s" % port)
###other computer has to connect to the same port at my ip

#################################################
############test with synthetic function#########
#################################################


def synthetic_function(data):
    for idx_data in range(len(data)):
        if data[idx_data]["measured"] == True: continue
        gp_idx = data[idx_data]["function name"]
        x1 = data[idx_data]["position"]["x1"]
        x2 = data[idx_data]["position"]["x2"]
        data[idx_data]["measurement values"]["values"] = np.array(
        [himmel_blau([x1, x2])]
        )
        data[idx_data]["cost"] = {"origin": np.random.uniform(low=0.0, high=1.0, size = 2),
                                     "point": np.random.uniform(low=0.0, high=1.0, size = 2),
                                     "cost": np.random.uniform(low=0.0, high=1.0)}
        data[idx_data]["measurement values"]["variances"] = np.array([0.01])
        data[idx_data]["measurement values"]["value positions"] = np.array([[0]])
        data[idx_data]["measured"] = True
        data[idx_data]["time stamp"] = time.time()
    return data

#################################################
############use zmq##############################
#################################################
def send_zipped_pickle(obj, socket, flags=0, protocol=-1):
        """pack and compress an object with pickle and zlib."""
        pobj = pickle.dumps(obj, protocol)
        zobj = zlib.compress(pobj)
        print('zipped pickle is %i bytes' % len(zobj))
        return socket.send(zobj,flags=flags)

def recv_zipped_pickle(socket,flags = 0):
        """reconstruct a Python object sent with zipped_pickle"""
        zobj = socket.recv(flags)
        pobj = zlib.decompress(zobj)
        return pickle.loads(pobj)

####the client has to send a starting message here
#msg = recv_zipped_pickle(socket)
#print(msg)

def comm_via_zmq(data):
        send_zipped_pickle(data,socket)
        print("gpCAM has sent data of length: ", len(data))
        #print(data)
        print("=================================")
        data = recv_zipped_pickle(socket)
        print("gpCAM has received data of length: ", len(data))
        #print(data)
        #input()
        return data

#################################################
############send data in files###################
#################################################
def send_data_as_files(data):
    path_new_command = "../data/command/"
    path_new_result = "../data/result/"
    while os.path.isfile(path_new_command + "command.npy"):
        time.sleep(1)
        print("Waiting for experiment device to read and subsequently delete last command.")

    write_success = False
    read_success = False
    while write_success == False:
        try:
            np.save(path_new_command + "command", data)
            np.save(path_new_command + "command_bak", data)
            write_success = True
            print("Successfully send data set of length ",len(data)," to experiment device")
        except:
            time.sleep(1)
            print("Saving new experiment command file not successful, trying again...")
            write_success = False
    while read_success == False:
        try:
            new_data = np.load(
                path_new_result + "result.npy", encoding="ASCII", allow_pickle=True
            )
            read_success = True
            print("Successfully received data set of length ",len(new_data)," from experiment device")
            copyfile(path_new_result + "result.npy",path_new_result + "result_bak.npy")
            os.remove(path_new_result + "result.npy")
        except:
            print("New measurement values have not been written yet.")
            print("exception: ", sys.exc_info()[0])
            time.sleep(1)
            read_success = False
    return list(new_data)



#################################################
############interpolate existing data############
#################################################
def interpolate_experiment_data(data):
    space_dim = 4
    File = ""
    method = ""
    interpolate_data_array = np.load(File)
    p = interpolate_data_array[:, 0 : space_dim]
    m = interpolate_data_array[:,space_dim]
    asked_points = np.zeros((len(data),space_dim))
    for idx_data in range(len(data)):
        index = 0
        for key in data[idx_data]["position"].keys():
            asked_points[idx_data][index] = data[idx_data]["position"][key]
            index += 1
    res = griddata(p, m, asked_points, method=method, fill_value=0)
    for idx_data in range(len(data)):
        for key in data[idx_data]["position"].keys():
            data[idx_data]["measurement values"]["values"] = np.array([res[idx_data]])
            data[idx_data]["measurement values"]["variances"] = np.array([0.0000])
            data[idx_data]["measurement values"]["value positions"] = np.array([[0]])
        data[idx_data]["cost"] = 0.0
        data[idx_data]["measured"] = True
        data[idx_data]["time stamp"] = time.time()
        data[idx_data]["meta data"] = None
    return data

###################################################
####predefined test function#######################
###################################################
def himmel_blau(x):
    return (x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2

def eggholder(x):
    x1 = x[0]
    x2 = x[1]
    term1 = -(x2+47.0) * np.sin(np.sqrt(abs(x2+x1/2.0+47.0)));
    term2 = -x1 * np.sin(np.sqrt(abs(x1-(x2+47.0))));

    y = term1 + term2
    return y

def Ackley(x):
    x = np.asarray(x)
    a = 20
    b = 0.2
    c = 2.0 * np.pi
    dim = len(x)
    return - a * np.exp(-b*np.sqrt(np.sum(x**2)/dim)) -np.exp(np.sum(np.cos(c*x))/dim) + a + np.exp(1.0)

