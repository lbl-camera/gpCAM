###############################################
##File Containing Data Acquisition Functions###
###############################################


import pickle
import time
import zlib
import numpy as np


# for zmq communication, when gpCAM runs on a remote server##
# port = "5555"
# context = zmq.Context()
# socket = context.socket(zmq.PAIR)
# socket.bind("tcp://*:%s" % port)
###other computer has to connect to the same port at my ip

#################################################
############test with synthetic function#########
#################################################
from loguru import logger


def synthetic_function(data):
    for idx_data in range(len(data)):
        if data[idx_data]["measured"] == True: continue
        x1 = data[idx_data]["position"][0]
        x2 = data[idx_data]["position"][1]
        data[idx_data]["value"] = himmel_blau([x1, x2])
        data[idx_data]["cost"] = {"origin": np.random.uniform(low=0.0, high=1.0, size=2),
                                  "point": np.random.uniform(low=0.0, high=1.0, size=2),
                                  "cost": np.random.uniform(low=0.0, high=1.0)}
        data[idx_data]["variance"] = 0.01
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
    logger.debug('zipped pickle is %i bytes' % len(zobj))
    return socket.send(zobj, flags=flags)


def recv_zipped_pickle(socket, flags=0):
    """reconstruct a Python object sent with zipped_pickle"""
    zobj = socket.recv(flags)
    pobj = zlib.decompress(zobj)
    return pickle.loads(pobj)


####the client has to send a starting message here
# msg = recv_zipped_pickle(socket)
# print(msg)

def comm_via_zmq(data):
    send_zipped_pickle(data, socket)
    logger.debug("gpCAM has sent data of length: ", len(data))
    logger.debug(data)
    logger.debug("=================================")
    data = recv_zipped_pickle(socket)
    logger.debug("gpCAM has received data of length: ", len(data))
    logger.debug(data)
    # input()
    return data


#################################################
############send data in files###################
#################################################
###################################################
####predefined test function#######################
###################################################
def himmel_blau(x):
    return (x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2


def eggholder(x):
    x1 = x[0]
    x2 = x[1]
    term1 = -(x2 + 47.0) * np.sin(np.sqrt(abs(x2 + x1 / 2.0 + 47.0)));
    term2 = -x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47.0))));

    y = term1 + term2
    return y


def Ackley(x):
    x = np.asarray(x)
    a = 20
    b = 0.2
    c = 2.0 * np.pi
    dim = len(x)
    return - a * np.exp(-b * np.sqrt(np.sum(x ** 2) / dim)) - np.exp(np.sum(np.cos(c * x)) / dim) + a + np.exp(1.0)
