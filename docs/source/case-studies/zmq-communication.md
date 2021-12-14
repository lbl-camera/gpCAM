# ZMQ Communication
**This example shows how communication can be managed via zmq when gpCAM is run on a server.**
1. Start this script on your instrument machine.
```python
import zmq
import numpy as np
import pickle
import zlib

port = "6885" #(just an example, choose a free one)
context = zmq.Context()
socket = context.socket(zmq.PAIR)
ip  = [ip of machine where gpCAM is running]
socket.connect("tcp://ip:%s" % port )

def instrument(data):
    ###here you can put all the code that actually talks to the instrument and acquires the data
    ###hew we just use an np.sin() as example    
    for entry in data:
        entry["value"] = np.sin(np.linalg.norm(entry["position"]))
    return data

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

while True:
    waiting_for_message = True
    print("I am waiting for messages from the server, a.k.a. gpCAM's command")
    msg = recv_zipped_pickle(socket,flags = 0)
    print("I have received gpCAM's command, which is: ", msg)
    print("I will now collect the requested data")
    data = instrument(msg)
    print("I have acquired the data and I am sending the result back.")
    send_zipped_pickle(data, socket, flags=0, protocol=-1)
```
2. Run this script (after adapting to your needs) on the server side.
```python
import numpy as np
import zmq
import pickle
import zlib
import dask.distributed
from gpcam.autonomous_experimenter import AutonomousExperimenterGP

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

def comm_via_zmq(data,instrument_dict = {}):
        socket = instrument_dict["socket"]
        send_zipped_pickle(data,socket)
        print("gpCAM has sent data of length: ", len(data))
        print("=================================")
        data = recv_zipped_pickle(socket)
        print("gpCAM has received data of length: ", len(data))
        return data

def main():
    #for zmq communication, when gpCAM runs on a remote server##
    port = "6885" ##make sure it's the same port
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:%s" % port)

    parameters = np.array([[3.0,45.8],
                           [4.0,47.0]])

    ##set up some hyperparameters, if you have no idea, set them to 1 and make the training bounds large
    init_hyperparameters = np.array([1,1,1])
    hyperparameter_bounds =  np.array([[0.01,100],[0.01,100.0],[0.01,100]])

    ##let's initialize the autonomous experimenter ...
    my_ae = AutonomousExperimenterGP(parameters, comm_via_zmq, init_hyperparameters,
            hyperparameter_bounds,  init_dataset_size=10,instrument_dict = {"socket":socket})
    #...train...
    my_ae.train()

    #...and run. That's it. You successfully executed an autonomous experiment.
    my_ae.go(N = 100)
```