import numpy as np
import os
import time

while True:
    read_success = False
    while read_success is False:
        time.sleep(2)
        try:
            d = np.load("../data/command/command.npy", allow_pickle = True)
            read_success = True
            os.remove("../data/command/command.npy")
        except:
            print("can't read file")
            read_success = False

    print("instrument received data")
    for entry in d:
        entry["measured"] = True
        entry["measurement values"]["values"] =  np.array([np.sin(entry["position"]["x1"])])
        entry["measurement values"]["variances"] =  np.array([np.random.rand()/10.0])
        entry["measurement values"]["value positions"] =  np.array([[0.0]])
        entry["time stamp"] = time.time()

    np.save("../data/result/result.npy",d)
    print("instrument sent back data")

