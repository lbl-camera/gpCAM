---
banner: _static/OMEGA.jpg
banner_brightness: .5
---
# Common Bugs and Fixes

| **Error Message** | **Solution** |
|---|---|
| Key Error: "Does not support option: 'fastmath' | A numba error. Please update numba |
| Matrix Singular | Normally that means that data points are too close and different for the given kernel definition. Try using the exponential kernel, check for duplicates in the data or add more noise to the data |
| Value Error: Object arrays cannot be loaded when allow_pickle = False| You probably used a hand-made python function that loads a file without specifying allow_pickle = True |
| General installation issues | update pip, rerun installation, pip install wheel |
| ERROR: Failed building wheel for psutil | Rerun installation |
| RuntimeError: <br />An attempt has been made to start a new process before the<br />current process has finished its bootstrapping phase.<br />This probably means that you are not using fork to start your<br />child processes and you have forgotten to use the proper idiom<br />in the main module:<br /><br /><pre>if \_\_name\_\_ == '\_\_main\_\_':<br>    freeze_support()<br>    ...</pre>The "freeze_support()" line can be omitted if the program<br />is not going to be frozen to produce an executable.<br /><br />distributed.nanny - WARNING - Restarting worker<br />Traceback (most recent call last):<br />File "<string>", line 1, in <module><br />Traceback (most recent call last):<br />File "/usr/lib/python3.8/multiprocessing/spawn.py", line 116, in spawn_main<br />File "zmq_test.py", line 75, in <module><br />exitcode = _main(fd, parent_sentinel)<br />File "/usr/lib/python3.8/multiprocessing/spawn.py", line 125, in _main<br />prepare(preparation_data)<br />File "/usr/lib/python3.8/multiprocessing/spawn.py", line 236, in prepare<br />[and lot more DASK stuff]<br /> | Put all your gpCAM code in <pre>def main():<br />    # all the gpcam code<br />    ...<br /><br />if \_\_name\_\_ == "\_\_main\_\_"<br />    main() |