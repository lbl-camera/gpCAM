import uuid
import itertools
from queue import Queue
import numpy as np
import bluesky.preprocessors as bpp
import bluesky.plan_stubs as bps
import bluesky.plans as bp

from event_model import RunRouter

from bluesky_adaptive.recommendations import NoRecommendation
from bluesky_adaptive.utils import extract_event_page

gp_initialized = False

def recommender_factory(
        gp_optimizer_obj, independent_keys, dependent_keys, variance_keys, *, max_count = 10, queue = None
        ):
    """
    Generate the callback and queue for an Adaptive API backed reccomender.

    This recommends a fixed step size independent of the measurement.

    For each Run (aka Start) that the callback sees it will place
    either a recommendation or `None` into the queue.  Recommendations
    will be of a dict mapping the independent_keys to the recommended
    values and should be interpreted by the plan as a request for more
    data.  A `None` placed in the queue should be interpreted by the
    plan as in instruction to terminate the run.

    The StartDocuments in the stream must contain the key
    ``'batch_count'``.


    Parameters
    ----------
    adaptive_object : adaptive.BaseLearner
        The recommendation engine

    independent_keys : List[str]
        The names of the independent keys in the events

    dependent_keys : List[str]
        The names of the dependent keys in the events

    variance_keys : List[str]
        The names of the variance keys in the events

    max_count : int, optional
        The maximum number of measurements to take before poisoning the queue.

    queue : Queue, optional
        The communication channel for the callback to feedback to the plan.
        If not given, a new queue will be created.

    Returns
    -------
    callback : Callable[str, dict]
        This function must be subscribed to RunEngine to receive the
        document stream.

    queue : Queue
        The communication channel between the callback and the plan.  This
        is always returned (even if the user passed it in).

    """

    if queue is None:
        queue = Queue()

    def callback(name, doc):
        # TODO handle multi-stream runs with more than 1 event!
        if name == "start":
            if doc["batch_count"] > max_count:
                queue.put(None)
                return

        if name == "event_page":

            independent, measurement, variances = extract_event_page(
                independent_keys, dependent_keys, variance_keys, payload = doc["data"]
            )
            #measurement = np.array([[np.sin(independent[0,0])]])
            variances[:,:] = 0.01
            value_positions = np.zeros((1,1,1))
            print("new measurement results:")
            print("x: ", independent)
            print("y: ", measurement)
            print("variance: ", variances)
            #################################
            #####HERE########################
            #################################
            ####independent, measurement, variances: 2d numpy arrays
            ####value pos: 3d numpy arrays
            gp_optimizer_obj.tell(independent, measurement, variances = variances,
                value_positions = value_positions,append = True)

            global gp_initialized
            if gp_initialized is False: 
                gp_optimizer_obj.init_gp(np.array([1.0,1.0,1.0,1.0]))
                gp_initialized = True
            ###possible cost update here
            ###possible training here
            #print("current data set: ", gp_optimizer_obj.points)
            #print("---------------------------------------")
            #pull the next point out of the adaptive API
            try:
                #################################
                #####HERE########################
                #################################
                ##position
                ##number of asked measurements = 1
                ##bounds numpy 2d array
                ##objective_function_pop_size = 20
                ##max_iter = 20
                ##tol = 0.0001
                res = gp_optimizer_obj.ask(position = None, n = 1)
                next_point = res["x"]
                func_eval = res["f(x)"]
                next_point = next_point.squeeze()
                func_eval = func_eval.squeeze()
                print("next requested point ", next_point)
                print("======================================")

            except NoRecommendation:
                queue.put(None)
            else:
                queue.put({k: v for k, v in zip(independent_keys, next_point)})

    rr = RunRouter([lambda name, doc: ([callback], [])])
    return rr, queue


