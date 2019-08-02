# @Author: Robert Lindner
# @Date:   Nov 10, 2014
# @Filename: batch_decomposition.py
# @Last modified by:   riener
# @Last modified time: 2019-03-16T15:59:58+01:00

import pickle
import multiprocessing
import signal
import numpy as np

# from . import AGD_decomposer
from .gp import GaussianDecomposer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# BUG FIXED:  UnboundLocalError: local variable 'result' referenced before assignment
# Caused by using more threads than spectra.


def init_worker():
    """Worker initializer to ignore Keyboard interrupt."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def init(*args):
    global agd_object, science_data_path, ilist, agd_data
    if args:
        [agd_object, agd_data, ilist] = args[0]
    else:
        [agd_object, science_data_path, ilist] = pickle.load(
            open("batchdecomp_temp.pickle", "rb"), encoding="latin1"
        )
        agd_data = pickle.load(open(science_data_path, "rb"), encoding="latin1")
    if ilist is None:
        ilist = np.arange(len(agd_data["data_list"]))


def decompose_one(i):
    print("   ---->  ", i)
    result = GaussianDecomposer.decompose(
        agd_object,
        agd_data["x_values"][i],
        agd_data["data_list"][i],
        agd_data["errors"][i],
    )
    return result


def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=1):
    """A parallel version of the map function with a progress bar.
    Args:
        array (array-like): An array to iterate over.
        function (function): A python function to apply to the array elements
        n_jobs (int, default=16): The number of cores to use
        use_kwargs (boolean, default=False): Whether to consider the elements
            as dictionaries of keyword arguments to function
        front_num (int, default=3): The number of iterations to run serially
            before kicking off the parallel job. Useful for catching bugs
    Returns:
        [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [
            function(**a) if use_kwargs else function(a) for a in array[:front_num]
        ]
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [
            function(**a) if use_kwargs else function(a)
            for a in tqdm(array[front_num:])
        ]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            "total": len(futures),
            "unit": "it",
            "unit_scale": True,
            "leave": True,
        }
        # Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


def func(use_ncpus=None):
    # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    if use_ncpus is None:
        use_ncpus = int(0.75 * ncpus)
    # p = multiprocessing.Pool(ncpus, init_worker)
    print("using {} out of {} cpus".format(use_ncpus, ncpus))
    try:
        results_list = parallel_process(ilist, decompose_one, n_jobs=use_ncpus)
    except KeyboardInterrupt:
        print("KeyboardInterrupt... quitting.")
        quit()
    return results_list
