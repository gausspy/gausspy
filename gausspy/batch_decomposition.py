import pickle
import multiprocessing
import signal
import numpy as np
import AGD_decomposer
from gp import GaussianDecomposer
from functools import partial

def init_worker():
    """ Worker initializer to ignore Keyboard interrupt """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def decompose_one(agd_object,agd_data,i):
    print '   ---->  ', i 
    try:
    	result = GaussianDecomposer.decompose(agd_object, 
                                          agd_data['x_values'][i], 
                                          agd_data['data_list'][i], 
                                          agd_data['errors'][i])
    except:
        result = {}
        result['N_components']=0
        result['initial_parameters']=[]
        result['best_fit_errors']=[]
    return result



def func(agd_object,agd_data, ilist):
 # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(ncpus, init_worker)
    if agd_object.p['verbose']: print 'N CPUs: ', ncpus
    decompose = partial(decompose_one,agd_object, agd_data)
    try:
        results_list = p.map(decompose, ilist, chunksize=1)
    except KeyboardInterrupt:
        print "KeyboardInterrupt... quitting."
        p.terminate()
        quit()
    p.close()
    del p
    return results_list


