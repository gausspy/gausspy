import pickle
import multiprocessing
import signal
import numpy as np
import AGD_decomposer
from gp import GaussianDecomposer

# BUG FIXED:  UnboundLocalError: local variable 'result' referenced before assignment
# Caused by using more threads than spectra.

def init_worker():
    """ Worker initializer to ignore Keyboard interrupt """
    signal.signal(signal.SIGINT, signal.SIG_IGN)



def init():
    global agd_object, science_data_path, ilist, agd_data
    [agd_object, science_data_path, ilist] = pickle.load(open('batchdecomp_temp.pickle'))
    agd_data = pickle.load(open(science_data_path))
    if ilist == None: ilist = np.arange(len(agd_data['x_values']))


def decompose_one(i):
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



def func():

 # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(ncpus, init_worker)
    if agd_object.p['verbose']: print 'N CPUs: ', ncpus
    try:
        results_list = p.map(decompose_one, ilist, chunksize=1)

    except KeyboardInterrupt:
        print "KeyboardInterrupt... quitting."
        p.terminate()
        quit()
    p.close()
    del p
    return results_list


