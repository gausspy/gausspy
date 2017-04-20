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



[agd_object, science_data_path, ilist] = pickle.load(open('batchdecomp_temp.pickle'))
agd_data = pickle.load(open(science_data_path))
if ilist == None: ilist = np.arange(len(agd_data['x_values']))


def decompose_one(i):
    print '   ---->  ', i 
    result = GaussianDecomposer.decompose(agd_object, 
                                          agd_data['x_values'][i], 
                                          agd_data['data_list'][i], 
                                          agd_data['errors'][i])
    return result

def decompose_double(i):
    print '   ---->  ', i , ' double'
    result = GaussianDecomposer.decompose_double(agd_object, 
                                          agd_data['x_values'][i], 
                                          agd_data['data_list'][i],
					  agd_data['x_values_em'][i],
				  	  agd_data['data_list_em'][i], 
                                          agd_data['errors'][i],
                                          agd_data['errors_em'][i])

    return result


def func():

 # Multiprocessing code
    ncpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(ncpus, init_worker)
    if agd_object.p['verbose']: print 'N CPUs: ', ncpus
    try:
	if agd_object.p['alpha_em'] is not None:
	    results_list = p.map(decompose_double, ilist)
	else:
	    results_list = p.map(decompose_one, ilist)

    except KeyboardInterrupt:
        print "KeyboardInterrupt... quitting."
        p.terminate()
        quit()
    p.close()
    del p
    return results_list


