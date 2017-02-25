# Robert Lindner
# Nov 10, 2014
# Import spectra and Gaussian component
# training data from ascii file source.

import os
import numpy as np

def loadAGDkey(agd_data, key, datapath):
    """ 
    Load contents of "filename" into 
    AGD dataset "agd_data" under key 
    """
    f = open(datapath + '/' + key)
    while True:
        l = f.readline()
        if l == '':
            break
        l = np.array(l .strip().split(), dtype='float32')
        agd_data[key] = agd_data[key] + [l]

def fromASCII(dirpath):

    # Initialize dataset
    data = {'data_list':[], 'x_values':[], 'means':[], 
             'fwhms':[], 'amplitudes':[], 'errors':[]}

    files = os.listdir(dirpath)

    for item in data.keys():
        if item not in files:
            print 'Missing: ', item        
            del data[item]
            if item != 'errors':
                print 'Required.'
                quit()    
    
    for key in data:
        loadAGDkey(data, key, dirpath)
   
        print ','.join([key ,str(len(data[key]))])


    return data

if __name__ == '__main__':
    fromASCII('data')





