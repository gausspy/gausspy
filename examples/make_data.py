# AGD Example 1
# Dummpy spectra with Gaussian profiles

import numpy as np
import matplotlib.pyplot as plt
import pickle

def gaussian(amp, fwhm, mean):
    return lambda x: amp * np.exp(-(x-mean)**2/4./(fwhm/2.355)**2)


# Data properties
RMS = 0.05
NCOMPS = 4
NCHANNELS = 512
NSPECTRA = 1000
TRAINING_SET = True
FILENAME = 'agd_data.pickle'

# Component properties
AMP_lims = [RMS * 5, RMS * 25]
FWHM_lims = [10, 35] # channels
MEAN_lims = [0.25 * NCHANNELS, 0.75 * NCHANNELS]


# Initialize 
agd_data = {}
chan = np.arange(NCHANNELS)
errors = chan * 0. + RMS # Constant noise for all spectra

# Begin populating data
for i in range(NSPECTRA):
    spectrum_i = np.random.randn(NCHANNELS) * RMS    
    
    # Sample random components:
    amps = np.random.rand(NCOMPS) * (AMP_lims[1] - AMP_lims[0]) + AMP_lims[0]
    fwhms = np.random.rand(NCOMPS) * (FWHM_lims[1] - FWHM_lims[0]) + FWHM_lims[0]
    means = np.random.rand(NCOMPS) * (MEAN_lims[1] - MEAN_lims[0]) + MEAN_lims[0]

    # Create spectrum
    for a, w, m in zip(amps, fwhms, means):
        spectrum_i += gaussian(a, w, m)(chan)

    # Enter results into AGD dataset
    agd_data['data_list'] = agd_data.get('data_list', []) + [spectrum_i]
    agd_data['x_values'] = agd_data.get('x_values', []) + [chan]
    agd_data['errors'] = agd_data.get('errors', []) + [errors]

    # If training data, keep answers
    if TRAINING_SET:
        agd_data['amplitudes'] = agd_data.get('amplitudes', []) + [amps]
        agd_data['fwhms'] = agd_data.get('fwhms', []) + [fwhms]
        agd_data['means'] = agd_data.get('means', []) + [means]


pickle.dump(agd_data, open(FILENAME, 'w'))
print 'Created: ', FILENAME





