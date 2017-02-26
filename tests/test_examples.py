
def test_make_science_data():

    # AGD Example 1
    # Dummpy spectra with Gaussian profiles

    import numpy as np
    import pickle

    def gaussian(amp, fwhm, mean):
        return lambda x: amp * np.exp(-(x-mean)**2/4./(fwhm/2.355)**2)

    # Data properties
    RMS = 0.05
    NCOMPS = 4
    NCHANNELS = 512
    NSPECTRA = 10
    TRAINING_SET = False
    FILENAME = 'agd_data_science.pickle'

    # Component properties
    AMP_lims = [RMS * 5, RMS * 25]
    FWHM_lims = [4.0 / NCHANNELS, NCHANNELS / 10.] # channels
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

def test_make_train_data():

    # AGD Example 1
    # Dummpy spectra with Gaussian profiles

    import numpy as np
    import pickle

    def gaussian(amp, fwhm, mean):
        return lambda x: amp * np.exp(-(x-mean)**2/4./(fwhm/2.355)**2)

    # Data properties
    RMS = 0.05
    NCOMPS = 4
    NCHANNELS = 512
    NSPECTRA = 10
    TRAINING_SET = True
    FILENAME = 'agd_data_train.pickle'

    # Component properties
    AMP_lims = [RMS * 5, RMS * 25]
    FWHM_lims = [4.0 / NCHANNELS, NCHANNELS / 10.] # channels
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

def test_train():

    import gausspy.gp as gp
    reload(gp)

    TRAINING_DATA = 'agd_data_train.pickle'

    g = gp.GaussianDecomposer()
    g.load_training_data(TRAINING_DATA)

    #One phase training
    g.set('phase', 'one')
    g.set('SNR_thresh', 5.)
    g.set('SNR2_thresh', 5.)

    g.train(alpha1_initial = 10., verbose = False, mode = 'conv',
                               learning_rate = 1.0, eps = 1.0, MAD = 0.1)
    # F1=60%, Alpha = 6.56

    #Two phase training
    g.set('phase', 'two')
    g.set('SNR_thresh', [5.,5.])
    g.set('SNR2_thresh', [5.,0.])

    g.train(alpha1_initial = 5.0, alpha2_initial = 7, plot=False,
                               verbose = False, mode = 'conv',
                               learning_rate = 1.0, eps = 1.0, MAD = 0.1)

def test_decompose():

    import gausspy.gp as gp
    import time
    import pickle

    SCIENCE_DATA = 'agd_data_science.pickle'

    g = gp.GaussianDecomposer()

    #Two phase
    g.set('phase', 'two')
    g.set('SNR_thresh', [5.,5.])
    g.set('SNR2_thresh', [5.,0.])
    g.set('alpha1', 4.19)
    g.set('alpha2', 6.45)
    g.set('mode', 'conv')

    t0 = time.time()
    new_data = g.batch_decomposition(SCIENCE_DATA)
    print 'Elapsed time [s]: ', int(time.time() - t0)
    pickle.dump(new_data, open('agd_data_science_decomposed.pickle', 'w'))
    # A=75%, 4.19, 6.45

def remove_test_files():

    import os

    files_to_remove = ['agd_data_science.pickle',
                       'agd_data_train.pickle',
                       'agd_data_science_decomposed.pickle',
                       'batchdecomp_temp.pickle',
                       ]

    for file_to_remove in files_to_remove:
        os.system('rm -rf ' + file_to_remove)

