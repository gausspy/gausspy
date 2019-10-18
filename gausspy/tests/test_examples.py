# import imp

def test_make_science_data():

    import numpy as np
    import pickle

    def gaussian(amp, fwhm, mean):
        return lambda x: amp * np.exp(-(x - mean) ** 2 / 4.0 / (fwhm / 2.355) ** 2)

    # Data properties
    RMS = 0.05
    NCOMPS = 3
    NCHANNELS = 512
    NSPECTRA = 10
    TRAINING_SET = False
    FILENAME = "agd_data_science.pickle"

    # Component properties
    AMP_lims = [0.5, 4.0]
    FWHM_lims = [20, 80]  # channels
    MEAN_lims = [0.25 * NCHANNELS, 0.75 * NCHANNELS]  # channels

    # Initialize
    agd_data = {}
    chan = np.arange(NCHANNELS)
    errors = chan * 0.0 + RMS  # Constant noise for all spectra

    # Begin populating data
    for i in range(NSPECTRA):
        spectrum_i = np.random.randn(NCHANNELS) * RMS

        amps = []
        fwhms = []
        means = []

        for comp in range(NCOMPS):
            # Select random values for components within specified ranges
            a = np.random.uniform(AMP_lims[0], AMP_lims[1])
            w = np.random.uniform(FWHM_lims[0], FWHM_lims[1])
            m = np.random.uniform(MEAN_lims[0], MEAN_lims[1])

            # Add Gaussian profile with the above random parameters to the spectrum
            spectrum_i += gaussian(a, w, m)(chan)

            # Append the parameters to initialized lists for storing
            amps.append(a)
            fwhms.append(w)
            means.append(m)

        # Enter results into AGD dataset
        agd_data["data_list"] = agd_data.get("data_list", []) + [spectrum_i]
        agd_data["x_values"] = agd_data.get("x_values", []) + [chan]
        agd_data["errors"] = agd_data.get("errors", []) + [errors]

        # If training data, keep answers
        if TRAINING_SET:
            agd_data["amplitudes"] = agd_data.get("amplitudes", []) + [amps]
            agd_data["fwhms"] = agd_data.get("fwhms", []) + [fwhms]
            agd_data["means"] = agd_data.get("means", []) + [means]

    pickle.dump(agd_data, open(FILENAME, "wb"))
    print("Created: ", FILENAME)


def test_make_train_data():

    import numpy as np
    import pickle

    def gaussian(amp, fwhm, mean):
        return lambda x: amp * np.exp(-(x - mean) ** 2 / 4.0 / (fwhm / 2.355) ** 2)

    # Data properties
    RMS = 0.05
    NCOMPS = 3
    NCHANNELS = 512
    NSPECTRA = 30
    TRAINING_SET = True
    FILENAME = "agd_data_train.pickle"

    # Component properties
    AMP_lims = [0.5, 4.0]
    FWHM_lims = [20, 80]  # channels
    MEAN_lims = [0.25 * NCHANNELS, 0.75 * NCHANNELS]  # channels

    # Initialize
    agd_data = {}
    chan = np.arange(NCHANNELS)
    errors = np.ones(NCHANNELS) * RMS

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
        agd_data["data_list"] = agd_data.get("data_list", []) + [spectrum_i]
        agd_data["x_values"] = agd_data.get("x_values", []) + [chan]
        agd_data["errors"] = agd_data.get("errors", []) + [errors]

        # If training data, keep answers
        if TRAINING_SET:
            agd_data["amplitudes"] = agd_data.get("amplitudes", []) + [amps]
            agd_data["fwhms"] = agd_data.get("fwhms", []) + [fwhms]
            agd_data["means"] = agd_data.get("means", []) + [means]

    pickle.dump(agd_data, open(FILENAME, "wb"))
    print("Created: ", FILENAME)

def _test_train():

    import gausspy.gp as gp

    # imp.reload(gp)

    TRAINING_DATA = "agd_data_train.pickle"

    g = gp.GaussianDecomposer()
    g.load_training_data(TRAINING_DATA)

    # One phase training
    g.set("phase", "one")
    g.set("SNR_thresh", 5.0)

    g.train(alpha1_initial=2.0)

    # Two phase training
    # g.set('phase', 'two')
    # g.set('SNR_thresh', 5.)
    # g.set('SNR2_thresh', 5.)

    # g.train(alpha1_initial = 1., alpha2_initial = 3.)


def test_onephase_decompose_python():

    import gausspy.gp as gp
    import time
    import pickle

    # imp.reload(gp)

    SCIENCE_DATA = "agd_data_science.pickle"

    g = gp.GaussianDecomposer()

    # Set defaults
    g.set("SNR_thresh", 5.0)
    g.set("alpha1", 1.02)
    g.set("phase", "one")
    g.set("mode", "python")

    t0 = time.time()
    new_data = g.batch_decomposition(SCIENCE_DATA)
    print("Elapsed time [s]: ", int(time.time() - t0))
    #pickle.dump(new_data, open("agd_data_science_decomposed.pickle", "wb"))

def test_onephase_decompose_conv():

    import gausspy.gp as gp
    import time
    import pickle

    # imp.reload(gp)

    SCIENCE_DATA = "agd_data_science.pickle"

    g = gp.GaussianDecomposer()

    # Set defaults
    g.set("SNR_thresh", 5.0)
    g.set("alpha1", 1.02)
    g.set("phase", "one")
    g.set("mode", "conv")

    t0 = time.time()
    new_data = g.batch_decomposition(SCIENCE_DATA)
    print("Elapsed time [s]: ", int(time.time() - t0))
    #pickle.dump(new_data, open("agd_data_science_decomposed.pickle", "wb"))

def test_twophase_decompose_python():

    import gausspy.gp as gp
    import time
    import pickle

    # imp.reload(gp)

    SCIENCE_DATA = "agd_data_science.pickle"

    g = gp.GaussianDecomposer()

    # Set defaults
    g.set("SNR_thresh", 5.0)
    g.set("SNR2_thresh", 5.0)
    g.set("alpha1", 1.02)
    g.set("alpha2", 2.22)
    g.set("phase", "two")
    g.set("mode", "python")

    t0 = time.time()
    new_data = g.batch_decomposition(SCIENCE_DATA)
    print("Elapsed time [s]: ", int(time.time() - t0))

def test_twophase_decompose_conv():

    import gausspy.gp as gp
    import time
    import pickle

    # imp.reload(gp)

    SCIENCE_DATA = "agd_data_science.pickle"

    g = gp.GaussianDecomposer()

    # Set defaults
    g.set("SNR_thresh", 5.0)
    g.set("SNR2_thresh", 5.0)
    g.set("alpha1", 1.02)
    g.set("alpha2", 2.22)
    g.set("phase", "two")
    g.set("mode", "conv")

    t0 = time.time()
    new_data = g.batch_decomposition(SCIENCE_DATA)
    print("Elapsed time [s]: ", int(time.time() - t0))



def test_remove_files():

    import os

    files_to_remove = [
        "agd_data_science.pickle",
        "agd_data_train.pickle",
        # "agd_data_science_decomposed.pickle",
        "batchdecomp_temp.pickle",
    ]

    for file_to_remove in files_to_remove:
        os.system("rm -rf " + file_to_remove)
