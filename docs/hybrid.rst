.. _hybrid:

============================================
Application: 21 cm Absorption & Emission Fit
============================================

Simultaneous Decomposition
--------------------------

One application of AGD is to decompose 21 cm spectral lines into Gaussian
components. To constrain the excitation temperature and column density of these
components, we need to know their absorption and emission properties. We have
developed a hybrid AGD method for simultaneously decomposing 21cm absorption
and emission. This method was first used in `Murray et al. 2018
<https://ui.adsabs.harvard.edu/abs/2018ApJS..238...14M/abstract>`_.

To start, we need to store all data in a format readable by GaussPy. First,
store the absorption spectrum, channel array and uncertainty arrays
(e.g., as discussed in :ref:`simple-example-tutorial`) in a Python dictionary:

.. code-block:: python

    # Initialize
    data = {}

    # Enter absorption data into AGD dataset
    data['data_list'] = data.get('data_list', []) + [absorption_spectrum]
    data['x_values'] = data.get('x_values', []) + [absorption_channels]
    data['errors'] = data.get('errors', []) + [absorption_errors]

Then, store the emission spectrum, channel array and uncertainty arrays (denoted
by keys with _em extensions) in the same dictionary and write it to disk:

.. code-block:: python

    import pickle

    # Enter emission data into AGD dataset
    data['data_list'] = data.get('data_list_em', []) + [absorption_spectrum]
    data['x_values'] = data.get('x_values_em', []) + [absorption_channels]
    data['errors'] = data.get('errors_em', []) + [absorption_errors]

    # Write input data to file
    data_filename = 'emission_absorption_data.pickle'
    pickle.dump(data, open(data_filename, 'wb'))

Next, we will load the data back in and decompose using the hybrid method. As
described in `Murray et al. 2018
<https://ui.adsabs.harvard.edu/abs/2018ApJS..238...14M/abstract>`_, this method
was developed for the specific case of estimating physical properties of
neutral hydrogen structures, and the steps are as follows:

1. First, the absorption spectrum is decomposed in the classic AGD manner, either
in the one-phase or two-phase versions (implemented by the trained values of
``alpha1`` and ``alpha2``). This results in N absorption components.

2. The N absorption components are then fitted to the emission spectrum. Their
Gaussian widths and mean positions are allowed to vary by +/- a small percentage (see
settings below) and the amplitudes are allowed to vary freely.

3. This initial fit is subtracted from the emission spectrum to produce a first
guess at the residuals. An additional K components are then fitted to these residuals,
(implemented by an additional regularization parameter for the emission fit, ``alpha_em``).
A minimum line width for these emission-only components can be specified here (see
settings below).
If any emission-only components fall within +/- a number of channels (see settings below)
from the position of an absorption component, the emission component is dropped from the fit.

4. A final least-squares fit is performed to the emission spectrum with all N
absorption and K emission components.

In this hybrid fit, there are several additional fit parameters which can be set.
These include:

1. ``alpha_em``: the regularization parameter governing the AGD fit to the
emission residuals.

2. ``max_tb``: the maximum brightness temperature absorption components are
allowed to have in emission. If set to a value, the maximum will be this value.
If set to "max" (string), the maximum Tb will be computed from the maximum
kinetic temperature assuming the absorption width and amplitude. If set to None
(the default), no max will be applied.

3. ``p_width``: the percentage by which the absorption component widths
are allowed to vary in the fit to emission (e.g., if set to 0.1, the widths
are allowed to vary by +/-10%). Default = 10%

4. ``d_mean``: the absolute number of channels the absorption components are
allowed to vary in the fit to emission (e.g., if set to 2, the mean positions
are allowed to vary by +/-2 channels). Default = 2

5. ``min_dv``: the minimum FWHM of the emission-only components (used to avoid fitting
unrealistically narrow components in emission which should be recovered in absorption).

6. ``drop_width``: if an emission component is fit by AGD within this number of
channels from the position of a fitted absorption component, it will be discarded.

7. ``SNR_em``: the signal-to-noise ratio limit for the emission components.

The fit then proceeds in a similar manner as in the standard AGD fit:

.. code-block:: python

    # Load input data
    input_data = pickle.load(open(data_filename, 'rb'))

    # Initialize
    g = gp.GaussianDecomposer()

    # Set two-phase absorption parameters
    g.set('phase', 'two')
    g.set('alpha1' , 1.12)
    g.set('alpha2' , 2.75)
    g.set('SNR_thresh', 3)
    g.set('SNR2_thresh', 3)

    # Set emission parameters
    g.set('alpha_em', 3.75)
    g.set('max_tb', None)
    g.set('p_width', 10)
    g.set('d_mean', 2)
    g.set('min_dv', 10)
    g.set('drop_width', 3)
    g.set('SNR_em', 3)

    # Decompose
    data_decomposed = g.batch_decomposition(input_file)

    # Write results to file
    output_data = 'mach_double_decomposed.pickle'
    pickle.dump(data_decomposed, open(output_data, 'wb'))
