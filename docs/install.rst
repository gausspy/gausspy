.. _install:

===============
Installation
===============

----------------
Download GaussPy
----------------

Download GaussPy using git (from this fork, which is currently updated to support Python 3.6+):

.. code-block:: bash

    git clone git://github.com/cmurray-astro/gausspy.git

------------
Dependencies
------------

You will need the following packages to run GaussPy:

* `Python 3.6+ <https://www.python.org/>`_

* `numpy <http://www.numpy.org/>`_

* `scipy <http://www.scipy.org/>`_

* `h5py <http://www.h5py.org/>`_

* `lmfit <https://lmfit.github.io/lmfit-py/intro.html>`_

* `tqdm <https://tqdm.github.io/>`_

---------------------
Optional Dependencies
---------------------

If you wish to use GaussPy's plotting capabilities you will need to install
`matplotlib`:

* `matplotlib <http://matplotlib.org/>`_

---------------------
Installing GaussPy
---------------------

To install make sure that all dependences are already installed and properly
linked to python --python has to be able to load them--. 

One way to achieve this goal is by using Conda. First, install either `Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_ 
(includes basic Conda root environment for Python 3.x, *recommended*) 
or `Anaconda3 <https://www.anaconda.com/distribution/>`_ 
(includes more features, tools, libraries for Python 3.x). 

You may wish to create a conda environment for the dependencies before installation. For example, to create an environment called ``gausspy":

.. code-block:: console

     $ conda env create -n gausspy --file conda-environment.yml
     $ conda activate gausspy

Then install the required dependencies:

.. code-block:: console

     $ conda install -n gausspy numpy scipy lmfit h5py tqdm
     
For now, GaussPy relies on lmfit, which can be installed using pip:

.. code-block:: console

     $ pip install lmfit
     
After verifying that the required dependencies are installed, 
return to the local directory containing GaussPy and install it via:

.. code-block:: console
    
    $ python setup.py install
    
If you would like to modify GaussPy, you may want to use links instead of
installing, which is best done by replacing the last line with:

.. code-block:: console

     $ python setup.py develop
