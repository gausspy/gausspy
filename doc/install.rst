.. _install:

===============
Installation
===============

------------
Dependencies
------------

You will need the following packages to run GaussPy. We list the version of each
package which we know to be compatible with GaussPy.

* `Python 3.6+ <https://www.python.org/>`_

* `numpy <http://www.numpy.org/>`_

* `scipy <http://www.scipy.org/>`_

* `lmfit <https://lmfit.github.io/lmfit-py/intro.html>`_

If you do not already have Python 3.6, you can install the `Anaconda Scientific
Python distribution <https://store.continuum.io/cshop/anaconda/>`.

---------------------
Optional Dependencies
---------------------

If you wish to use GaussPy's plotting capabilities you will need to install
`matplotlib`:

* `matplotlib <http://matplotlib.org/>`_

----------------
Download GaussPy
----------------

Download GaussPy using git $ git clone git://github.com/gausspy/gausspy.git

--------------------------------
Installing Dependencies on Linux
--------------------------------

Install pip for easy installation of python packages:

.. code-block:: bash

    sudo apt-get install python-pip

Then install the required python packages:

.. code-block:: bash

    sudo pip install scipy numpy lmfit

Install the optional dependencies for plotting:

.. code-block:: bash

    sudo pip install matplotlib
    
------------------------------
Installing Dependencies on OSX
------------------------------

Installation on OSX can be done easily with homebrew. Install pip for easy
installation of python packages:

.. code-block:: bash

    sudo easy_install pip

Then install the required python packages:

.. code-block:: bash

    sudo pip install numpy scipy lmfit

Install the optional dependencies for plotting and optimization:

.. code-block:: bash

    sudo pip install matplotlib
    
------------------
Installing GaussPy
------------------

To install make sure that all dependences are already installed and properly
linked to python --python has to be able to load them--. Then cd to the local
directory containing GaussPy and install via

.. code-block:: bash
    
    python setup.py install

If you don't have root access and/or wish a local installation of
GaussPy then use

.. code-block:: bash
    
    python setup.py install --user

change the 'requires' statement in setup.py to include `scipy` and `lmfit`.

