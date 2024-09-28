.. _install:

===============
Installation
===============

----------------
Download GaussPy
----------------

Download GaussPy using git (from this fork, which is currently updated to support Python <= 3.9):

.. code-block:: bash

    git clone git://github.com/gausspy/gausspy.git

---------------------
Installing GaussPy
---------------------

Create a conda environment for the dependencies before installation. For example, to create an environment called gausspy:

.. code-block:: console

     $ conda env create -n gausspy --file conda-environment.yml
     $ conda activate gausspy

     
Return to the local directory containing GaussPy and install it via:

.. code-block:: console
    
    $ python setup.py install
    
If you would like to modify GaussPy, you may want to use links instead of
installing, which is best done by replacing the last line with:

.. code-block:: console

     $ python setup.py develop
