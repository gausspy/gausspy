# GaussPy
A python tool for implementing the Autonomous Gaussian Decomposition algorithm.

Robert R. Lindner, Carlos Vera-Ciro, Claire E. Murray, Elijah Bernstein-Cooper

[Lindner et al. 2015](https://arxiv.org/abs/1409.2840)

# Documentation

New! The GaussPy documentation can be found on ReadTheDocs [here](http://gausspy.readthedocs.io/en/latest/) 

# Installation

### Dependencies

You will need the following packages to run GaussPy. We list the version of each
package which we know to be compatible with GaussPy.

* [python 3.6] or later

* [numpy](http://www.numpy.org/)

* [scipy](http://www.scipy.org/)

* [lmfit](https://lmfit.github.io/lmfit-py/intro.html)

If you do not already have Python 3.6, you can install the [Anaconda Scientific
Python distribution](https://store.continuum.io/cshop/anaconda/).

### Optional Dependencies

If you wish to use GaussPy's plotting capabilities you will need to install
matplotlib:

* [matplotlib](http://matplotlib.org/)


### Download GaussPy

Download GaussPy using git `$ git clone git://github.com/gausspy/gausspy.git`


### Installing Dependencies on Linux

Install pip for easy installation of python packages:

```bash
sudo apt-get install python-pip
```

Then install the required python packages:

```bash
sudo pip install scipy numpy lmfit
```

Install the optional dependencies for plotting:

```bash
sudo pip install matplotlib
```

### Installing Dependencies on OSX

Installation on OSX can be done easily with homebrew. Install pip for easy
installation of python packages:

```bash
sudo easy_install pip
```

Then install the required python packages:

```bash
sudo pip install numpy scipy lmfit
```

Install the optional dependencies for plotting:

```bash
sudo pip install matplotlib
``` 

### Installing GaussPy

To install make sure that all dependences are already installed and properly
linked to python --python has to be able to load them--. Then cd to the local
directory containing GaussPy and install via

```bash
python setup.py install
```

If you don't have root access and/or wish a local installation of
GaussPy then use

```bash
python setup.py install --user
```

### Contributing to GaussPy

To contribute to GaussPy, see [Contributing to GaussPy](CONTRIBUTING.md)
