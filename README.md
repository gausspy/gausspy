# Installation

### Dependencies

You will need the following packages to run GaussPy. We list the version of each
package which we know to be compatible with GaussPy.

* [python 2.7](http://www.numpy.org/)

* [numpy (v1.6.1)](http://www.numpy.org/)

* [scipy (v0.17.0)](http://www.scipy.org/)

* [lmfit (v0.9.3)](https://lmfit.github.io/lmfit-py/intro.html)

* [h5py (v2.0.1)](http://www.h5py.org/)

If you do not already have Python 2.7, you can install the [Anaconda Scientific
Python distribution](https://store.continuum.io/cshop/anaconda/), which comes
pre-loaded with numpy, scipy, and h5py.

### Optional Dependencies

If you wish to use GaussPy's plotting capabilities you will need to install
matplotlib:

* [matplotlib (v1.1.1)](http://matplotlib.org/)

If you wish to use optimization with Fortran code you will need

* [GNU Scientific Library (GSL)](http://www.gnu.org/software/gsl/)


### Download GaussPy

Download GaussPy using git `$ git clone git://github.com/gausspy/gausspy.git`


### Installing Dependencies on Linux

You will need several libraries which the `GSL`, `h5py`, and `scipy` libraries
depend on. Install these required packages with:

```bash
sudo apt-get install libblas-dev liblapack-dev gfortran libgsl0-dev libhdf5-serial-dev 
sudo apt-get install hdf5-tools
```

Install pip for easy installation of python packages:

```bash
sudo apt-get install python-pip
```

Then install the required python packages:

```bash
sudo pip install scipy numpy h5py lmfit
```

Install the optional dependencies for plotting and optimization:

```bash
sudo pip install matplotlib
sudo apt-get install libgsl0-dev
```

### Installing Dependencies on OSX

Installation on OSX can be done easily with homebrew. Install pip for easy
installation of python packages:

```bash
sudo easy_install pip
```

Then install the required python packages:

```bash
sudo pip install numpy scipy h5py lmfit
```

Install the optional dependencies for plotting and optimization:

```bash
sudo pip install matplotlib
sudo brew install gsl
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
change the 'requires' statement in setup.py to include `scipy` and `lmfit`.

### Contributing to GaussPy

To contribute to GaussPy, see [Contributing to GaussPy](CONTRIBUTING.md)
