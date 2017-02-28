from distutils.core import setup, Extension
import numpy.distutils.misc_util
import numpy
import subprocess
import sys

if sys.argv[1] == 'install': subprocess.call('lib/gsl-install.sh')

tv_module = Extension ('tv', 
                 library_dirs = ['lib/lib'],
                 include_dirs = [numpy.get_include(),
                                 numpy.distutils.misc_util.get_numpy_include_dirs(), 'lib/include'],
                 libraries = ['gsl', 'gslcblas'],
                 requires = ['numpy'],
                 sources = ['src/tvmodule.c', 'src/tvmethods.c'])

setup (name = 'gausspy',
       version = '1.0',
       author='Robert Lindner',
       author_email='robertrlindner@gmail.com',
       description = 'Autonomous Gaussian Decomposition',
       ext_modules = [tv_module],
       packages = ['gausspy'],
       data_files = [('src',['src/tv.h']),
                     ('src',['lib/gsl-install.sh']),
                     ('src',['lib/gsl-1.16.tar.gz']) ]   )
