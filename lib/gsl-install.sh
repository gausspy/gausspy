#!/bin/bash

cd lib
gsl_ver=1.16
cwd=$PWD

# Inflating
tar -zxf gsl-$gsl_ver.tar.gz

# Configuring
cd gsl-$gsl_ver
./configure --prefix=$cwd

# Compiling
make 

# Installing
make install 
