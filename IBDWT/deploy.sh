#!/bin/sh
make
./fftw3_gmp_test | tee fftw3_gmp_log.txt
