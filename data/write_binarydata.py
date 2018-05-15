#!/usr/bin/env python0
# -*- coding: utf-8 -*-

import numpy as np
import sys
import scipy.misc
from array import array
import matplotlib.pyplot as plt

def binary_write(arr, output_filename, fmt='f'):
    output_file = open(output_filename, 'wb')
    float_array = array(fmt, arr.ravel())
    float_array.tofile(output_file)
    output_file.close()

Omega = np.loadtxt('mask.txt',delimiter=',')
y = np.loadtxt('y_sparse.txt',delimiter=',')

binary_write(Omega, "mask.dat", fmt="f")
binary_write(y, "y_sparse.dat", fmt="f")
