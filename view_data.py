#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: view_data.py
# Date: Tue Jul 22 17:17:44 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cPickle, gzip, numpy
from numpy import random
import scipy
from scipy.ndimage import zoom
from scipy.misc import toimage
from dataio import read_data
import matplotlib.pyplot as plot
#toimage(data).show()

import sys

# Load the dataset
input = sys.argv[1]

train_set, valid_set, test_set = read_data(input)
print len(train_set[0]), len(valid_set[0]), len(test_set[0])

#print train_set[0][0].shape
#toimage(test_set[0][0].reshape(28, 28)).show()

for k in train_set[0]:
    shape = k.shape
    if len(shape) == 1:
        size = int(np.sqrt(shape[0]))
        assert size * size == shape[0]
        k = k.reshape((size, size))
    # show images in blocking way
    plot.imshow(k)
    plot.show()
    print 'here'
