#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: resize_data.py\2
# Date: Wed Jun 04 20:28:23 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cPickle, gzip, numpy
from numpy import random
import scipy
from scipy.ndimage import zoom
import sys
from dataio import save_data, save_data

# prepare params
input = sys.argv[1]
SIZE = float(sys.argv[2])
output_basename = input[:-6] + "resample{0}".format(SIZE)

train_set, valid_set, test_set = save_data(input)

def resample(dataset):
    Xs = dataset[0]
    newX = []
    for (idx, k) in enumerate(Xs):
        k = k.reshape((28, 28))
        k = zoom(k, SIZE, order=1)
        newX.append(numpy.ndarray.flatten(k))
    return (numpy.asarray(newX), dataset[1])

train_set = resample(train_set)
valid_set = resample(valid_set)
test_set = resample(test_set)

print "Writing..."
data = (train_set, valid_set, test_set)

save_data(data, output_basename)

# Usage: ./resize_data.py input.pkl.gz 2
# will generate 'input.resample2.pkl.gz',
# which is resampled on input.pkl.gz with a factor of 2 (28x28 -> 56x56)
