#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: resize_data.py\2
# Date: Sun May 18 17:52:09 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cPickle, gzip, numpy
from numpy import random
import scipy
from scipy.ndimage import zoom
import sys

# prepare params
input = sys.argv[1]
SIZE = float(sys.argv[2])
output = input[:-6] + "resample{0}.pkl.gz".format(SIZE)

# Load the dataset
f = gzip.open(input, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
print len(train_set[0]), len(valid_set[0]), len(test_set[0])
f.close()


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
fout = gzip.open(output, 'wb')
cPickle.dump(data, fout, -1)
fout.close()

# Usage: ./resize_data.py input.pkl.gz 2
# will generate 'input.resample2.pkl.gz',
# which is resampled on input.pkl.gz with a factor of 2 (28x28 -> 56x56)
