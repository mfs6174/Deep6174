#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gen_noise_data.py
# Date: Sun May 18 17:40:34 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cPickle, gzip, numpy
from numpy import random
import sys

# prepare params
input = sys.argv[1]
NOISE_MAX = float(sys.argv[2])
output = input[:-6] + "noise{0}.pkl.gz".format(NOISE_MAX)

# Load the dataset
f = gzip.open(input, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
print len(train_set[0]), len(valid_set[0]), len(test_set[0])
f.close()


def add_noise(dataset):
    Xs = dataset[0]
    for k in Xs:
        k += (random.rand(*k.shape) - 0.5) * NOISE_MAX
        k = k.clip(0, 1)


add_noise(train_set)
add_noise(valid_set)
add_noise(test_set)
print "Writing..."
data = (train_set, valid_set, test_set)
fout = gzip.open(output, 'wb')
cPickle.dump(data, fout, -1)
fout.close()

# Usage: ./gen_noise_data.py input.pkl.gz 1.2
# will generate input.noise1.pkl.gz,
# with white noise ranged from [-0.6, 0.6] added upon input.pkl.gz
