#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gen-noise-data.py
# Date: Thu Sep 18 15:44:27 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cPickle, gzip, numpy
from numpy import random
import sys

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
from dataio import read_data, save_data

# prepare params
input = sys.argv[1]
NOISE_MAX = float(sys.argv[2])
output_basename = input[:-6] + "noise{0}".format(NOISE_MAX)

# Load the dataset
train_set, valid_set, test_set = read_data(input)
print len(train_set[0]), len(valid_set[0]), len(test_set[0])


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
save_data(data, output_basename)

# Usage: ./gen_noise_data.py input.pkl.gz 1.2
# will generate input.noise1.pkl.gz,
# with white noise ranged from [-0.6, 0.6] added upon input.pkl.gz
