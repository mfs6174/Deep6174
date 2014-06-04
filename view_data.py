#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: view_data.py
# Date: Wed Jun 04 20:30:43 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cPickle, gzip, numpy
from numpy import random
import scipy
from scipy.ndimage import zoom
from scipy.misc import toimage
from dataio import read_data
#toimage(data).show()

import sys

# Load the dataset
input = sys.argv[1]

train_set, valid_set, test_set = read_data(input)
print len(train_set[0]), len(valid_set[0]), len(test_set[0])

print train_set[0][0].shape
toimage(test_set[0][0].reshape(28, 28)).show()
