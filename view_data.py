#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: view_data.py
# Date: Mon May 05 20:25:04 2014 +0000
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cPickle, gzip, numpy
from numpy import random
import scipy
from scipy.ndimage import zoom
from scipy.misc import toimage
#toimage(data).show()

import sys

# Load the dataset
input = sys.argv[1]

f = gzip.open(input, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
print len(train_set[0]), len(valid_set[0]), len(test_set[0])
f.close()

print train_set[0][0].shape
