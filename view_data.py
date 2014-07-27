#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: view_data.py
# Date: Sat Jul 26 20:29:25 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cPickle, gzip, numpy
from numpy import random
import scipy
from scipy.ndimage import zoom
from scipy.misc import toimage
from dataio import read_data
import matplotlib.pyplot as plot
from utils import get_image_matrix
#toimage(data).show()

import sys

# Load the dataset
input = sys.argv[1]

train_set, valid_set, test_set = read_data(input)
print len(train_set[0]), len(valid_set[0]), len(test_set[0])

#print train_set[0][0].shape
#toimage(test_set[0][0].reshape(28, 28)).show()

for k in train_set[0]:
    k = get_image_matrix(k)
    # show images in blocking way
    plot.imshow(k)
    plot.show()
