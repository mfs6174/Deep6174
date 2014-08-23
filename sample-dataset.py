#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: sample-dataset.py
# Date: Sat Aug 23 13:57:24 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from dataio import read_data
from imageutil import get_image_matrix

import sys
from itertools import izip
import os
import numpy as np
from scipy.misc import imsave

if len(sys.argv) < 2:
    print "Usage: {0} data.pkl.gz ".format(sys.argv[0]) + \
        " [number of images(default to 40)]"
    sys.exit(1)

input_data = sys.argv[1]
assert os.path.isfile(input_data)
output_dir = input_data + '-samples'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# number of images to sample
if len(sys.argv) == 3:
    n_img = int(sys.argv[2])
else:
    n_img = 40
print "Number of samples: {0}".format(n_img)

# randomly sample some imgs/labels from test set
data = read_data(input_data)[2]
index = np.random.choice(len(data[0]), n_img)
imgs = [data[0][k] for k in index]
labels = [data[1][k] for k in index]

for idx, img, label in izip(xrange(n_img), imgs, labels):
    # check whether label is a sequence or a digit
    if hasattr(label, '__iter__'):
        label = "".join(map(str, label))
    else:
        label = str(label)
    fname = "{0:02d}-{1}.png".format(idx, label)
    fname = os.path.join(output_dir, fname)
    img = get_image_matrix(img)
    imsave(fname, img)
    print "Saving to {0}".format(fname)
