#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: sample-dataset.py
# Date: Thu Sep 18 15:44:42 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import shutil
import gzip
import cPickle as pickle
import sys, os
from itertools import izip
import numpy as np
from scipy.misc import imsave

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
from dataio import read_data
from lib.imageutil import get_image_matrix

if len(sys.argv) < 2:
    print "Usage: {0} data.pkl.gz ".format(sys.argv[0]) + \
        " [number of images(default to 100)]"
    sys.exit(1)

input_data = sys.argv[1]
assert os.path.isfile(input_data)
output_dir = input_data + '-samples'
try:
    shutil.rmtree(output_dir)
except:
    pass
os.mkdir(output_dir)

# number of images to sample
if len(sys.argv) == 3:
    n_img = int(sys.argv[2])
else:
    n_img = 100
print "Number of samples: {0}".format(n_img)

# randomly sample some imgs/labels from test set
try:
    data = read_data(input_data)[2]
except:
    print "Failed to read (train, valid, test), try only one"
    f = gzip.open(input_data, 'rb')
    data = pickle.load(f)

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
