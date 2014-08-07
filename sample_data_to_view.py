#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: sample_data_to_view.py
# Date: Thu Aug 07 00:42:52 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from dataio import read_data
from utils import get_image_matrix

import sys
from itertools import izip
import os
import numpy as np
from scipy.misc import imsave

if len(sys.argv) < 3:
    print "Usage: {0} data.pkl.gz <output dir>".format(sys.argv[0]) + \
        " [number of images(default to 20)]"
    sys.exit(1)

input_data = sys.argv[1]
assert os.path.isfile(input_data)

output_dir = sys.argv[2]
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

if len(sys.argv) == 4:
    n_img = int(sys.argv[3])
else:
    n_img = 20

train = read_data(input_data)[0]
index = np.random.choice(len(train[0]), n_img)
imgs = train[0][index]
labels = train[1][index]

for idx, img, label in izip(xrange(n_img), imgs, labels):
    if hasattr(label, '__iter__'):
        label = "".join(map(str, label))
    else:
        label = str(label)
    fname = "{0:02d}-{1}.png".format(idx, label)
    fname = os.path.join(output_dir, fname)
    img = get_image_matrix(img)
    imsave(fname, img)
    print "Saving to {0}".format(fname)
