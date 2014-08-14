#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: run-and-rename.py
# Date: Thu Aug 14 12:21:58 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
from scipy.misc import imread, imsave
from itertools import izip
import sys
import os
import shutil
import os.path
import glob

if len(sys.argv) != 3:
    print "Usage: {0} <input directory with images> <model>".format(sys.argv[0])
    sys.exit(0)


from network_runner import NetworkRunner, get_nn
from imageutil import get_image_matrix
from dataio import read_data
from progress import Progressor

input_dir = sys.argv[1]
output_dir = os.path.join(input_dir, 'predicted')
shutil.rmtree(output_dir, ignore_errors=True)
os.mkdir(output_dir)

print "Reading images from {0}".format(input_dir)
print "Writing predicted results to {0}".format(output_dir)

model_file = sys.argv[2]
nn = get_nn(model_file)
print "Running network with model {0}".format(model_file)

tot, corr = 0, 0
for f in glob.glob(input_dir + '/*'):
    if not os.path.isfile(f):
        continue
    img = imread(f) / 255.0
    pred = nn.predict(img)
    label = f.split('-')[-1].split('.')[0]
    real_pred = ''.join(map(str, pred[1:1+pred[0]]))

    new_fname = "{:04d}:{}-{},{}.png".format(tot, label, pred[0],
                                         ''.join(map(str, pred[1:])))
    new_fname = os.path.join(output_dir, new_fname)
    imsave(new_fname, img)

    tot += 1
    corr += label == real_pred
    if tot > 0 and tot % 1000 == 0:
        print "Progress:", tot
print corr, tot




