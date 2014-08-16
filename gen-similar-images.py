#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gen-similar-images.py
# Date: Sat Aug 16 10:59:05 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from scipy.misc import imsave, imresize
from itertools import izip
import numpy as np
import os

from dataio import read_data
from imageutil import get_image_matrix, show_img_sync, get_label_from_dataset

dt = read_data('./data/mnist.pkl.gz')[0]

NINE = get_label_from_dataset(dt, 9)
THREE = get_label_from_dataset(dt, 3)

def paste(small, large, offset):
    assert type(offset) == tuple
    end = small.shape + offset
    assert end[0] < large.shape[0] and end[1] < large.shape[1]
    large[offset[0]: offset[0] + small.shape[0], offset[1]: offset[1] + small.shape[1]] = small
    return large


FRAME = np.zeros((50, 100), dtype='float32')

out_dir = 'data/images'
try:
    os.mkdir(out_dir)
except:
    pass

def gen_shift_images(shift1, shift2):
    nine = imresize(NINE, 0.7)
    three = imresize(THREE, 1.2)
    frame = paste(NINE, FRAME, shift1)
    frame = paste(THREE, frame, shift2)

    fname = '({0},{1})({2},{3}).png'.format(shift1[0], shift1[1], shift2[0], shift2[1])
    imsave(os.path.join(out_dir, fname), frame)

shift1 = (3, 3)
for x in range(30, 60):
    shift2 = (4, x)
    gen_shift_images(shift1, shift2)
