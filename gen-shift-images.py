#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gen-shift-images.py
# Date: Fri Aug 22 14:43:05 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from scipy.misc import imsave, imresize
from itertools import izip
from copy import copy
import numpy as np
import os

from dataio import read_data
from imageutil import get_image_matrix, show_img_sync, get_label_from_dataset

dt = read_data('./data/mnist.pkl.gz')[0]

NINE = get_label_from_dataset(dt, 9)
THREE = get_label_from_dataset(dt, 3)

def paste(small, large, offset):
    large = copy(large)
    assert type(offset) == tuple
    end = small.shape + offset
    assert end[0] < large.shape[0] and end[1] < large.shape[1]
    large[offset[0]: offset[0] + small.shape[0], offset[1]: offset[1] + small.shape[1]] = small
    return large


FRAME = np.zeros((50, 100), dtype='float32')

out_dir = 'data/images-933'
try:
    os.mkdir(out_dir)
except:
    pass

def gen_shift_images(shift1, shift2):
    nine = imresize(NINE, 0.7)
    three = imresize(THREE, 1.2)
    frame = paste(nine, FRAME, shift1)
    frame = paste(three, frame, shift2)

    fname = '({0},{1})({2},{3}).png'.format(shift1[0], shift1[1], shift2[0], shift2[1])
    imsave(os.path.join(out_dir, fname), frame)

def gen_resize_images(size):
    nine = imresize(NINE, 1.0)
    three = imresize(THREE, size)
    frame = paste(nine, FRAME, (3, 3))
    frame = paste(three, frame, (3, 40))

    fname = 'resize3-{0}.png'.format(size)
    imsave(os.path.join(out_dir, fname), frame)

def gen_two_three():
    frame = paste(NINE, FRAME, (3, 3))
    frame = paste(THREE, frame, (3, 40))

    imsave(os.path.join(out_dir, '93.png'), frame)

    frame = paste(THREE, frame, (3, 70))
    imsave(os.path.join(out_dir, '933.png'), frame)

    frame = paste(THREE, FRAME, (3, 70))
    imsave(os.path.join(out_dir, '3.png'), frame)

gen_two_three()
#shift1 = (3, 3)
#for x in range(6, 15):
    #gen_resize_images(x / 10.0)
