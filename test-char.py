#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test-char.py
# Date: Thu Dec 04 20:43:50 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import sys
from itertools import izip
import cv2

from network_runner import get_nn
from dataio import read_data, save_data
from lib.imageutil import get_image_matrix
from lib.accuracy import AccuracyRecorder

params_file = sys.argv[1]
chars_file = sys.argv[2]
chars = map(lambda x: x.strip(), open(chars_file).readlines())
nn = get_nn(params_file, 1)

def padding(img, shape, fill):
    h, w = img.shape[:2]
    assert w <= shape[0] and h <= shape[1]
    pad_width = shape[0] - w
    pad_height = shape[1] - h

    pad_w0 = pad_width / 2
    pad_w1 = shape[0] - (pad_width - pad_w0)
    pad_h0 = pad_height / 2
    pad_h1 = shape[1] - (pad_height - pad_h0)

    ret = np.ones((shape[1], shape[0]), dtype='uint8') * fill
    ret[pad_h0:pad_h1,pad_w0:pad_w1] = img
    return ret

def resize(img, shape, fill_empty=0):
    h, w = img.shape[:2]
    tw, th = shape[:2]

    w_ratio = tw / float(w)
    h_ratio = th / float(h)

    ratio = min(w_ratio, h_ratio)

    cw = int(max(1, w * ratio))
    ch = int(max(1, h * ratio))
    img = cv2.resize(img, (cw, ch))
    return padding(img, shape, fill_empty)

img = cv2.imread('/tmp/a.jpg', cv2.IMREAD_GRAYSCALE)
img = resize(img, np.array([48, 48]), 255)
img = np.asarray(img, 'float32') / 255.0
pred = nn.predict(img)


print chars[pred]

