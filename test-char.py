#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test-char.py
# Date: Thu Dec 11 00:19:50 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import sys
from itertools import izip
import cv2
if len(sys.argv) != 4:
    sys.exit("Usage: {} <model> <charlist> <picture>".format(sys.argv[0]))

from network_runner import get_nn
from lib.imageutil import get_image_matrix, resize_preserve

params_file = sys.argv[1]
chars_file = sys.argv[2]
chars = map(lambda x: x.strip(), open(chars_file).readlines())
nn = get_nn(params_file, 1)

img = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE)
img = resize_preserve(img, np.array([48, 48]), 255)
img = np.asarray(img, 'float32') / 255.0
pred = nn.predict(img)

print chars[pred]

