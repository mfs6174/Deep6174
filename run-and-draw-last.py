#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: run-and-draw-last.py
# Date: Thu Aug 14 12:16:56 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import numpy as np
import sys
import glob

if len(sys.argv) < 3:
    print "Usage: {0} <model> <input images>".format(sys.argv[0])
    sys.exit()

from network_runner import NetworkRunner, get_nn


fig = plt.figure(figsize = (19, 2))
def draw(vec):
    plt.plot(range(len(vec)), vec)
    plt.show()

model_file = sys.argv[1]
nn = get_nn(model_file)
print "Running network with model {0}".format(model_file)

def gen_file_list():
    for k in range(2, len(sys.argv)):
        pattern = sys.argv[k]
        for f in glob.glob(pattern):
            yield f

for f in gen_file_list():
    img = imread(f) / 255.0
    results = nn.run(img)
    pred = nn.predict(img)
    print "Predict: ", pred

    #print [results[-1][k].shape for k in range(len(results[-1]))]

    hidden_vec = results[-2].reshape((results[-2].shape[1],))
    draw(hidden_vec)

