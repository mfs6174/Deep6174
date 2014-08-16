#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: run-and-draw-last.py
# Date: Sat Aug 16 11:15:04 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import numpy as np
import sys
import glob
import os.path

if len(sys.argv) < 3:
    print "Usage: {0} <model> <input images>".format(sys.argv[0])
    sys.exit()

from network_runner import NetworkRunner, get_nn


def draw(vec, ofname):
    fig = plt.figure(figsize = (19, 2))
    plt.plot(range(len(vec)), vec)
    fig.savefig(ofname)

model_file = sys.argv[1]
nn = get_nn(model_file)
print "Running network with model {0}".format(model_file)

def gen_file_list():
    for k in range(2, len(sys.argv)):
        pattern = sys.argv[k]
        for f in glob.glob(pattern):
            if os.path.isfile(f):
                yield f

vec = nn.nn.layers[-1].get_params()['Ws'][2][:,3]
print vec.shape
draw(vec, './3-matrix.png')
for f in gen_file_list():
    print "Running {0}...".format(f)
    img = imread(f) / 255.0
    results = nn.run(img)
    pred = nn.predict(img)
    print "Predict: ", pred

    #print [results[-1][k].shape for k in range(len(results[-1]))]


    hidden_vec = results[-2].reshape((results[-2].shape[1],))

    pred = str(pred[0]) + '-' + ''.join(map(str, pred[1:]))
    fname = os.path.join(os.path.dirname(f) + '/vec', os.path.basename(f)[:-4] +
                         '-{0}-vec.jpg'.format(pred))
    draw(hidden_vec, fname)

