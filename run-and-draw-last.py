#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: run-and-draw-last.py
# Date: Fri Aug 22 15:05:42 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import numpy as np
import os
import sys
import glob
import os.path
from copy import copy

from imageutil import stack_vectors

if len(sys.argv) < 3:
    print "Usage: {0} <model> <input images>".format(sys.argv[0])
    sys.exit()

from network_runner import NetworkRunner, get_nn


def draw(vec, ofname):
    fig = plt.figure(figsize = (38, 2))
    plt.plot(range(len(vec)), vec,'bo')
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

prms = nn.nn.layers[-1].get_params()['Ws'][2]
#orig_vecs = [prms[:,k] for k in range(10)]
orig_vecs = [prms[:,3]]
vec = nn.nn.layers[-1].get_params()['Ws'][2][:,3]
print vec.shape
draw(vec, './weight-secondposition-3.png')

for idx, f in enumerate(gen_file_list()):
    print "Running {0}...".format(f)
    img = imread(f) / 255.0
    results = nn.run(img)
    pred = nn.predict(img)
    #print "Predict: ", pred

    #print [results[-1][k].shape for k in range(len(results[-1]))]
    outdir = os.path.dirname(f) + '/vec'
    try:
        os.mkdir(outdir)
    except:
        pass

    hidden_vec = results[-2].reshape((results[-2].shape[1],))

    pred = str(pred[0]) + '-' + ''.join(map(str, pred[1:]))
    basename = os.path.basename(f)[:-4]
    fname = os.path.join(outdir, basename + '-{0}-vec.jpg'.format(pred))
    draw(hidden_vec, fname)
    fname = os.path.join(outdir, basename + '-{0}-vec.txt'.format(pred))
    with open(fname, 'w') as f:
        f.write(repr(hidden_vec))

    vecs = copy(orig_vecs)
    vecs.append(hidden_vec)
    img = stack_vectors(vecs)
    fig = plt.figure()
    plt.imshow(img)
    plt.savefig(os.path.join(outdir, basename + '-{0}-color.jpg'.format(pred)))

