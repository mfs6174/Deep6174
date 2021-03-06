#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: run-and-draw-last.py
# Date: Thu Sep 18 15:43:47 2014 -0700

import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
import numpy as np
import os, sys
import glob
from copy import copy
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
from lib.imageutil import stack_vectors
from network_runner import get_nn

if len(sys.argv) < 3:
    print "Usage: {0} <model> <input images>".format(sys.argv[0])
    sys.exit()

def draw(vec, ofname):
    """ draw a vector in dots or lines, also save the vector
        ofname: output filename
    """
    fig = plt.figure(figsize = (38, 2))
    plt.plot(range(len(vec)), vec,'bo')
    fig.savefig(ofname)

    # also save the vector
    fname = ofname[:-3] + 'txt'
    with open(fname, 'w') as f:
        f.write(repr(vec))
    fig = plt.figure()

def gen_file_list():
    """ generate image filenames from arguments given in the command line"""
    for k in range(2, len(sys.argv)):
        pattern = sys.argv[k]
        for f in glob.glob(pattern):
            if os.path.isfile(f):
                yield f

# We have already saved the learned parameters in sys.argv[1]
# build nn with params
model_file = sys.argv[1]
# get a network from the saved file
nn = get_nn(model_file)
print "Running network with model {0}".format(model_file)

# get the weight of the digit '3' at the second position
prms = nn.nn.layers[-1].get_params()['Ws'][2][:,3]
# save the weight in all_vecs, to draw together with another vector later
all_vecs = [prms]
draw(prms, './weight-secondposition-3.png')

for idx, f in enumerate(gen_file_list()):
    print "Running {0}...".format(f)
    # network accepts images ranging from [0, 1]
    img = imread(f) / 255.0
    # run the network against the image
    results = nn.run(img)
    pred = nn.predict(img)
    print "Predict: ", pred

    #print [results[-1][k].shape for k in range(len(results[-1]))]
    outdir = os.path.dirname(f) + '/vec'
    try:
        os.mkdir(outdir)
    except:
        pass

    # get the representation after the last hidden layer, which is [-2]
    # layer[-1] is the output layer.
    hidden_vec = results[-2].reshape((results[-2].shape[1],))

    # build filename for output
    pred = str(pred[0]) + '-' + ''.join(map(str, pred[1:]))
    basename = os.path.basename(f)[:-4]
    fname = os.path.join(outdir, basename + '-{0}-vec.jpg'.format(pred))

    draw(hidden_vec, fname)

    # plot color-graph of weight vector and representation
    vecs = copy(all_vecs)
    vecs.append(hidden_vec)
    img = stack_vectors(vecs)
    plt.imshow(img)
    plt.savefig(os.path.join(outdir, basename + '-{0}-color.jpg'.format(pred)))
print "Results written to {0}.".format(outdir)
