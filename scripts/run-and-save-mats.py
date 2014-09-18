#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: run-and-save-mats.py
# Date: Thu Sep 18 15:44:07 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from scipy.misc import imread
import numpy as np
import os
import sys
import glob
import os.path
import scipy.io as sio

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
from network_runner import get_nn
from layers.layers import name_dict

if len(sys.argv) < 3:
    print "Usage: {0} <model> <input images>".format(sys.argv[0])
    sys.exit()

def gen_file_list():
    """ generate image filenames from arguments given in the command line"""
    for k in range(2, len(sys.argv)):
        pattern = sys.argv[k]
        for f in glob.glob(pattern):
            if os.path.isfile(f):
                yield f

def process(img, network, out_dir):
    """ run the network, parse and save results
        out_dir: directory for output
    """
    try:
        os.mkdir(out_dir)
    except:
        pass
    assert os.path.isdir(out_dir), "cannot create directory {0}".format(out_dir)

    results = network.run(img)
    # write the prediction to txt file
    pred = network.predict(img)
    pred = "{0}-{1}".format(pred[0], ''.join(map(str, pred[1:])))
    with open(os.path.join(out_dir, 'predict.txt'), 'w') as f:
        print >> f, pred

    for idx, layer in enumerate(network.nn.layers):
        layertype = name_dict[type(layer)]
        # save the params of this layer to .mat file
        layer.save_params_mat(os.path.join(out_dir, '{}-{}-prms'.format(idx,
                                                                        layertype)))
        if layertype == 'convpool':
        # taking [idx][0] because we only have 1 image in each batch, therefore the output
        # only contain 1 set of convolved images
            sio.savemat(os.path.join(out_dir,
                                     'after{}-convolved.mat'.format(idx)),
                        {'conv{0}'.format(idx): results[idx][0]})
        elif layertype == 'hidden':
            sio.savemat(os.path.join(out_dir,
                                    'after{0}-vec.mat'.format(idx)),
                        {'vec{0}'.format(idx): results[idx][0]})
        elif layertype == 'ssm':        # sequence softmax
            # save the probabilities in each classifier
            sio.savemat(os.path.join(out_dir,
                                    'predicted-probs.mat'),
                        {'prob': results[idx]})


if __name__ == '__main__':
# We have already saved the learned parameters in sys.argv[1]
# build nn with params
    model_file = sys.argv[1]
# get a network from the saved file
    nn = get_nn(model_file)
    print "Running network with model {0}".format(model_file)

    for idx, f in enumerate(gen_file_list()):
        print "Running image #{}: {} ...".format(idx, f)
        # network accepts images ranging from [0, 1]
        img = imread(f) / 255.0
        # save in a directory with the same name of the image
        process(img, nn, f[:-4])
