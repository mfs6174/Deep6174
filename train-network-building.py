#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: train-network-building.py
# Date: Fri Sep 19 15:42:17 2014 -0700
# Author: Xi-Jin Zhang <zhangxijin91@gmail.com>

import sys
import operator
import numpy as np

from network_trainer import NNTrainer
from layers.layers import *
#from dataio import read_data
from theano import tensor as T

if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        output_directory = sys.argv[2] if len(sys.argv) == 3 else None
    else:
        print "Usage: {0} [input directory] [output directory]".format(sys.argv[0])
        sys.exit(0)
    print "Dataset: ", dataset
    #ds = read_data(dataset)[0]
    img_size = (3, 80, 80)
    multi_output = False
    print "Input img size is {0}, multioutput={1}".format(img_size, multi_output)

    if len(img_size) == 1:
        assert int(np.sqrt(img_size[0])) == np.sqrt(img_size[0])
        s = int(np.sqrt(img_size[0]))
        img_size = (s, s)
    #load_all = reduce(operator.mul, img_size) < 100 ** 2
    load_all = False
    print "Load All Data: ", load_all

    # config the nn
    batch = 101
    if len(img_size) == 3:
        shape = (batch, ) + img_size
    else:
        assert len(img_size) == 2
        shape = (batch, 1) + img_size
    nn = NNTrainer(shape, multi_output=multi_output,patch_output=True,stride=20)

    nn.add_layer(ConvLayer, {'filter_shape': (64, 9, 9)})
    nn.add_layer(MaxoutLayer, {'maxout_unit': 4})
    nn.add_layer(PoolLayer, {'pool_size': 2,'stride': 1})
    #nn.add_layer(MeanSubtractLayer, {'filter_size': 3})
    #nn.add_layer(DropoutLayer, {})

    nn.add_layer(ConvLayer, {'filter_shape': (128, 7, 7)})
    nn.add_layer(MaxoutLayer, {'maxout_unit': 4})
    nn.add_layer(PoolLayer, {'pool_size': 2})
    #nn.add_layer(MeanSubtractLayer, {'filter_size': 3})
    #nn.add_layer(DropoutLayer, {})

    nn.add_layer(ConvLayer, {'filter_shape': (128, 5, 5)})
    nn.add_layer(MaxoutLayer, {'maxout_unit': 4})
    nn.add_layer(PoolLayer, {'pool_size': 2})
    #nn.add_layer(MeanSubtractLayer, {'filter_size': 3})
    nn.add_layer(DropoutLayer, {})

    #nn.add_layer(ConvLayer, {'filter_shape': (160, 5, 5)})
    #nn.add_layer(PoolLayer, {'pool_size': 2, 'stride': 1})
    #nn.add_layer(MeanSubtractLayer, {'filter_size': 3})
    #nn.add_layer(DropoutLayer, {})

    nn.add_layer(FullyConnectedLayer, {'n_out': 1600})
    nn.add_layer(DropoutLayer, {})
    nn.add_layer(FullyConnectedLayer, {'n_out': 1200})
    
    #nn.add_layer(SequenceSoftmax, {'seq_max_len': 5, 'n_out': 10})
    #nn.add_layer(LogisticRegression, {'n_out': 10})
    nn.add_layer(SoftmaxLoss, {'s_out': 20, 'n_out': 3})
    nn.work(0.05, dataset, load_all, output_directory)

# Usage: ./train-network.py dataset.pkl.gz
