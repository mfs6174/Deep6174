#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: train-network-mnist.py
# Date: Fri Sep 19 15:42:17 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Author: Xi-Jin Zhang <zhangxijin91@gmail.com>

import sys
import operator
import numpy as np

from network_trainer import NNTrainer
from layers.layers import *
from dataio import read_data
from theano import tensor as T

if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        output_directory = sys.argv[2] if len(sys.argv) == 3 else None
    else:
        print "Usage: {0} dataset.pkl.gz [output directory]".format(sys.argv[0])
        sys.exit(0)
    print "Dataset: ", dataset
    ds = read_data(dataset)[0]
    img_size = ds[0][0].shape
    multi_output = hasattr(ds[1][0], '__iter__')
    #img_size = (3, 64, 64)
    multi_output = False
    print "Input img size is {0}, multioutput={1}".format(img_size, multi_output)

    if len(img_size) == 1:
        assert int(np.sqrt(img_size[0])) == np.sqrt(img_size[0])
        s = int(np.sqrt(img_size[0]))
        img_size = (s, s)
    load_all = reduce(operator.mul, img_size) < 100 ** 2
    print "Load All Data: ", load_all

    # config the nn
    batch = 500
    if len(img_size) == 3:
        shape = (batch, ) + img_size
    else:
        assert len(img_size) == 2
        shape = (batch, 1) + img_size
    nn = NNTrainer(shape, multi_output=multi_output,patch_output=False)

    nn.add_layer(ConvLayer, {'filter_shape': (20, 5, 5), 'activation': T.tanh})
    #nn.add_layer(MaxoutLayer, {'maxout_unit': 3})
    nn.add_layer(PoolLayer, {'pool_size': 2})
    #nn.add_layer(MeanSubtractLayer, {'filter_size': 3})
    #nn.add_layer(DropoutLayer, {})

    nn.add_layer(ConvLayer, {'filter_shape': (50, 5, 5)})
    nn.add_layer(PoolLayer, {'pool_size': 2})
    #nn.add_layer(MeanSubtractLayer, {'filter_size': 3})
    #nn.add_layer(DropoutLayer, {})

    #nn.add_layer(ConvLayer, {'filter_shape': (128, 5, 5)})
    #nn.add_layer(PoolLayer, {'pool_size': 2})
    #nn.add_layer(MeanSubtractLayer, {'filter_size': 3})
    #nn.add_layer(DropoutLayer, {})

    #nn.add_layer(ConvLayer, {'filter_shape': (160, 5, 5)})
    #nn.add_layer(PoolLayer, {'pool_size': 2, 'stride': 1})
    #nn.add_layer(MeanSubtractLayer, {'filter_size': 3})
    #nn.add_layer(DropoutLayer, {})

    nn.add_layer(FullyConnectedLayer, {'n_out': 500,'activation': T.tanh})
    #nn.add_layer(DropoutLayer, {})

    #nn.add_layer(SequenceSoftmax, {'seq_max_len': 5, 'n_out': 10})
    nn.add_layer(LogisticRegression, {'n_out': 10})
    nn.work(0.1, dataset, load_all, output_directory)

# Usage: ./train-network.py dataset.pkl.gz
