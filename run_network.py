#!/usr/bin/env python2
# -*- coding: UTF-8 -*-

import numpy as np
import scipy
import scipy.io as sio
import scipy.signal as signal
import theano.tensor as T
import theano
from itertools import count
from convolutional_mlp import LeNetConvPoolLayer
from multi_convolution_mlp import ConfigurableNN

class NetworkRunner(object):

    def __init__(self, input_size):
        self.layers = []
        self.layer_config = []
        self.input_size = input_size
        self.nn = ConfigurableNN(1, (input_size, input_size))

    def add_convpool_layer(self, W, b, pool_size):
        if len(self.nn.layers) == 0:
            # this is first layer
            ninput = 1
        else:
            ninput = self.nn.layer_config[-1]['filter_shape'][0]
        shape = W.shape
        nfilter = shape[0]
        assert ninput == shape[1]
        filter_size = shape[2]
        assert shape[2] == shape[3]
        assert b.shape[1] == nfilter

        self.nn.add_convpoollayer((nfilter, filter_size), pool_size)
        last_layer = self.nn.layers[-1]
        last_layer.W.set_value(W.astype('float32'))
        last_layer.b.set_value(b.flatten().astype('float32'))

    def add_hidden_layer(self, W, b):
        n_out = W.shape[1]
        assert b.shape[1] == W.shape[1]

        self.nn.add_hidden_layer(n_out, activation=T.tanh)
        last_layer = self.nn.layers[-1]
        last_layer.W.set_value(W.astype('float32'))
        last_layer.b.set_value(b.flatten().astype('float32'))

    def add_LR_layer(self, W, b):
        assert W.shape[1] == 10
        assert b.shape[1] == 10

        self.nn.add_LR_layer()
        last_layer = self.nn.layers[-1]
        last_layer.W.set_value(W.astype('float32'))
        last_layer.b.set_value(b.flatten().astype('float32'))

    def run(self, img):
        assert img.shape == (self.input_size, self.input_size)
        for (idx, layer) in enumerate(self.nn.layers):
            if idx == len(self.nn.layers) - 1:
                f = theano.function([self.nn.orig_input],
                                     layer.y_pred)
            else:
                f = theano.function([self.nn.orig_input],
                                   layer.output)
            print f([[img]])



fname = 'logs/10.mat'
mat = sio.loadmat(fname)

runner = NetworkRunner(28)
for nlayer in count(start=1, step=1):
    layername = 'layer' + str(nlayer)
    if layername not in mat:
        break
    layerdata = mat[layername]
    layertype = layerdata['type'][0][0][0]
    print "Layer ", nlayer, ' is ', layertype

    if layertype == 'convpool':
        runner.add_convpool_layer(layerdata['W'][0][0],
                                  layerdata['b'][0][0],
                                  layerdata['pool_size'][0][0][0][0])
    elif layertype == 'hidden':
        runner.add_hidden_layer(layerdata['W'][0][0],
                                layerdata['b'][0][0])
    elif layertype == 'lr':
        runner.add_LR_layer(layerdata['W'][0][0],
                            layerdata['b'][0][0])


data = sio.loadmat('./pic5r.mat')
data = data['pic5r']
print data
#data = np.random.rand(28, 28)
runner.run(data)
