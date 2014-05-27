#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
import numpy as np
import scipy
import scipy.io as sio
import scipy.signal as signal
import theano.tensor as T
import theano

import operator
from itertools import count
from convolutional_mlp import LeNetConvPoolLayer
from multi_convolution_mlp import ConfigurableNN

class NetworkRunner(object):

    def __init__(self, input_size):
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

        results = []
        for (idx, layer) in enumerate(self.nn.layers):
            if idx == len(self.nn.layers) - 1:
                f = theano.function([self.nn.orig_input],
                                     layer.p_y_given_x)
                results.append(f([[img]])[0])
            else:
                f = theano.function([self.nn.orig_input],
                                   layer.output)
                results.append(f([[img]]))
        return results

def build_nn_with_params(params, input_size):
    """ params: the object load from {epoch}.mat file
        input_size: an integer. e.g. 28
    """
    runner = NetworkRunner(input_size)
    for nlayer in count(start=1, step=1):
        layername = 'layer' + str(nlayer)
        if layername not in params:
            break
        layerdata = params[layername]
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
    return runner

def get_nn(filename, image_size):
    """ img: a (size x size) matrix
       caller should gurantee that
       img size is the same size as those used to build the network
    """
    data = sio.loadmat(filename)
    nn = build_nn_with_params(data, image_size)
    return nn

def get_an_image(dataset='mnist.pkl.gz'):
    import cPickle, gzip
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return test_set[0][0].reshape(28, 28)

if __name__ == '__main__':
    nn = get_nn('logs/60.mat', 28)
    img = get_an_image()

    results = nn.run(img)
    label = max(enumerate(results[-1]), key=operator.itemgetter(1))
    print "Predicted Label(prob): ", label
