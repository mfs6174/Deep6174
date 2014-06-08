#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
import numpy as np
import scipy
import scipy.io as sio
import scipy.signal as signal
from scipy.misc import imsave
import theano.tensor as T
import theano
import sys, cPickle, gzip

from dataio import read_data, save_data, get_dataset_imgsize
import operator
from itertools import count
from convolutional_mlp import LeNetConvPoolLayer
from multi_convolution_mlp import ConfigurableNN

from IPython.core.debugger import Tracer

class NetworkRunner(object):

    def __init__(self, input_size):
        self.input_size = input_size
        self.nn = ConfigurableNN(1, (input_size, input_size))
        self.n_conv_layer = 0

    def add_convpool_layer(self, W, b, pool_size):
        self.n_conv_layer += 1
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
        self.LR_W = W
        assert W.shape[1] == 10
        assert b.shape[1] == 10

        self.nn.add_LR_layer()
        last_layer = self.nn.layers[-1]
        last_layer.W.set_value(W.astype('float32'))
        last_layer.b.set_value(b.flatten().astype('float32'))

    def _finish(self):
        """ compile all the functions """
        self.funcs = []
        for (idx, layer) in enumerate(self.nn.layers):
            if idx == len(self.nn.layers) - 1:
                f = theano.function([self.nn.orig_input],
                                     layer.p_y_given_x)
            else:
                f = theano.function([self.nn.orig_input],
                                   layer.output)
            self.funcs.append(f)

    def run(self, img):
        assert img.shape == (self.input_size, self.input_size)

        results = []
        for (idx, layer) in enumerate(self.nn.layers):
            if idx == len(self.nn.layers) - 1:
                results.append(self.funcs[idx]([[img]])[0])
            else:
                results.append(self.funcs[idx]([[img]]))
        return results

    def predict(self, img):
        res = self.funcs[-1]([[img]])[0]
        label = max(enumerate(res), key=operator.itemgetter(1))
        return label

    def get_LR_W(self):
        return self.LR_W


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
    runner.finish()
    return runner

def get_nn(filename, image_size):
    """ img: a (size x size) matrix
       caller should gurantee that
       img size is the same size as those used to build the network
    """
    data = sio.loadmat(filename)
    nn = build_nn_with_params(data, image_size)
    return nn

def get_an_image(dataset, label=7):
    """ get an image with label=number"""
    train_set, valid_set, test_set = read_data(dataset)
    for idx, img in enumerate(test_set[0]):
        if int(test_set[1][idx]) != label:
            continue
        size = int(np.sqrt(img.shape[0]))
        return img.reshape(size, size)

def save_LR_W(W):
    for l in range(10):
        imgs = W[:,l].reshape(20, 7, 7)
        for idx, img in enumerate(imgs):
            imsave('LRW-label{0}-weight{1}.jpg'.format(l, idx), img)

if __name__ == '__main__':
    epoch = int(sys.argv[2])
    dataset = sys.argv[1]
    size = get_dataset_imgsize(dataset)

    try:
        label = int(sys.argv[3])
    except:
        label = 3
    print "Using dataset {0} with size {1}x{1}".format(dataset, size)
    nn = get_nn('logs/{0}.mat'.format(epoch), size)

    W = nn.get_LR_W()
    save_LR_W(W)

    img = get_an_image(dataset, label)
    # run the network
    results = nn.run(img)

    # save all the representations
    #sio.savemat('logs/representations.mat', mdict={'results': results})

    # save convolved images
    for nl in xrange(nn.n_conv_layer):
        layer = results[nl][0]
        for idx, pic in enumerate(layer):
            imsave('convolved_layer{0}.{1}.jpg'.format(nl, idx), pic)

    # the predicted results
    label = max(enumerate(results[-1]), key=operator.itemgetter(1))
    print "Predicted Label(prob): ", label

# Usage ./run_network.py dataset.pkl.gz 60 [label]
