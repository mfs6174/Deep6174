#!/usr/bin/env python2
# -*- coding: UTF-8 -*-

import numpy as np
import scipy
import scipy.io as sio
from scipy.misc import imsave, toimage, imread
import theano.tensor as T
import theano
import sys, gzip
import cPickle as pickle
import operator
import itertools
from itertools import count, izip
import time

from train_network import NNTrainer
from imageutil import tile_raster_images, get_image_matrix

N_OUT = 10

class NetworkRunner(object):
    def __init__(self, rgb_input_size, batch_size=1):
        """ input size in (height, width)"""
        self.rgb_input_size = rgb_input_size
        self.batch_size = batch_size
        # nn is the underlying neural network object to run with
        self.nn = NNTrainer(batch_size, rgb_input_size)
        self.n_conv_layer = 0
        self.var_len_output = False
        self.multi_output = False

    def get_layer_by_index(self, idx):
        """ return the instance of certain layer.
            idx can be negative to get layers from the end
        """
        return self.nn.layers[idx]

    def add_convpool_layer(self, W, b, pool_size):
        self.n_conv_layer += 1
        if len(self.nn.layers) == 0:
            # this is the first layer
            ninput = 3 if len(self.rgb_input_size) == 3 else 1
        else:
            ninput = self.nn.layer_config[-1]['filter_shape'][0]
        # check the shapes
        shape = W.shape
        print shape
        nfilter = shape[0]
        assert ninput == shape[1], "{0}!={1}".format(ninput, shape[1])
        filter_size = shape[2]
        assert shape[2] == shape[3]
        assert b.shape[0] == nfilter

        # add a conv layer in network
        self.nn.add_convpoollayer((nfilter, filter_size), pool_size)
        last_layer = self.nn.layers[-1]
        last_layer.W.set_value(W.astype('float32'))
        last_layer.b.set_value(b.flatten().astype('float32'))

    def add_hidden_layer(self, W, b):
        assert b.shape[0] == W.shape[1]

        self.nn.add_hidden_layer(W.shape[1], activation=T.tanh)
        last_layer = self.nn.layers[-1]
        last_layer.W.set_value(W.astype('float32'))
        last_layer.b.set_value(b.flatten().astype('float32'))

    def add_LR_layer(self, W, b):
        self.LR_W = W
        # LR layer can only be used output
        assert W.shape[1] == N_OUT
        assert b.shape[0] == N_OUT

        self.nn.add_LR_layer()
        last_layer = self.nn.layers[-1]
        last_layer.W.set_value(W.astype('float32'))
        last_layer.b.set_value(b.flatten().astype('float32'))

    def add_FLSM_layer(self, Ws, bs):
        """ add a FixedLengthSoftmax layer, with a list of W and b
            Ws: list of W
            bs: list of b
        """
        self.multi_output = True
        num_out = len(Ws)
        for W, b in itertools.izip(Ws, bs):
            assert W.shape[1] == N_OUT
            assert b.shape[0] == N_OUT

        self.nn.add_nLR_layer(num_out)
        last_layer = self.nn.layers[-1]
        last_layer.set_params(Ws, bs)

    def add_sequence_softmax(self, Ws, bs):
        """ add a Sequence Softmax layer, with list of W and b
            Ws: list of W
            bs: list of b
        """
        self.var_len_output = True
        self.multi_output = True
        n_softmax = len(Ws)
        assert len(bs) == n_softmax, \
                "{0}!={1}".format(len(bs), n_softmax)

        max_seq_len = n_softmax - 1
        assert Ws[0].shape[1] == max_seq_len
        assert bs[0].shape[0] == max_seq_len
        for W, b in itertools.izip(Ws[1:], bs[1:]):
            assert W.shape[1] == N_OUT
            assert b.shape[0] == N_OUT
        self.nn.add_sequence_softmax(max_seq_len)
        last_layer = self.nn.layers[-1]
        last_layer.set_params(Ws, bs)

    def finish(self):
        """ compile the output of each layer as theano function"""
        self.funcs = []
        for (idx, layer) in enumerate(self.nn.layers):
            if idx == len(self.nn.layers) - 1:
                # the output layer: use likelihood of the label
                f = theano.function([self.nn.x],
                                     layer.p_y_given_x,
                                    allow_input_downcast=True)
            else:
                # layers in the middle: use its output fed into the next layer
                f = theano.function([self.nn.x],
                                   layer.output, allow_input_downcast=True)
            self.funcs.append(f)

    def run(self, img):
        """ return all the representations after each layer"""
        assert self.batch_size == 1, \
                "batch_size of runner is not 1, but trying to run against 1 image"
        assert img.shape == self.rgb_input_size

        img = img.flatten()
        results = []
        for (idx, layer) in enumerate(self.nn.layers):
            # why [img]?
            # theano needs arguments to be listed, although there is only 1 argument here
            results.append(self.funcs[idx]([img]))
        return results

    def run_only_last(self, img):
        """ return representation of the last layer"""
        assert img.shape == self.rgb_input_size
        img = img.flatten()
        return self.funcs[-1]([img])

    def predict(self, img):
        """ return predicted label (either a list or a digit)"""
        img = get_image_matrix(img, show=False)
        results = [self.run_only_last(img)]
        label = NetworkRunner.get_label_from_result(img, results,
                                                    self.multi_output,
                                                    self.var_len_output)
        return label

    @staticmethod
    def get_label_from_result(img, results, multi_output, var_len_output=True):
        """ parse the results and get label
            results: return value of run() or run_only_last()
        """
        if not multi_output:
            # the predicted results for single digit output
            label = max(enumerate(results[-1]), key=operator.itemgetter(1))
            return label[0]
        else:
            # predicted results for multiple digit output
            ret = []
            for r in results[-1]:
                label = max(enumerate(r[0]), key=operator.itemgetter(1))
                ret.append(label[0])
            if var_len_output:
                # the first element is 'length - 1', make it 'length'
                ret[0] += 1
            return ret


def build_nn_with_params(params, batch_size=1):
    """ build a network and return it
        params: the object load from param{epoch}.pkl.gz file
    """
    rgb_input_size = params['input_shape']
    print "Size={0}".format(rgb_input_size)
    runner = NetworkRunner(rgb_input_size, batch_size)
    for nlayer in count(start=0, step=1):
        layername = 'layer' + str(nlayer)
        if layername not in params:
        # BACKWARD COMPATIBILITY:
        # some old models I generated starts counting layers from one
            if nlayer == 0:
                continue
            else:
                break
        layerdata = params[layername]
        layertype = layerdata['type']
        print "Layer ", nlayer, ' is ', layertype

        if layertype == 'convpool':
            runner.add_convpool_layer(layerdata['W'],
                                      layerdata['b'],
                                      layerdata['pool_size'])
        elif layertype == 'hidden':
            runner.add_hidden_layer(layerdata['W'],
                                    layerdata['b'])
        elif layertype == 'lr':
            runner.add_LR_layer(layerdata['W'],
                                layerdata['b'])
        elif layertype == 'fl-sm':
            runner.add_FLSM_layer(layerdata['Ws'],
                                 layerdata['bs'])
        elif layertype == 'ssm':
            runner.add_sequence_softmax(
                layerdata['Ws'], layerdata['bs'])
    return runner

def get_nn(filename):
    """ get a network from a saved model file"""
    with gzip.open(filename, 'r') as f:
        data = pickle.load(f)

    nn = build_nn_with_params(data)
    # compile all the functions
    nn.finish()
    return nn

#def save_LR_W_img(W, n_filter):
    #""" save W as images """
    #for l in range(N_OUT):
        #w = W[:,l]
        #size = int(np.sqrt(w.shape[0] / n_filter))
        #imgs = w.reshape(n_filter, size, size)
        #for idx, img in enumerate(imgs):
            #imsave('LRW-label{0}-weight{1}.jpg'.format(l, idx), img)

#def save_convolved_images(nn, results):
    #for nl in xrange(nn.n_conv_layer):
        #layer = results[nl][0]
        #img_shape = layer[0].shape
        #tile_len = int(np.ceil(np.sqrt(len(layer))))
        #tile_shape = (tile_len, int(np.ceil(len(layer) * 1.0 / tile_len)))
        #layer = layer.reshape((layer.shape[0], -1))
        #raster = tile_raster_images(layer, img_shape, tile_shape,
                                    #tile_spacing=(3, 3))
        #imsave('{0}.jpg'.format(nl), raster)

