#!/usr/bin/env python2
# -*- coding: UTF-8 -*-

import numpy as np
import scipy
import scipy.io as sio
from scipy.misc import imsave, toimage, imread
import theano.tensor as T
import theano
from utils import tile_raster_images, get_image_matrix

import sys, gzip
import cPickle as pickle
from IPython.core.debugger import Tracer
import operator
import itertools
from itertools import count, izip
import time

from convolutional_mlp import LeNetConvPoolLayer
from multi_convolution_mlp import ConfigurableNN
from dataio import read_data, save_data, get_dataset_imgsize

class NetworkRunner(object):

    def __init__(self, input_size):
        self.input_size = input_size
        self.nn = ConfigurableNN(1, input_size)
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
        assert b.shape[0] == nfilter

        self.nn.add_convpoollayer((nfilter, filter_size), pool_size)
        last_layer = self.nn.layers[-1]
        last_layer.W.set_value(W.astype('float32'))
        last_layer.b.set_value(b.flatten().astype('float32'))

    def add_hidden_layer(self, W, b):
        n_out = W.shape[1]
        assert b.shape[0] == W.shape[1]

        self.nn.add_hidden_layer(n_out, activation=T.tanh)
        last_layer = self.nn.layers[-1]
        last_layer.W.set_value(W.astype('float32'))
        last_layer.b.set_value(b.flatten().astype('float32'))

    def add_LR_layer(self, W, b):
        self.LR_W = W
        assert W.shape[1] == 10
        assert b.shape[0] == 10

        self.nn.add_LR_layer()
        last_layer = self.nn.layers[-1]
        last_layer.W.set_value(W.astype('float32'))
        last_layer.b.set_value(b.flatten().astype('float32'))

    def add_FLSM_layer(self, Ws, bs):
        """ add a FixedLengthSoftmax layer, with list of W and b"""
        num_out = len(Ws)
        for W, b in itertools.izip(Ws, bs):
            assert W.shape[1] == 10
            assert b.shape[0] == 10

        self.nn.add_nLR_layer(num_out)
        last_layer = self.nn.layers[-1]
        last_layer.set_params(Ws, bs)

    def add_sequence_softmax(self, Ws, bs):
        n_softmax = len(Ws)
        assert len(bs) == n_softmax

        print [k.shape for k in Ws]
        print [k.shape for k in bs]

        max_seq_len = n_softmax - 1
        assert Ws[0].shape[1] == max_seq_len
        assert bs[0].shape[0] == max_seq_len
        for W, b in itertools.izip(Ws[1:], bs[1:]):
            assert W.shape[1] == 10
            assert b.shape[0] == 10
        self.nn.add_sequence_softmax(max_seq_len)
        last_layer = self.nn.layers[-1]
        last_layer.set_params(Ws, bs)

    def finish(self):
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
        assert img.shape == self.input_size

        results = []
        for (idx, layer) in enumerate(self.nn.layers):
            results.append(self.funcs[idx]([[img]]))
        return results

    def run_only_last(self, img):
        assert img.shape == self.input_size
        return self.funcs[-1]([[img]])

    def predict(self, img):
        res = self.funcs[-1]([[img]])[0]
        label = max(enumerate(res), key=operator.itemgetter(1))
        return label

    def get_LR_W(self):
        return self.LR_W

def build_nn_with_params(params):
    """ params: the object load from param{epoch}.mat file
    """
    input_size = params['input_shape']
    print "Size={0}".format(input_size)
    runner = NetworkRunner(input_size)
    for nlayer in count(start=1, step=1):
        layername = 'layer' + str(nlayer)
        if layername not in params:
            break
        layerdata = params[layername]
        layertype = layerdata['type']
        print "Layer ", nlayer, ' is ', layertype

        for k, v in layerdata.iteritems():
            if type(v) == list:
                layerdata[k] = np.asarray(v)

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
    runner.finish()
    return runner

def get_nn(filename):
    """ img: a (size x size) matrix
       caller should gurantee that
       img size is the same size as those used to build the network
    """
    if filename.endswith('.mat'):
        data = sio.loadmat(filename)
    else:
        with gzip.open(filename, 'r') as f:
            data = pickle.load(f)

    nn = build_nn_with_params(data)
    return nn

def get_an_image(dataset, label):
    """ get an image with label=number"""
    train_set, valid_set, test_set = read_data(dataset)
    for idx, img in enumerate(test_set[0]):
        ys = test_set[1][idx]
        if hasattr(ys, '__iter__'):
            ys = ys[0]
        if int(ys) != label:
            continue
        img = get_image_matrix(img)

def save_LR_W_img(W, n_filter):
    """ save W as images """
    for l in range(10):
        w = W[:,l]
        size = int(np.sqrt(w.shape[0] / n_filter))
        imgs = w.reshape(n_filter, size, size)
        for idx, img in enumerate(imgs):
            imsave('LRW-label{0}-weight{1}.jpg'.format(l, idx), img)

def get_label_from_result(img, results):
    if img.shape[0] == img.shape[1]:
        # the predicted results for single digit output
        label = max(enumerate(results[-1][0]), key=operator.itemgetter(1))
        return label[0]
    else:
        # predicted results for multiple digit output
        ret = []
        for r in results[-1]:
            label = max(enumerate(r[0]), key=operator.itemgetter(1))
            ret.append(label[0])
        return ret

def save_convolved_images(nn, results):
    for nl in xrange(nn.n_conv_layer):
        layer = results[nl][0]
        img_shape = layer[0].shape
        tile_len = int(np.ceil(np.sqrt(len(layer))))
        tile_shape = (tile_len, int(np.ceil(len(layer) * 1.0 / tile_len)))
        layer = layer.reshape((layer.shape[0], -1))
        raster = tile_raster_images(layer, img_shape, tile_shape,
                                    tile_spacing=(3, 3))
        imsave('{0}.jpg'.format(nl), raster)

def label_match(l1, l2):
    """ whether two sequences of labels match"""
    l1, l2 = list(l1), list(l2)
    if len(l1) > len(l2):
        l1, l2 = l2, l1
    now = -1
    for k in l1:
        try:
            now = l2.index(k, now + 1)
        except:
            return False
    return True

if __name__ == '__main__':
    params_file = sys.argv[1]
    nn = get_nn(params_file)

    # save W matrix in LR layer
    #W = nn.get_LR_W()
    #save_LR_W_img(W, 20)        # 20 is the number of filters in the last convpool layer
    #sio.savemat('W.mat', mdict={'W': W})

    train, valid, test = read_data(sys.argv[2])
    corr, tot = 0, 0
    for img, label in izip(test[0], test[1]):
        img = get_image_matrix(img)
        # run the network
        results = [nn.run_only_last(img)]
        pred = get_label_from_result(img, results)
        print pred

        if hasattr(label, '__iter__'):
            if len(label) == len(pred):
                tot += len(label)
                corr += len([k for k, _ in izip(pred, label) if k == _])
            else:
                tot += 1
                corr += label_match(pred, label)
        else:
            tot += 1
            corr += label == pred

        time.sleep(10)

        if tot % 1000 == 0:
            print "Rate: {0}".format(corr * 1.0 / tot)


# Usage ./run_network.py dataset.pkl.gz param_file.pkl.gz [label]
