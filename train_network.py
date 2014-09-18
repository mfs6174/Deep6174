#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: train_network.py
import os
import sys
from itertools import chain, izip
import cPickle
import gzip
import operator
import pprint
pprint = pprint.PrettyPrinter(indent=4).pprint

import numpy
import numpy as np
from numpy import random
import theano
import theano.tensor as T
import theano.printing as PP

from params_logger import ParamsLogger
from learningrate import LearningRateProvider
from dataio import read_data, save_data, get_dataset_imgsize

from layers.layers import *
from progress import Progressor
from shared_dataio import SharedDataIO

from training_policy import TrainForever

N_OUT = 10
MOMENT = 0.6

class NNTrainer(object):
    """ Configurable Neural Network Trainer,
        Currently support several convolution-pooling layer followed by hidden layers.
    """
    def __init__(self, input_image_shape, multi_output=True):
        """ input_image_shape: 4D tuple"""
        self.layer_config = []
        self.layers = []
        self.batch_size = input_image_shape[0]
        self.input_shape = input_image_shape
        self.rng = numpy.random.RandomState(23455)

        self.x = T.fmatrix('x')
        #self.x.tag.test_value = np.random.rand(self.batch_size,
                                               #reduce(operator.mul,
                                                      #self.input_shape)).astype('float32')
        if multi_output:
            self.y = T.imatrix('y')
            #t = np.zeros((self.batch_size, 4), dtype='int32')
            #t[:2] = np.asarray([[2, 1, 2, 3], [1, 1, 2, -1]])
            #self.y.tag.test_value = t
        else:
            self.y = T.ivector('y')

        self.orig_input = self.x.reshape(self.input_shape)
        self.last_updates = []

    def add_layer(self, layer_class, params):
        """ add a layer to the network.
            layer_class: the layer class,
            params: a dict with necessary params.
            input_shape is not needed in params, since it will be derived from the previous layer.
        """
        assert issubclass(layer_class, Layer)
        if len(self.layers) == 0:
            # first layer
            params['input_shape'] = self.input_shape
            layer = layer_class.build_layer_from_params(
                params, self.rng, self.orig_input)
        else:
            last_layer = self.layers[-1]
            params['input_shape'] = last_layer.get_output_shape()
            layer = layer_class.build_layer_from_params(
                params, self.rng,
                last_layer.get_output_train(),
                last_layer.get_output_test())
        self.layers.append(layer)
        self.layer_config.append(params)

    def n_params(self):
        """ Calculate total number of params in this model"""
        def get_layer_nparam(layer):
            prms = layer.params
            ret = sum([reduce(operator.mul, k.get_value().shape) for k in prms])
            if ret > 0:
                print "Layer {0} has {1} params".format(type(layer), ret)
            return ret
        return sum([get_layer_nparam(l) for l in self.layers])

    def print_config(self):
        print "Network has {0} params in total.".format(nn.n_params())
        print "Layers:"
        print self.layers
        pprint(self.layer_config)

    def _prepare_sharedio(self, filename, load_all_data):
        """ read dataset from filename and create theano shared variables"""
        dataset = read_data(filename)
        if type(self.layers[-1]) == SequenceSoftmax:
            max_len = self.layer_config[-1]['seq_max_len']
            print "Using Sequence Softmax Output with max_len = {0}".format(max_len)
            shared_io = SharedDataIO(dataset, self.batch_size, load_all_data, max_len)
        else:
            shared_io = SharedDataIO(dataset, self.batch_size, load_all_data)
        return shared_io

    def finish(self):
        """ call me before training"""
        self.print_config()

        print "... compiling"
        layer = self.layers[-1]
        assert type(layer) in [SequenceSoftmax, LogisticRegression]
        # cost to minimize
        self.cost = layer.negative_log_likelihood(self.y)

        # all the params to optimize on
        self.params = list(chain.from_iterable([x.params for x in self.layers]))
        #params = self.layers[-1].params     # only train last layer

        # take derivatives on those params
        self.grads = T.grad(self.cost, self.params)

        # save last updates for momentum
        if not self.last_updates:
            self.last_updates = []
            for param in self.params:
                self.last_updates.append(
                    theano.shared(
                        np.zeros(param.get_value(borrow=True).shape,
                                 dtype=theano.config.floatX)
                    ))
        assert len(self.last_updates) == len(self.params), 'last updates don\'t match params'

    def work(self, init_learning_rate=0.1, dataset_file='mnist.pkl.gz',
             load_all_data=True):
        """ read data and train"""
        self.finish()

        shared_io = self._prepare_sharedio(dataset_file, load_all_data)
        layer = self.layers[-1]

        def gen_updates_with_learning_rate(rate):
            # gradient descent on those params
            updates = []
            for param_i, grad_i, last_update in izip(self.params, self.grads, self.last_updates):
                upd = - rate * grad_i + MOMENT * last_update
                updates.append((param_i, param_i + upd))
                updates.append((last_update, upd))
            return updates

        n_batches = list(shared_io.get_dataset_size())
        n_batches = [x / self.batch_size for x in n_batches]
        lr_rate = T.fscalar()

        if load_all_data:
            # load all data into GPU
            datasets = shared_io.shared_dataset
            train_set_x, train_set_y = datasets[0]
            valid_set_x, valid_set_y = datasets[1]
            test_set_x, test_set_y = datasets[2]
            index = T.lscalar()

            # symbolic function to train and update
            train_model = theano.function([index, lr_rate], self.cost,
                  updates=gen_updates_with_learning_rate(lr_rate),
                  givens={
                    self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                    self.y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]})

            # symbolic function to validate and test model error
            validate_model = theano.function([index], layer.errors(self.y),
                    givens={
                        self.x: valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                        self.y: valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]})
            test_model = theano.function([index], layer.errors(self.y),
                     givens={
                        self.x: test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                        self.y: test_set_y[index * self.batch_size: (index + 1) * self.batch_size]})
        else:
            # only load necessary data as shared variable in each batch
            do_train_model = theano.function([lr_rate], self.cost,
                updates=gen_updates_with_learning_rate(lr_rate),
                givens={
                    self.x: shared_io.shared_Xs[0],
                    self.y: shared_io.shared_ys[0]})
            def train_model(index, learning_rate):
                # update the shared variable with data used in this batch
                shared_io.get_train(index)
                return do_train_model(learning_rate)

            do_valid_model = theano.function([], layer.errors(self.y),
                givens={
                    self.x: shared_io.shared_Xs[1],
                    self.y: shared_io.shared_ys[1]})
            def validate_model(index):
                shared_io.get_valid(index)
                return do_valid_model()

            do_test_model = theano.function([], layer.errors(self.y),
                givens={
                    self.x: shared_io.shared_Xs[2],
                    self.y: shared_io.shared_ys[2]})
            def test_model(index):
                shared_io.get_test(index)
                return do_test_model()

        print '... training'
        test_freq = n_batches[0] / 2
        logger = ParamsLogger(logdir=dataset_file + '-models', trainer=self)
        rate_provider = LearningRateProvider(dataset_file + '-learnrate', init_learning_rate)

        training = TrainForever(train_model, test_model,
                                n_batches[0], test_freq,
                                logger, rate_provider)
        training.work()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        print "Usage: {0} dataset.pkl.gz".format(sys.argv[0])
        sys.exit(0)
    print "Dataset: ", dataset
    ds = read_data(dataset)[0]
    shape = ds[0][0].shape
    multi_output = hasattr(ds[1][0], '__iter__')
    print "Input img size is {0}, multioutput={1}".format(shape, multi_output)

    if len(shape) == 1:
        assert int(np.sqrt(shape[0])) == np.sqrt(shape[0])
        s = int(np.sqrt(shape[0]))
        img_size = (s, s)
    else:
        img_size = shape
    load_all = reduce(operator.mul, img_size) < 100 ** 2
    print "Load All Data: ", load_all

    # config the nn
    batch = 64
    if len(img_size) == 3:
        shape = (batch, ) + img_size
    else:
        shape = (batch, 1) + img_size
    nn = NNTrainer(shape, multi_output=multi_output)

    nn.add_layer(ConvLayer, {'filter_shape': (48, 5, 5), 'activation': None})
    nn.add_layer(MaxoutLayer, {'maxout_unit': 3})
    nn.add_layer(PoolLayer, {'pool_size': 2})
    nn.add_layer(MeanSubtractLayer, {'filter_size': 3})
    nn.add_layer(DropoutLayer, {})

    nn.add_layer(ConvLayer, {'filter_shape': (64, 5, 5)})
    nn.add_layer(PoolLayer, {'pool_size': 2})
    nn.add_layer(MeanSubtractLayer, {'filter_size': 3})
    nn.add_layer(DropoutLayer, {})

    nn.add_layer(ConvLayer, {'filter_shape': (128, 5, 5)})
    nn.add_layer(PoolLayer, {'pool_size': 2})
    nn.add_layer(MeanSubtractLayer, {'filter_size': 3})
    nn.add_layer(DropoutLayer, {})

    nn.add_layer(ConvLayer, {'filter_shape': (160, 5, 5)})
    nn.add_layer(PoolLayer, {'pool_size': 2})
    nn.add_layer(MeanSubtractLayer, {'filter_size': 3})
    nn.add_layer(DropoutLayer, {})

    nn.add_layer(FullyConnectedLayer, {'n_out': 3072})
    nn.add_layer(DropoutLayer, {})

    if multi_output:
        nn.add_layer(SequenceSoftmax, {'seq_max_len': 5, 'n_out': 10})
        #nn.add_nLR_layer(2)
    else:
        nn.add_layer(LogisticRegression, {'n_out': 10})
    nn.work(init_learning_rate=0.04, dataset_file=dataset, load_all_data=load_all)

# Usage: ./train_network.py dataset.pkl.gz
