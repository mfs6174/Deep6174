#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: network_trainer.py
from itertools import chain, izip
import operator
import pprint
import os
pprint = pprint.PrettyPrinter(indent=4).pprint

import numpy as np
import theano
import theano.tensor as T
import theano.printing as PP

from layers.layers import *
from params_logger import ParamsLogger
from learningrate import LearningRateProvider
from shared_dataio import SharedDataIO
from training_policy import TrainEarlyStopping as POLICY

N_OUT = 10
MOMENT = 0.6

OUTPUT_ENABLE_LIST=[SequenceSoftmax, LogisticRegression, SoftmaxLoss]

class NNTrainer(object):
    """ Neural Network Trainer
    """
    def __init__(self, input_image_shape, multi_output=True, patch_output=False):
        """ input_image_shape: 4D tuple
            multi_output: whether a image has more than 1 labels
        """
        self.layer_config = []
        self.layers = []
        self.batch_size = input_image_shape[0]
        self.input_shape = input_image_shape
        self.rng = np.random.RandomState()
        self.multi_output = multi_output

        self.x = T.fmatrix('x')
        Layer.x = self.x        # only for debug purpose
        if multi_output and patch_output:
            raise NotImplementedError()
        if multi_output or patch_output:
            self.y = T.imatrix('y')
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
        params['type'] = layer_class.get_class_name()

        # remove W & b in params, for better printing
        params = dict([k, v] for k, v in params.iteritems() if type(v) not in [np.ndarray, list])
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
        print "Network has {0} params in total.".format(self.n_params())
        pprint(self.layer_config)

    def finish(self):
        """ call me before training"""
        self.print_config()

        if type(self.layers[-1]) == SequenceSoftmax:
            self.max_len = self.layer_config[-1]['seq_max_len']
            print "Using Sequence Softmax Output with max_len = {0}".format(self.max_len)
        else:
            self.max_len = 0

        layer = self.layers[-1]
        assert type(layer) in OUTPUT_ENABLE_LIST
        # cost to minimize
        self.cost = layer.negative_log_likelihood(self.y)

        # all the params to optimize on
        self.params = list(chain.from_iterable([x.params for x in self.layers]))

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

    def work(self, init_learning_rate, dataset_file,
             load_all_data=True, output_directory=None):
        """ Compile, read data, and train
            dataset_file: dataset in .pkl.gz, of (train, valid, test)
            load_all_data: whether to try to load all training data into gpu,
                or only load necessary ones in every batch.
            output_directory:
                directory to save logs and outputs
        """
        if output_directory is None:
            output_directory = dataset_file + '-output'
        try:
            os.mkdir(output_directory)
        except:
            pass
        assert os.path.isdir(output_directory), "cannot create directory " + output_directory

        self.finish()
        shared_io = SharedDataIO(dataset_file, load_all_data, self)

        layer = self.layers[-1]

        def gen_updates_with_learning_rate(rate):
            # gradient descent on those params
            updates = []
            for param_i, grad_i, last_update in izip(self.params, self.grads, self.last_updates):
                upd = - rate * grad_i + MOMENT * last_update
                updates.append((param_i, param_i + upd))
                updates.append((last_update, upd))
            return updates
        lr_rate = T.fscalar()

        print "Compiling ..."
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

            def err_func_with_dataset_index(i):
                return theano.function([], layer.errors(self.y),
                        givens={
                            self.x: shared_io.shared_Xs[i],
                            self.y: shared_io.shared_ys[i]})

            do_valid_model = err_func_with_dataset_index(1)
            def validate_model(index):
                shared_io.get_valid(index)
                return do_valid_model()

            do_test_model = err_func_with_dataset_index(2)
            def test_model(index):
                shared_io.get_test(index)
                return do_test_model()
            print "Compiled."
            # read data into memory only after compilation, to save memory
            shared_io.read_delay()

        n_batches = list(shared_io.get_dataset_size())
        n_batches = [x / self.batch_size for x in n_batches]

        logger = ParamsLogger(logdir=output_directory, trainer=self)
        rate_provider = LearningRateProvider(
            os.path.join(output_directory, 'learnrate.txt'), init_learning_rate)

        # train forever and test on test set (index 2), ignore validation set.
        training = POLICY(train_model, test_model,
                                n_batches, logger, rate_provider)
        training.work()
