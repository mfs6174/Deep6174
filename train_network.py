#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: train_network.py
# Date: Tue Aug 05 12:55:12 2014 -0700
import os
import sys
import time
from itertools import chain
import cPickle
import gzip
import operator

import numpy
import numpy as np
from numpy import random

import theano
import theano.tensor as T

from params_logger import ParamsLogger
from logistic_sgd import LogisticRegression, load_data
from dataio import read_data, save_data, get_dataset_imgsize
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer
from fixed_length_softmax import FixedLengthSoftmax
from sequence_softmax import SequenceSoftmax
from progress import Progressor
from shared_dataio import SharedDataIO

N_OUT = 10

class NNTrainer(object):
    """ Configurable Neural Network Trainer,
        Currently support several convolution-pooling layer followed by hidden layers.
    """
    def __init__(self, batch_size, input_shape, multi_output=False):
        self.layer_config = []
        self.layers = []
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.rng = numpy.random.RandomState(23455)

        self.x = T.matrix('x')
        if multi_output:
            self.y = T.imatrix('y')
        else:
            self.y = T.ivector('y')

        self.orig_input = self.x.reshape((self.batch_size, 1) + input_shape)

    def add_convpoollayer(self, filter_config, pool_size):
        """ filter_config: tuple(nfilters, filter_size)
            pool_size: tuple or int
        """
        if type(pool_size) == int:
            pool_size = (pool_size, pool_size)
        if len(self.layer_config):
            assert type(self.layers[-1]) == LeNetConvPoolLayer, \
                "currently convpool layer must come after a convpool layer"


        if not len(self.layers):
            # This is the First Convolutional Layer
            image_shape = (self.batch_size, 1) + self.input_shape
            filter_shape = (filter_config[0], 1,
                            filter_config[1], filter_config[1])

            layer = LeNetConvPoolLayer(self.rng, input=self.orig_input,
                        image_shape=image_shape, filter_shape=filter_shape,
                        poolsize=pool_size)
            self.layers.append(layer)
        else:
            # this is a layer following previous convolutional layer
            # calculate new config based on config of previous layer
            last_config = self.layer_config[-1]
            last_filt_shape = last_config['filter_shape']
            last_img_shape = last_config['image_shape']
            now_img_size = [(last_img_shape[k] - last_filt_shape[k] + 1) / last_config['pool_size'][k - 2] for k in [2, 3]]
            image_shape = (self.batch_size, last_filt_shape[0]) + tuple(now_img_size)
            filter_shape = (filter_config[0], last_filt_shape[0],
                            filter_config[1], filter_config[1])

            layer = LeNetConvPoolLayer(self.rng, input=self.layers[-1].output,
                        image_shape=image_shape, filter_shape=filter_shape,
                        poolsize=pool_size)
            self.layers.append(layer)

        # save the config for next layer to use
        self.layer_config.append({'image_shape': image_shape,
                                  'filter_shape': filter_shape,
                                  'pool_size': pool_size })

    def _get_nin_after_convlayer(self):
        """ get n_in of next layer after a convpool layer"""
        assert type(self.layers[-1]) == LeNetConvPoolLayer
        last_config = self.layer_config[-1]
        # calculate the image size generated by last layer
        last_img_shape = last_config['image_shape']
        last_filt_shape = last_config['filter_shape']
        now_img_size = [(last_img_shape[k] - last_filt_shape[k] + 1) / last_config['pool_size'][k-2] for k in [2, 3]]
        last_filt_shape = last_config['filter_shape']
        return last_filt_shape[0] * now_img_size[0] * now_img_size[1]

    def add_hidden_layer(self, n_out, activation):
        if not len(self.layer_config):
        # XXX do something
            assert len(self.layer_config), "hidden layer must be added after ConvPool layer"
            return

        last_config = self.layer_config[-1]
        if type(self.layers[-1]) == LeNetConvPoolLayer:
            input = self.layers[-1].output.flatten(2)
            layer = HiddenLayer(self.rng, input=input,
                                n_in=self._get_nin_after_convlayer(),
                                n_out=n_out, activation=activation)
            self.layers.append(layer)
            self.layer_config.append({'n_out': n_out, 'activation': activation})
        else:
            assert "currently hidden layers must follow a conv layer"

    def add_LR_layer(self):
        """ Can only be used as output layer"""
        if type(self.layers[-1]) == HiddenLayer:
            layer = LogisticRegression(input=self.layers[-1].output,
                                       n_in=self.layer_config[-1]['n_out'],
                                       n_out=N_OUT)
        elif type(self.layers[-1]) == LeNetConvPoolLayer:
            input = self.layers[-1].output.flatten(2)
            layer = LogisticRegression(input=input,
                                      n_in=self._get_nin_after_convlayer(),
                                      n_out=N_OUT)

        self.layers.append(layer)
        self.layer_config.append(None)

    def add_nLR_layer(self, n):
        """ Can only be used as output layer"""
        if type(self.layers[-1]) == HiddenLayer:
            layer = FixedLengthSoftmax(input=self.layers[-1].output,
                                       n_in=self.layer_config[-1]['n_out'],
                                       n_out=N_OUT,
                                       num_out=n)
        elif type(self.layers[-1])== LeNetConvPoolLayer:
            input = self.layers[-1].output.flatten(2)
            layer = FixedLengthSoftmax(input=input,
                                      n_in=self._get_nin_after_convlayer(),
                                      n_out=N_OUT,
                                      num_out=n)
        self.layers.append(layer)
        self.layer_config.append(None)

    def add_sequence_softmax(self, max_len):
        """ Can only be used as output layer"""
        assert type(self.layers[-1]) == HiddenLayer
        layer = SequenceSoftmax(input=self.layers[-1].output,
                                   n_in=self.layer_config[-1]['n_out'],
                                   seq_max_len = max_len,
                                   n_out=N_OUT)
        self.layers.append(layer)
        self.layer_config.append({'max_len': max_len})

    def n_params(self):
        """ number of params in this model"""
        def get_layer_nparam(layer):
            prms = layer.params
            ret = sum([reduce(operator.mul, k.get_value().shape) for k in prms])
            print "Layer {0} has {1} params".format(type(layer), ret)
            return ret
        return sum([get_layer_nparam(l) for l in self.layers])

    def work(self, learning_rate=0.1, n_epochs=60, dataset='mnist.pkl.gz',
             load_all_data=True):
        """ read data and start training"""
        print self.layers
        print self.layer_config
        assert type(self.layers[-1]) in [LogisticRegression,
                                         FixedLengthSoftmax, SequenceSoftmax]

        dataset = read_data(dataset)
        if type(self.layers[-1]) == SequenceSoftmax:
            max_len = self.layer_config[-1]['max_len']
            print "Using Sequence Softmax Output with max_len = {0}".format(max_len)
            shared_io = SharedDataIO(dataset, self.batch_size, load_all_data, max_len)
        else:
            shared_io = SharedDataIO(dataset, self.batch_size, load_all_data)

        print "... compiling"
        layer = self.layers[-1]

        # cost to minimize
        cost = layer.negative_log_likelihood(self.y)

        # all the params to optimize on
        params = list(chain.from_iterable([x.params for x in self.layers]))
        #params = self.layers[-1].params     # only train last layer
        # take derivatives on those params
        grads = T.grad(cost, params)

        # gradient descent on those params
        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - learning_rate * grad_i))

        n_batches = list(shared_io.get_dataset_size())
        n_batches = [x / self.batch_size for x in n_batches]

        if load_all_data:
            datasets = shared_io.shared_dataset
            train_set_x, train_set_y = datasets[0]
            valid_set_x, valid_set_y = datasets[1]
            test_set_x, test_set_y = datasets[2]
            index = T.lscalar()

            # symbolic function to train and update
            train_model = theano.function([index], cost,
                  updates=updates,
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
            do_train_model = theano.function([], cost,
                updates=updates,
                givens={
                    self.x: shared_io.shared_Xs[0],
                    self.y: shared_io.shared_ys[0]})
            def train_model(index):
                shared_io.get_train(index)
                return do_train_model()

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
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_batches[0], patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False

        logger = ParamsLogger(self.input_shape)
        progressor = Progressor(n_epochs)

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            if epoch > 1: progressor.report(1, True)
            logger.save_params(epoch, self.layers, self.layer_config)
            for minibatch_index in xrange(n_batches[0]):
                iter = (epoch - 1) * n_batches[0] + minibatch_index

                if iter % 100 == 0 or (iter % 10 == 0 and iter < 30) or (iter < 5):
                    print 'training @ iter = ', iter
                cost_ij = train_model(minibatch_index)


                if (iter + 1) % validation_frequency == 0 or iter < 3:
                    # do a validation

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_batches[1])]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                          (epoch, minibatch_index + 1, n_batches[0], \
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [test_model(i) for i in xrange(n_batches[2])]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of best '
                               'model %f %%') %
                              (epoch, minibatch_index + 1, n_batches[0],
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i,'\
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        print "Usage: {0} dataset.pkl.gz".format(sys.argv[0])
        sys.exit(0)
    print "Dataset: ", dataset
    #train_set = read_data(dataset)[0]
    #shape = train_set[0][0].shape
    shape = (200, 200)
    print "Input img size is {0}".format(shape)

    if len(shape) == 1:
        assert int(np.sqrt(shape[0])) == np.sqrt(shape[0])
        s = int(np.sqrt(shape[0]))
        img_size = (s, s)
        multi_output = False
    else:
        img_size = shape
        # if input is not square, then probably is multiple output.
        multi_output = True

    # config the nn
    nn = NNTrainer(300, img_size, multi_output=multi_output)

    # a NN with two conv-pool layer
    # params are: (n_filters, filter_size), pooling_size
    nn.add_convpoollayer((20, 5), 2)
    nn.add_convpoollayer((50, 5), 2)
    nn.add_convpoollayer((20, 5), 2)

    nn.add_hidden_layer(n_out=500, activation=T.tanh)
    if multi_output:
        nn.add_sequence_softmax(3)
        #nn.add_nLR_layer(2)
    else:
        nn.add_LR_layer()
    print "Network has {0} params in total.".format(nn.n_params())
    nn.work(dataset=dataset, n_epochs=100, load_all_data=False)

# Usage: ./multi_convolution_mlp.py dataset.pkl.gz
