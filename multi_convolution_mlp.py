#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: multi_convolution_mlp.py
# Date: Mon Jul 21 02:21:21 2014 -0700
import os
import sys
import time
from itertools import chain
import cPickle
import gzip

import numpy
from numpy import random

import theano
import theano.tensor as T

from params_logger import ParamsLogger
from logistic_sgd import LogisticRegression
from dataio import read_data, save_data, get_dataset_imgsize
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer

class ConfigurableNN(object):
    """ Configurable Neural Network,
        Currently support several convolution-pooling layer followed by hidden layers.
    """
    def __init__(self, batch_size, input_shape):
        self.layer_config = []
        self.layers = []
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.rng = numpy.random.RandomState(23455)

        self.x = T.matrix('x')
        self.y = T.ivector('y')

        self.orig_input = self.x.reshape((self.batch_size, 1) + input_shape)

    def add_convpoollayer(self, filter_config, pool_size):
        """ filter_config: tuple(nfilters, filter_size)
            pool_size: int
        """
        if len(self.layer_config):
            assert type(self.layers[-1]) == LeNetConvPoolLayer, \
                "currently only support conv-pool layer followed by hidden layers"


        if not len(self.layers):
            # This is the First Convolutional Layer
            image_shape = (self.batch_size, 1) + self.input_shape
            filter_shape = (filter_config[0], 1,
                            filter_config[1], filter_config[1])

            layer = LeNetConvPoolLayer(self.rng, input=self.orig_input,
                        image_shape=image_shape, filter_shape=filter_shape,
                        poolsize=(pool_size, pool_size))
            self.layers.append(layer)
        else:
            # this is a layer following previous convolutional layer
            # calculate new config based on config of previous layer
            last_config = self.layer_config[-1]
            now_img_size = (last_config['image_shape'][3] -
                            last_config['filter_shape'][3] + 1) / last_config['pool_size']

            image_shape = (self.batch_size, last_config['filter_shape'][0],
                           now_img_size, now_img_size)
            filter_shape = (filter_config[0], last_config['filter_shape'][0],
                            filter_config[1], filter_config[1])

            layer = LeNetConvPoolLayer(self.rng, input=self.layers[-1].output,
                        image_shape=image_shape, filter_shape=filter_shape,
                        poolsize = (pool_size, pool_size))
            self.layers.append(layer)

        # save the config for next layer to use
        self.layer_config.append({'image_shape': image_shape,
                                  'filter_shape': filter_shape,
                                  'pool_size': pool_size })

    def add_hidden_layer(self, n_out, activation):
        if not len(self.layer_config):
        # XXX do something
            assert len(self.layer_config), "hidden layer must be added after ConvPool layer"
            return

        last_config = self.layer_config[-1]
        if type(self.layers[-1])== LeNetConvPoolLayer:
            input = self.layers[-1].output.flatten(2)
            # calculate the image size generated by last layer
            now_img_size = (last_config['image_shape'][3] -
                            last_config['filter_shape'][3] + 1) / last_config['pool_size']
            layer = HiddenLayer(self.rng, input=input,
                                n_in=last_config['filter_shape'][0] * now_img_size * now_img_size,
                                n_out=n_out, activation=activation)
            self.layers.append(layer)
            self.layer_config.append({'n_out': n_out, 'activation': activation})
        else:
            assert "currently hidden layers must follow a conv layer"

    def add_LR_layer(self):
        """ This must be the last output layer"""
        n_out = 10
        if type(self.layers[-1]) == HiddenLayer:
            layer = LogisticRegression(input=self.layers[-1].output,
                                       n_in=self.layer_config[-1]['n_out'],
                                       n_out=n_out)
        elif type(self.layers[-1])== LeNetConvPoolLayer:
            last_config = self.layer_config[-1]
            input = self.layers[-1].output.flatten(2)
            # calculate the image size generated by last layer
            now_img_size = (last_config['image_shape'][3] -
                            last_config['filter_shape'][3] + 1) / last_config['pool_size']
            layer = LogisticRegression(input=input,
                                      n_in = last_config['filter_shape'][0] * now_img_size * now_img_size,
                                      n_out=n_out)

        self.layers.append(layer)
        self.layer_config.append(None)

    def work(self, learning_rate=0.1, n_epochs=60, dataset='mnist.pkl.gz'):
        """ read data and start training"""
        print self.layers
        print self.layer_config
        assert type(self.layers[-1]) == LogisticRegression

        datasets = read_data(dataset)

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= self.batch_size
        n_valid_batches /= self.batch_size
        n_test_batches /= self.batch_size

        index = T.lscalar()

        layer = self.layers[-1]

        print "... compiling"

        # cost to minimize
        cost = layer.negative_log_likelihood(self.y)

        # symbolic function to test model error
        test_model = theano.function([index], layer.errors(self.y),
                 givens={
                    self.x: test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                    self.y: test_set_y[index * self.batch_size: (index + 1) * self.batch_size]})

        # symbolic function to validate model error
        validate_model = theano.function([index], layer.errors(self.y),
                givens={
                    self.x: valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                    self.y: valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]})

        # all the params to optimize on
        params = list(chain.from_iterable([x.params for x in self.layers]))
        #params = self.layers[-1].params     # only train last layer
        # take derivatives on those params
        grads = T.grad(cost, params)

        # gradient descent on those params
        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - learning_rate * grad_i))

        train_model = theano.function([index],
              cost,
              updates=updates,
              givens={
                self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]})


        print '... training'
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
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

        logger = ParamsLogger()

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            logger.save_params(epoch, self.layers, self.layer_config)
            for minibatch_index in xrange(n_train_batches):
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print 'training @ iter = ', iter
                cost_ij = train_model(minibatch_index)


                if (iter + 1) % validation_frequency == 0:
                    # do a validation

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                          (epoch, minibatch_index + 1, n_train_batches, \
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
                        test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of best '
                               'model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
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

    size = get_dataset_imgsize(dataset)
    print "Input img size is {0}x{0}".format(size)

    # config the nn
    nn = ConfigurableNN(500, (size, size))

    # a NN with two conv-pool layer
    # params are: (n_filters, filter_size), pooling_size
    nn.add_convpoollayer((20, 5), 2)
    nn.add_convpoollayer((20, 5), 2)
    nn.add_convpoollayer((50, 5), 1)
    nn.add_convpoollayer((20, 5), 1)

    nn.add_hidden_layer(n_out=500, activation=T.tanh)
    nn.add_LR_layer()
    nn.work(dataset=dataset, n_epochs=60)

# Usage: ./multi_convolution_mlp.py dataset.pkl.gz
