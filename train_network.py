#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: train_network.py
import os
import sys
import time
from itertools import chain, izip
import cPickle
import gzip
import operator

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

        self.x = T.matrix('x')
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

    #def add_convpoollayer(self, filter_config, pool_size, norm='mean', maxout=0):
        #""" filter_config: tuple(nfilters, filter_size)
            #pool_size: tuple or int
        #"""
        #if type(pool_size) == int:
            #pool_size = (pool_size, pool_size)
        #if len(self.layer_config):
            #assert type(self.layers[-1]) == ConvPoolLayer, \
                #"currently convpool layer must come after a convpool layer"


        #if not len(self.layers):
            ## This is the First Convolutional Layer
            #image_shape = (self.batch_size, 3 if self.rgb_input else 1) + self.input_shape
            #filter_shape = (filter_config[0], 3 if self.rgb_input else 1,
                            #filter_config[1], filter_config[1])

            #layer = ConvPoolLayer(self.rng, input=self.orig_input,
                        #image_shape=image_shape, filter_shape=filter_shape,
                        #poolsize=pool_size, norm=norm, maxout=maxout)
            ## maxout is only supported in first layer
            #self.layers.append(layer)
        #else:
            #assert maxout == 0
            ## this is a layer following previous convolutional layer
            ## calculate new config based on config of previous layer
            #last_config = self.layer_config[-1]
            #last_filt_shape = last_config['filter_shape']
            #last_img_shape = last_config['image_shape']
            #now_img_size = [last_img_shape[k] / last_config['pool_size'][k - 2] for k in [2, 3]]
            #if last_config['maxout']:
                #last_n_filter = last_filt_shape[0] / last_config['maxout']
            #else:
                #last_n_filter = last_filt_shape[0]
            #image_shape = (self.batch_size, last_n_filter) + tuple(now_img_size)
            #filter_shape = (filter_config[0], last_n_filter,
                            #filter_config[1], filter_config[1])

            #layer = ConvPoolLayer(self.rng, input=self.layers[-1].output,
                        #image_shape=image_shape, filter_shape=filter_shape,
                        #poolsize=pool_size, norm=norm)
            #self.layers.append(layer)

        ## save the config for next layer to use
        #self.layer_config.append({'image_shape': image_shape,
                                  #'filter_shape': filter_shape,
                                  #'pool_size': pool_size,
                                  #'maxout': maxout})

    #def _get_nin_after_convlayer(self):
        #""" get n_in of next layer after a convpool layer"""
        #assert type(self.layers[-1]) == ConvPoolLayer
        #last_config = self.layer_config[-1]
        ## calculate the image size generated by last layer
        #last_img_shape = last_config['image_shape']
        #last_filt_shape = last_config['filter_shape']
        #now_img_size = [last_img_shape[k] / last_config['pool_size'][k-2] for k in [2, 3]]
        #last_filt_shape = last_config['filter_shape']
        #return last_filt_shape[0] * now_img_size[0] * now_img_size[1]

    #def add_hidden_layer(self, n_out, activation):
        #if not len(self.layer_config):
            #assert len(self.layer_config), "hidden layer must be added after ConvPool layer"
            #return

        #last_config = self.layer_config[-1]
        #if type(self.layers[-1]) == ConvPoolLayer:
            #input = self.layers[-1].output.flatten(2)
            #layer = HiddenLayer(self.rng, input=input,
                                #n_in=self._get_nin_after_convlayer(),
                                #n_out=n_out, activation=activation)
            #self.layers.append(layer)
            #self.layer_config.append({'n_out': n_out, 'activation': activation})
        #else:
            #assert "currently hidden layers must follow a conv layer"

    #def add_hidden_layers(self, layer_sizes, dropout_rate, activation=T.tanh):
        #if not len(self.layer_config):
            #assert len(self.layer_config), "hidden layer must be added after ConvPool layer"
            #return

        #last_config = self.layer_config[-1]
        #if type(self.layers[-1]) == ConvPoolLayer:
            #input = self.layers[-1].output.flatten(2)
            #layer = DropoutMLP(self.rng, input,
                               #self._get_nin_after_convlayer(),
                               #layer_sizes,
                               #dropout_rate,
                               #activation)
            #self.layers.append(layer)
            #self.layer_config.append({'layer_sizes': layer_sizes, 'activation': activation})
        #else:
            #assert "currently hidden layers must follow a conv layer"

    #def add_LR_layer(self):
        #""" Can only be used as output layer"""
        #last_layer = self.layers[-1]
        #if type(last_layer) == HiddenLayer:
            #layer = LogisticRegression(input=last_layer.output,
                                       #n_in=self.layer_config[-1]['n_out'],
                                       #n_out=N_OUT)
        #elif type(last_layer) == DropoutMLP:
            #layer = LogisticRegression(input=last_layer.output,
                                       #n_in=self.layer_config[-1]['layer_sizes'][-1],
                                       #n_out=N_OUT,
                                       #dropout_input=last_layer.dropout_output)
        #elif type(last_layer) == ConvPoolLayer:
            #input = last_layer.output.flatten(2)
            #layer = LogisticRegression(input=input,
                                      #n_in=self._get_nin_after_convlayer(),
                                      #n_out=N_OUT)

        #self.layers.append(layer)
        #self.layer_config.append(None)

    #def add_nLR_layer(self, n):
        #""" Fixed length sequence output layer, which is equivalent to n LR layers
        #Can only be used as output layer"""
        #if type(self.layers[-1]) == HiddenLayer:
            #layer = FixedLengthSoftmax(input=self.layers[-1].output,
                                       #n_in=self.layer_config[-1]['n_out'],
                                       #n_out=N_OUT,
                                       #num_out=n)
        #elif type(self.layers[-1])== ConvPoolLayer:
            #input = self.layers[-1].output.flatten(2)
            #layer = FixedLengthSoftmax(input=input,
                                      #n_in=self._get_nin_after_convlayer(),
                                      #n_out=N_OUT,
                                      #num_out=n)
        #else:
            #assert False
        #self.layers.append(layer)
        #self.layer_config.append(None)

    #def add_sequence_softmax(self, max_len):
        #""" SequenceSoftmax layer
        #Can only be used as output layer"""
        #last_layer = self.layers[-1]
        #if type(last_layer) == HiddenLayer:
            #layer = SequenceSoftmax(input=last_layer.output,
                                       #n_in=self.layer_config[-1]['n_out'],
                                       #seq_max_len = max_len,
                                       #n_out=N_OUT)
        #elif type(last_layer) == DropoutMLP:
            #layer = SequenceSoftmax(input=last_layer.output,
                               #n_in=self.layer_config[-1]['layer_sizes'][-1],
                               #seq_max_len = max_len,
                               #n_out=N_OUT,
                               #dropout_input=last_layer.dropout_output)
        #else:
            #assert False
        #self.layers.append(layer)
        #self.layer_config.append({'max_len': max_len})

    def n_params(self):
        """ Calculate total number of params in this model"""
        def get_layer_nparam(layer):
            prms = layer.params
            ret = sum([reduce(operator.mul, k.get_value().shape) for k in prms])
            if ret > 0:
                print "Layer {0} has {1} params".format(type(layer), ret)
            return ret
        return sum([get_layer_nparam(l) for l in self.layers])

    def work(self, init_learning_rate=0.1, n_epochs=60, dataset_file='mnist.pkl.gz',
             load_all_data=True):
        """ read data and train"""
        print self.layers
        print self.layer_config
        assert type(self.layers[-1]) in [SequenceSoftmax, LogisticRegression]

        dataset = read_data(dataset_file)
        if type(self.layers[-1]) == SequenceSoftmax:
            max_len = self.layer_config[-1]['seq_max_len']
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

        # save last updates for momentum
        if not self.last_updates:
            self.last_updates = []
            for param in params:
                self.last_updates.append(
                    theano.shared(
                        np.zeros(param.get_value(borrow=True).shape,
                                 dtype=theano.config.floatX)
                    ))
        assert len(self.last_updates) == len(params), 'last updates don\'t match params'

        def gen_updates_with_learning_rate(rate):
            # gradient descent on those params
            updates = []
            for param_i, grad_i, last_update in izip(params, grads, self.last_updates):
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
            train_model = theano.function([index, lr_rate], cost,
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
            do_train_model = theano.function([lr_rate], cost,
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
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_batches[0], patience / 2)
        validation_frequency /= 2
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

        logger = ParamsLogger(dataset_file + '-models')
        progressor = Progressor(n_epochs)
        rate_provider = LearningRateProvider(dataset_file + '-learnrate', init_learning_rate)

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            if epoch > 1: progressor.report(1, True)
            # save params at the beginning of each epoch
            logger.save_params(epoch, self)
            learning_rate = rate_provider.get_rate(epoch)
            print "In epoch {0}: learning rate is {1}".format(epoch, learning_rate)
            for minibatch_index in xrange(n_batches[0]):
                iter = (epoch - 1) * n_batches[0] + minibatch_index

                if iter % 200 == 0 or (iter % 10 == 0 and iter < 30) or (iter < 5):
                    print 'training @ iter = ', iter
                cost_ij = train_model(minibatch_index, learning_rate)


                if (iter + 1) % validation_frequency == 0 or iter in [5, 10, 20, 30, 60, 100, 200]:
                    # do a validation:

                    # compute zero-one loss on validation set
                    validation_losses = [test_model(i) for i
                                         in xrange(n_batches[1])]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('After epoch %i, minibatch %i/%i, test error %f %%' % \
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
                        if best_validation_loss < 0.85:
                            # save best params
                            print 'Yay! Saving best model ...'
                            logger.save_params('best', self)

                #if patience <= iter:
                    #done_looping = True
                    #break

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
    batch = 500
    if len(img_size) == 3:
        shape = (batch, ) + img_size
    else:
        shape = (batch, 1) + img_size
    nn = NNTrainer(shape, multi_output=multi_output)

    nn.add_layer(ConvLayer, {'filter_shape': (20, 5, 5), 'keep_size': False})
    nn.add_layer(PoolLayer, {'pool_size': 2})
    nn.add_layer(ConvLayer, {'filter_shape': (50, 5, 5), 'keep_size': False})
    nn.add_layer(PoolLayer, {'pool_size': 2})
    nn.add_layer(ConvLayer, {'filter_shape': (50, 5, 5), 'keep_size': False})
    nn.add_layer(PoolLayer, {'pool_size': 2})
    nn.add_layer(FullyConnectedLayer, {'n_out': 500, 'activation': T.tanh})

    if multi_output:
        nn.add_layer(SequenceSoftmax, {'seq_max_len': 4, 'n_out': 10})
        #nn.add_nLR_layer(2)
    else:
        nn.add_layer(LogisticRegression, {'n_out': 10})
    print "Network has {0} params in total.".format(nn.n_params())
    nn.work(init_learning_rate=0.1, dataset_file=dataset, n_epochs=1000,
            load_all_data=load_all)

# Usage: ./train_network.py dataset.pkl.gz
