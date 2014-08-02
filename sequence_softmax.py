#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: sequence_softmax.py
# Date: Sat Aug 02 01:52:01 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cPickle
from itertools import chain
import gzip
import os
import sys
import time

import numpy
import numpy as np

import theano
import theano.tensor as T
import theano.printing as PP

class SequenceSoftmax(object):
    def __init__(self, input, n_in, seq_max_len, n_out):
        """
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        possible length is 1 ... seq_max_len

        it is actually a layer with (seq_max_len + seq_max_len * n_out) output.
        """
        self.n_softmax = seq_max_len + 1
        self.n_out = n_out

        def gen_W(out, k):
            # TODO init with other values
            return theano.shared(value=numpy.zeros((n_in, out),
                                             dtype=theano.config.floatX),
                            name='W' + str(k), borrow=True)
        self.Ws = [gen_W(seq_max_len, 0)]
        self.Ws.extend([gen_W(n_out, _ + 1) for _ in range(seq_max_len)])

        def gen_b(out, k):
            return theano.shared(value=numpy.zeros((out,),
                                                   dtype=theano.config.floatX),
                                 name='b' + str(k), borrow=True)

        self.bs = [gen_b(seq_max_len, 0)]
        self.bs.extend([gen_b(n_out, _ + 1) for _ in range(seq_max_len)])

        self.p_y_given_xs = [T.nnet.softmax(T.dot(input, self.Ws[k]) +
                                            self.bs[k]) for k in
                             xrange(self.n_softmax)]
        self.pred = [T.argmax(self.p_y_given_xs[k], axis=1) for k in
                     xrange(self.n_softmax)]
        self.pred = T.stacklists(self.pred).dimshuffle(1, 0)
        # self.pred[idx]: output labels of the 'idx' input

        self.params = self.Ws
        self.params.extend(self.bs)

    def negative_log_likelihood(self, y):
        """ y: a batch_size x n_softmax 2d matrix. each row: (len, l1, l2, l3, ...)
        """
        M = [T.log(self.p_y_given_xs[k])[T.arange(y.shape[0]), y[:,k]] for k in range(self.n_softmax)]
        M = T.stacklists(M)
        #return -T.sum(T.sum(M)) / y.shape[0]

        # switch line and row
        M = M.dimshuffle(1, 0)

        #M = [T.sum(M[:t[0] + 1]) for t in M]
        #return -T.sum(M) / y.shape[0]

        #from operator import add
        def f(probs, label):
            # + 1 is real sequence length, + 1 include first element (length)
            return T.sum(probs[:label[0] + 1 + 1])
        sr, su = theano.map(fn=f, sequences=[M, y])

        return -T.sum(sr) / y.shape[0]

    def errors(self, y):
        if not y.dtype.startswith('int'):
            raise NotImplementedError()

        def f(pred, label):
            return T.mean(T.neq(pred[:label[0] + 2], label[:label[0] + 2]))
        sr, su = theano.map(fn=f, sequences=[self.pred, y])
        return T.sum(sr) / y.shape[0]

        #return sum([T.mean(T.neq(self.pred[k], y[:,k])) for k in
                        #range(self.n_softmax)]) / self.n_softmax

        #from operator import mul
        #corr = reduce(mul, [PP.Print("neq for each")(T.neq(self.pred, y[:,k])) for k in
                            #range(self.n_softmax)])
        #return T.mean(corr)

    def set_params(self, Ws, bs):
        assert self.n_softmax == len(Ws) and len(Ws) == len(bs)
        for k in range(self.n_softmax):
            self.Ws[k].set_value(Ws[k].astype('float32'))
            self.bs[k].set_value(bs[k].astype('float32'))

# The Following is modified from logistic_sgd.py

from dataio import read_data
def load_data(dataset):
    train_set, valid_set, test_set = read_data(dataset)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
# 1 + 1 is the length of sequence
        data_y = [[1, t, t] for t in data_y]
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.imatrix('y')  # the labels are presented as 1D vector of
                           # [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = SequenceSoftmax(input=x, n_in=28 * 28, seq_max_len = 2, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    print "Compiling..."
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    params = classifier.params
    # compute the gradient of cost with respect to theta = (W,b)
    grads = T.grad(cost, params)

    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    theano.printing.pydotprint(train_model, outfile='out.png' )
    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    sgd_optimization_mnist()
