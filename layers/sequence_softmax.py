#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: sequence_softmax.py
# Date: Thu Sep 04 21:30:03 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cPickle
from itertools import chain, izip
import gzip
import os
import sys
import time
from copy import copy

import numpy
import numpy as np

import theano
import theano.tensor as T
import theano.printing as PP
import scipy.io as sio

class SequenceSoftmax(object):
    def __init__(self, input, n_in, seq_max_len, n_out, dropout_input=None):
        """
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :param dropout_input: symbolic variable describing the input with dropout
               if provided, this should be used for train

        possible length is 1 ... seq_max_len

        it is actually a layer with (seq_max_len + seq_max_len * n_out) output.
        """
        self.n_softmax = seq_max_len + 1
        self.n_out = n_out

        # generate n_softmax W matrices
        def gen_W(out, k):
            # TODO init with other values
            return theano.shared(value=numpy.zeros((n_in, out),
                                             dtype=theano.config.floatX),
                            name='W' + str(k), borrow=True)
        self.Ws = [gen_W(seq_max_len, 0)]
        self.Ws.extend([gen_W(n_out, _ + 1) for _ in range(seq_max_len)])

        # generate n_softmax b vectors
        def gen_b(out, k):
            return theano.shared(value=numpy.zeros((out,),
                                                   dtype=theano.config.floatX),
                                 name='b' + str(k), borrow=True)

        self.bs = [gen_b(seq_max_len, 0)]
        self.bs.extend([gen_b(n_out, _ + 1) for _ in range(seq_max_len)])

        assert len(self.Ws) == self.n_softmax
        assert len(self.bs) == self.n_softmax

        # p_y_given_x[k]: kth output for all y, each of size (batch_size * n_out)
        self.p_y_given_x = [T.nnet.softmax(T.dot(input, self.Ws[k]) +
                                            self.bs[k]) for k in
                             xrange(self.n_softmax)]

        # self.pred[idx]: output labels of the 'idx' input
        self.pred = [T.argmax(self.p_y_given_x[k], axis=1) for k in
                     xrange(self.n_softmax)]
        self.pred = T.stacklists(self.pred).dimshuffle(1, 0)

        # when using dropout, log_likelihood should be calculated
        # using dropout input for training
        if dropout_input is not None:
            self.p_y_given_x = [T.nnet.softmax(T.dot(dropout_input, self.Ws[k]) +
                                                self.bs[k]) for k in
                                 xrange(self.n_softmax)]

        self.params = copy(self.Ws)
        self.params.extend(self.bs)

    def negative_log_likelihood(self, y):
        """ y: a batch_size x n_softmax 2d matrix. each row: (len, l1, l2, l3, ...)
        """
        batch_size = y.shape[0]
        rg = T.arange(batch_size)       # range of the matrix indices

        loglikelihood = T.log(self.p_y_given_x[1:])
        # correct label is the index used in the loglikelihood matrix
        idxs = y.dimshuffle(1, 0)[1:]
        # select the loglikelihood of the specific label from the matrix
        sr, _ = theano.map(fn=lambda l, idx: l[rg, idx],
                           sequences=[loglikelihood, idxs])
        M = sr.dimshuffle(1, 0)
        length_probs = T.log(self.p_y_given_x[0])[rg, y[:,0]]

        # sum of log_likelihood for digits & length, on each example
        def f(probs, label, length_prob):
            return T.sum(probs[:label[0] + 1]) + length_prob
        sr, _ = theano.map(fn=f, sequences=[M, y, length_probs])

        #log_matrices = [T.log(self.p_y_given_x[k]) for k in range(self.n_softmax)]
        #idxs = [y[:,k] for k in range(self.n_softmax)]
        ##idxs[3] = PP.Print("idx-1")(idxs[3])
        #M = [m[rg, idx] for m, idx in izip(log_matrices, idxs)]
        #M = T.stacklists(M)
        #M = M.dimshuffle(1, 0)
        #def f(probs, label):
            #return T.sum(probs[:label[0] + 1 + 1])
        #sr, su = theano.map(fn=f, sequences=[M, y])

        return -T.mean(sr)

    def errors(self, y):
        if not y.dtype.startswith('int'):
            raise NotImplementedError()

        def f(pred, label):
            # label[0] + 1 == length of sequence;
            # label[0] + 2 == length of the label (including sequence and length)
            return T.sum(T.neq(pred[1:label[0] + 2], label[1:label[0] + 2]))
        sr, _ = theano.map(fn=f, sequences=[self.pred, y])

        # sum of lengths
        len_sum = T.sum(y[:,0] + 1)
        # avg of errors
        return T.cast(T.sum(sr), 'float32') / len_sum

    def set_params(self, Ws, bs):
        assert self.n_softmax == len(Ws) and len(Ws) == len(bs)
        for k in range(self.n_softmax):
            self.Ws[k].set_value(Ws[k].astype('float32'))
            self.bs[k].set_value(bs[k].astype('float32'))

    def get_params(self):
        Ws = [k.get_value(borrow=True) for k in self.Ws]
        bs = [k.get_value(borrow=True) for k in self.bs]
        return {"Ws": Ws, "bs": bs}

    def save_params_mat(self, basename):
        """ save params in .mat format
            file name will be built by adding suffix to 'basename'
        """
        prms = self.get_params()
        Ws = prms['Ws']
        bs = prms['bs']
        def save(name, idx):
            """ save each classifier separately"""
            sio.savemat(basename + '-{0}.mat'.format(name),
                        {'W{0}'.format(name): Ws[idx],
                         'b{0}'.format(name): bs[idx]})
        save('length', 0)
        for k in range(1, self.n_softmax):
            save('position{0}'.format(k), k)
