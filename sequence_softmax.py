#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: sequence_softmax.py
# Date: Tue Aug 05 20:51:58 2014 -0700
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

        assert len(self.Ws) == self.n_softmax
        assert len(self.bs) == self.n_softmax

        self.p_y_given_x = [T.nnet.softmax(T.dot(input, self.Ws[k]) +
                                            self.bs[k]) for k in
                             xrange(self.n_softmax)]
        # p_y_given_x[k] = kth output for all y, each of size (batch_size * n_out)
        self.pred = [T.argmax(self.p_y_given_x[k], axis=1) for k in
                     xrange(self.n_softmax)]
        self.pred = T.stacklists(self.pred).dimshuffle(1, 0)
        # self.pred[idx]: output labels of the 'idx' input

        self.params = copy(self.Ws)
        self.params.extend(self.bs)

    def negative_log_likelihood(self, y):
        """ y: a batch_size x n_softmax 2d matrix. each row: (len, l1, l2, l3, ...)
        """
        batch_size = y.shape[0]
        log_matrices = [T.log(self.p_y_given_x[k]) for k in range(self.n_softmax)]
        idxs = [y[:,k] for k in range(self.n_softmax)]
        #idxs[3] = PP.Print("idx-1")(idxs[3])
        rg = T.arange(batch_size)
        M = [m[rg, idx] for m, idx in izip(log_matrices, idxs)]
        M = T.stacklists(M)

        # switch line and row
        M = M.dimshuffle(1, 0)

        #from operator import add
        def f(probs, label):
            # + 1 is real sequence length, + 1 include first element (length)
            return T.sum(probs[:label[0] + 1 + 1])
        sr, su = theano.map(fn=f, sequences=[M, y])
        return -T.mean(sr)

    def errors(self, y):
        if not y.dtype.startswith('int'):
            raise NotImplementedError()

        def f(pred, label):
            return T.mean(T.neq(pred[1:label[0] + 2], label[1:label[0] + 2]))
        sr, su = theano.map(fn=f, sequences=[self.pred, y])
        return T.mean(sr)

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

    def get_params(self):
        Ws = [k.get_value(borrow=True) for k in self.Ws]
        bs = [k.get_value(borrow=True) for k in self.bs]
        return {"Ws": Ws, "bs": bs}
