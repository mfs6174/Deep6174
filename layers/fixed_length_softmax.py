#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: fixed_length_softmax.py
# Date: Mon Aug 04 00:10:54 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cPickle
import itertools
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression

class FixedLengthSoftmax(object):
    """ Combination of multiple LogisticRegression output layer"""

    def __init__(self, input, n_in, n_out, num_out):
        """ n_in: number of input units
            n_out: number of output units for each LR output layer
            num_out: number of output layer
        """

        print "FixedLengthSoftmax with n_in={0}, n_out={1}, num_out={2}".format(n_in, n_out, num_out)
        self.num_out = num_out
        self.LRs = [LogisticRegression(input, n_in, n_out) for _ in range(self.num_out)]

        self.params = list(itertools.chain.from_iterable([lr.params for lr in
                                                          self.LRs]))

        self.p_y_given_x = [k.p_y_given_x for k in self.LRs]

    def negative_log_likelihood(self, y):
        """ y: a matrix, each row is the correct labels of each example
        """
        y = y.dimshuffle(1, 0)

        ret = sum([self.LRs[k].negative_log_likelihood(y[k]) for k in
                   range(self.num_out)]) / self.num_out

        return ret

    def errors(self, y):
        """ y: a matrix, each row is the correct labels of each example
        """
        y = y.dimshuffle(1, 0)
        return sum([self.LRs[k].errors(y[k]) for k in
                    range(self.num_out)]) / self.num_out

    def set_params(self, Ws, bs):
        assert self.num_out == len(Ws) and len(Ws) == len(bs)
        for k in range(self.num_out):
            self.params[2 * k].set_value(Ws[k].astype('float32'))
            self.params[2 * k + 1].set_value(bs[k].flatten().astype('float32'))

    def get_params(self):
        Ws = [k.get_value() for k in self.params[::2]]
        bs = [k.get_value() for k in self.params[1::2]]
        return {"Ws": Ws, "bs": bs}
