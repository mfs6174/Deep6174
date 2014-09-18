#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: fixed_length_softmax.py
# Date: Wed Sep 17 17:18:19 2014 -0700
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

from LR import LogisticRegression

class FixedLengthSoftmax(object):
    """ Combination of multiple LogisticRegression output layer"""
    NAME = 'fl-sm'

    def __init__(self, input_train, input_test, input_shape,
                 n_out, num_out):
        """ n_out: number of output units for each softmax output layer
            num_out: number of output softmax layer
        """
        self.num_out = num_out
        self.n_out = n_out
        self.input_shape = input_shape

        self.LRs = [LogisticRegression(input_train, input_test,
                                       input_shape, n_out)
                    for _ in range(self.num_out)]

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
            self.params[2 * k + 1].set_value(bs[k].astype('float32'))

    def get_params(self):
        Ws = [k.get_value() for k in self.params[::2]]
        bs = [k.get_value() for k in self.params[1::2]]
        return {"Ws": Ws, "bs": bs
                'n_out': self.n_out,
                'input_shape': self.input_shape,
                'num_out': self.num_out
               }

    @staticmethod
    def build_layer_from_params(params, rng, input_train, input_test=None):
        layer = FixedLengthSoftmax(input_train, input_test,
                                   params['input_shape'],
                                   params['n_out'],
                                   params['num_out'])
        if 'Ws' in params:
            layer.set_params(params['Ws'], params['bs'])
        return layer
