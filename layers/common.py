#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: common.py
# Date: Tue Sep 16 23:46:33 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import theano.tensor as T

def ReLu(x):
    return T.maximum(x, 0.0)

def dropout_from_tensor(rng, input, p):
    """ p is the dropout probability
    """
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=1-p, size=input.shape)

    # The cast is important because int * float32 = float64 which pulls things off the gpu
    output = input * T.cast(mask, theano.config.floatX)
    return output


class Layer(object):
    def __init__(self, rng, input_train, input_test):
        self.params = []
        self.dropout = 0.0
        self.input_train = input_train
        self.rng = rng
        if input_test is None or input_test is input_train:
            self.input_test = input_train
            self.has_dropout_input = False
        else:
            self.input_test = input_test
            self.has_dropout_input = True

    def get_params(self):
        """ get params dict to save"""
        pass

    def get_output_train(self):
        output = self.output_train
        if self.dropout == 0.0:
            return output
        else:
            return dropout_from_tensor(self.rng, output, self.dropout)

    def get_output_test(self):
        return self.output_test * (1 - self.dropout)

    def get_output_shape(self):
        pass
