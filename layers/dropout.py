#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: dropout.py
# Date: Wed Sep 17 14:31:48 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from common import Layer

class DropoutLayer(Layer):
    def __init__(self, rng, input_train, input_test,
                 input_shape, dropout):
        super(DropoutLayer, self).__init__(rng, input_train, input_test)
        self.input_shape = input_shape
        self.dropout = dropout

        self.output_train = input_train
        if self.has_dropout_input:
            self.output_test = self.output_test

    def get_output_shape(self):
        return self.input_shape

    def get_params(self):
        return {'input_shape': self.input_shape,
                'dropout': self.dropout}

    @staticmethod
    def build_layer_from_params(params, rng, input_train, input_test=None):
        layer = DropoutLayer(rng, input_train, input_test,
                             params['input_shape'],
                             params.get('dropout', 0.5))
        return layer
