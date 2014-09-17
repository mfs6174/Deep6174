#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: pool.py
# Date: Wed Sep 17 16:06:37 2014 +0000
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from theano.tensor.signal import downsample
import theano.printing as PP

from common import Layer

class PoolLayer(Layer):
    def __init__(self, input_train, input_test,
                 image_shape, pool_size):
        super(PoolLayer, self).__init__(None, input_train, input_test)
        self.image_shape = image_shape
        if type(pool_size) == int:
            pool_size = (pool_size, pool_size)
        self.pool_size = pool_size

        self.output_train = downsample.max_pool_2d(input=self.input_train,
                                            ds=pool_size,
                                            ignore_border=True)
        if self.has_dropout_input:
            self.output_test = downsample.max_pool_2d(input=self.input_test,
                                            ds=pool_size,
                                            ignore_border=True)

    def get_output_shape(self):
        return (self.image_shape[0], self.image_shape[1],
                self.image_shape[2] / self.pool_size[0],
                self.image_shape[3] / self.pool_size[1])

    def get_params(self):
        return {'pool_size': self.pool_size,
                'input_shape': self.image_shape}

    @staticmethod
    def build_layer_from_params(params, rng, input_train, input_test=None):
        layer = PoolLayer(input_train, input_test,
                          params['input_shape'],
                          params['pool_size'])
        return layer

