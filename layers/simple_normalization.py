#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: pool.py
# Date: Thu Sep 18 01:55:04 2014 -0700
# Author: Xi-Jin Zhang <zhangxijin91@gmail.com>

import __builtin__
import theano
import theano.tensor as T
import theano.printing as PP
import numpy as np

from common import Layer

class SimpleNormalizingLayer(Layer):
    NAME = 'sn'

    def __init__(self, input_train, input_test,
                 image_shape, provide_stdvar, provide_mean):
        super(SimpleNormalizingLayer, self).__init__(None, input_train, input_test)
        self.image_shape = image_shape
        self.provide_stdvar = provide_stdvar
        self.provide_mean = provide_mean
        if self.provide_stdvar is not None:
            def get_stdvar(intput):
                return self.provide_stdvar
        else:
            def get_stdvar(input):
                return T.std(input)
        if self.provide_mean is not None:
            def get_mean(intput):
                return self.provide_mean
        else:
            def get_mean(input):
                return T.mean(input)
            
        def gen_normalize(input):
            stdvar = get_stdvar(input)
            ret = T.zeros_like(input)
            for i in range(image_shape[0]):
                rt[i,:] = (input[i,:]-get_mean(input[i,:]))/stdvar
            return ret    

        self.output_train = gen_normalize(self.input_train)
        if self.has_dropout_input:
            self.output_test = gen_normalize(self.input_test)

    def get_output_shape(self):
        return image_shape
    
    def get_params(self):
        return {'provide_stdvar': self.provide_stdvar,
                'input_shape': self.image_shape,
                'provide_mean': self.provide_mean
               }

    @staticmethod
    def build_layer_from_params(params, rng, input_train, input_test=None):
        layer = SimpleNormalizingLayer(input_train, input_test,
                                       params['input_shape'],
                                       params.get('provide_stdvar',None),
                                       params.get('provide_mean', None))
        return layer

