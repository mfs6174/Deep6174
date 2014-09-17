#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: conv.py
# Date: Tue Sep 16 23:45:37 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import scipy.io as sio
import theano
import numpy
from theano.tensor.nnet import conv

from common import ReLu, dropout_from_tensor, Layer

class ConvLayer(Layer):

    def __init__(self, rng, input_train, input_test,
                 filter_shape, image_shape,
                 keep_size,
                 activation, dropout):
        """ filter_shape: 3D tuple of (n_channel, size_w, size_h)"""
        super(ConvLayer, self).__init__(rng, input_train, input_test)
        self.dropout = dropout
        self.keep_size = keep_size
        self.activation = activation
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        conv_filter_shape = (filter_shape[0], image_shape[1],
                             filter_shape[1], filter_shape[2])

        # there are
        # "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:] * image_shape[1])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" / pooling size
        fan_out = filter_shape[0] * numpy.prod(filter_shape[1:]) / 4
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=conv_filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        def do_conv(input):
            if self.keep_size:
                conv_out = conv.conv2d(input=input, filters=self.W,
                        filter_shape=conv_filter_shape, image_shape=image_shape,
                                       border_mode='full')
                mid = (int(numpy.floor(filter_shape[1] / 2.)),
                       int(numpy.floor(filter_shape[2] / 2.)))
                # keep representation size
                conv_out = conv_out[:, :, mid[0]:-mid[0], mid[1]:-mid[1]]
            else:
                conv_out = conv.conv2d(input=input, filters=self.W,
                        filter_shape=conv_filter_shape, image_shape=image_shape)
            activate_out = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            return activate_out

        self.output_train = do_conv(input_train)
        if not self.has_dropout_input:
            self.output_test = self.output_train
        else:
            print "In Convlayer: not detecting last layer dropout"
            self.output_test = do_conv(input_test)

        self.params = [self.W, self.b]

    def get_output_shape(self):
        if self.keep_size:
            return (self.image_shape[0], self.filter_shape[0],
                    self.image_shape[2], self.image_shape[3])
        else:
            return (self.image_shape[0], self.filter_shape[0],
                    self.image_shape[2] - self.filter_shape[1] + 1,
                    self.image_shape[3] - self.filter_shape[2] + 1)


    def get_params(self):
        return {'W': self.W.get_value(borrow=True),
                'b': self.b.get_value(borrow=True),
                'activation': self.activation,
                'filter_shape': self.filter_shape,
                'input_shape': self.image_shape,
                'dropout': self.dropout}

    def save_params_mat(self, basename):
        """ save params in .mat format
            file name will be built by adding suffix to 'basename'
        """
        params = self.get_params()
        sio.savemat(basename + '.mat', params)

    @staticmethod
    def build_layer_from_params(params, rng, input_train, input_test=None):
        layer = ConvLayer(rng, input_train, input_test,
                          params['filter_shape'],
                          params['input_shape'],
                          params.get('keep_size', True),
                          params.get('activation', ReLu),
                          params.get('dropout', 0.0))

        if 'W' in params:
            layer.W.set_value(params['W'].astype(theano.config.floatX))
            layer.b.set_value(params['b'].flatten().astype(theano.config.floatX))
        return layer
