#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: conv.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import scipy.io as sio
import theano
import numpy as np
from theano.tensor.nnet import conv
import theano.printing as PP
import theano.tensor as T

from common import Layer
float_x = theano.config.floatX
class InputLayer(Layer):
    """ A input and preprocesing layer"""

    NAME = 'input'

    def __init__(self, rng, input_train, input_test,input_label,
                 image_shape, label_shape, deal_label,
                 angle,
                 #below here, not tested:
                 translation,
                 zoom,
                 magnitude,
                 pflip,
                 invert_image,
                 sigma,
                 nearest):
        """ deal_label: for patch predicting, need to process label
        """

        if invert_image:
            if input_train is not None:
                input_train = 1-input_train
            if input_test is not None:
                input_test = 1-input_test

        super(InputLayer, self).__init__(rng, input_train, input_test)
        self.input_train = self.input_train.reshape(image_shape)
        self.input_test = self.input_test.reshape(image_shape)
        self.label_shape = label_shape
        self.image_shape = image_shape
        self.deal_label = deal_label
        self.input_label = input_label.reshape(label_shape)
        
        self.angle = angle
        self.translation = translation
        self.zoom = zoom
        self.magnitude = magnitude
        self.sigma = sigma
        self.plip=plip
        self.invert = invert_image
        self.nearest = nearest
        
        assert zoom > 0
        self.need_proc = (not (magnitude or translation or pflip or angle) and zoom == 1)

        # do the preprocessing
        def do_preproc(input,label):
            if not self.need_proc:
                return input,label
            if self.deal_label:
                assert len(label_shape)==3
            else:
                outlabel=label
            srs=self.rng
            w = self.image_shape[-1]
            h = self.image_shape[-2]
            target = T.as_tensor_variable(np.indices((self.image_shape[-2], self.image_shape[-1])))
            if self.deal_label:
                tarlab = T.as_tensor_variable(np.indices((self.label_shape[-2], label_shape[-1])))
                lw = self.label_shape[-1]
                lh = self.label_shape[-2]

            # Translate
            if self.translation:
                transln = self.translation * srs.uniform((2, 1, 1), -1)
                target += transln
                if self.deal_label:
                    tarlab += transln

            # Apply elastic transform
            if self.magnitude:
                # Build a gaussian filter
                var = self.sigma ** 2
                filt = np.array([[np.exp(-.5 * (i * i + j * j) / var)
                                 for i in range(-self.sigma, self.sigma + 1)]
                                 for j in range(-self.sigma, self.sigma + 1)], dtype=float_x)
                filt /= 2 * np.pi * var

                # Elastic
                elast = self.magnitude * srs.normal((2, h, w))
                elast = sigconv.conv2d(elast, filt, (2, h, w), filt.shape, 'full')
                elast = elast[:, self.sigma:h + self.sigma, self.sigma:w + self.sigma]
                target += elast
                if deal_label:
                    raise NotImplementedError()

            # Center at 'about' half way
            if self.zoom-1 or self.angle:
                origin = srs.uniform((2, 1, 1), .25, .75) * \
                         np.array((h, w)).reshape((2, 1, 1))
                if self.deal_label:
                    lorigin = srs.uniform((2, 1, 1), .25, .75) * \
                         np.array((lh, lw)).reshape((2, 1, 1))
                    tarlab -= lorigin
                target -= origin

                # Zoom
                if self.zoom-1:
                    zoomer = T.exp(np.log(self.zoom) * srs.uniform((2, 1, 1), -1))
                    target *= zoomer
                    if self.deal_label:
                        tarlab *= zoomer

                # Rotate
                if self.angle:
                    theta = self.angle * np.pi / 180 * srs.uniform(low=-1)
                    c, s = T.cos(theta), T.sin(theta)
                    rotate = T.stack(c, -s, s, c).reshape((2,2))
                    target = T.tensordot(rotate, target, axes=((0, 0)))
                    if self.deal_label:
                        tarlab = T.tensordot(rotate, targlab, axes=((0, 0)))

                # Uncenter
                target += origin
                if self.deal_label:
                    tarlab -= lorigin

            # Clip the mapping to valid range and linearly interpolate
            transy = T.clip(target[0], 0, h - 1 - .001)
            transx = T.clip(target[1], 0, w - 1 - .001)
            if self.deal_label:
                ltransy = T.clip(tarlab[0], 0, lh - 1 - .001)
                ltransx = T.clip(tarlab[1], 0, lw - 1 - .001)

            if self.nearest:
                vert = T.iround(transy)
                horz = T.iround(transx)
                output = input[:, :, vert, horz]
            else:
                topp = T.cast(transy, 'int32')
                left = T.cast(transx, 'int32')
                fraction_y = T.cast(transy - topp, float_x)
                fraction_x = T.cast(transx - left, float_x)

                output = input[:, :, topp, left] * (1 - fraction_y) * (1 - fraction_x) + \
                         input[:, :, topp, left + 1] * (1 - fraction_y) * fraction_x + \
                         input[:, :, topp + 1, left] * fraction_y * (1 - fraction_x) + \
                         input[:, :, topp + 1, left + 1] * fraction_y * fraction_x
            if self.deal_lab:
                vert = T.iround(ltransy)
                horz = T.iround(ltransx)
                outlabel = label[:, vert, horz]


            # Now add some noise
            if self.pflip:
                mask = srs.binomial(n=1, p=self.pflip, size=input.shape, dtype=float_x)
                output = (1 - output) * mask + output * (1 - mask)
            return output,outlabel

        self.output_train,self.output_train_label = do_preproc(self.input_train,self.input_label)
        if not self.has_dropout_input:
            self.output_test = self.input_train
        else:
            self.output_test = self.input_test

        self.output_test_label = self.input_label

        self.output_train = self.output_train.reshape(image_shape[0],-1)
        self.output_test = self.output_test.reshape(image_shape[0],-1)
        self.output_train_label = self.output_train_label.reshape(label_shape[0],-1)
        self.output_test_label = self.output_test_label.reshape(label_shape[0],-1)

        
    def get_output_test_label(self):
        return self.output_test_label

    def get_output_train_label(self):
        return self.output_train_label

    def get_output_shape(self):
        return self.image_shape

    def get_params(self):
        return {'angle': self.angle,
                'translation': self.translation,
                'zoom': self.zoom,
                'magnitude': self.magnitude,
                'sigma': self.sigma,
                'plip': self.plip,
                'invert': self.invert,
                'nearest': self.nearest,
                'deal_label': self.deal_label,
                'label_shape': self.label_shape,
                'input_shape': self.image_shape }

    def save_params_mat(self, basename):
        """ save params in .mat format
            file name will be built by adding suffix to 'basename'
        """
        params = self.get_params()
        sio.savemat(basename + '.mat', params)

    @staticmethod
    def build_layer_from_params(params, rng, input_train, input_label,input_test=None):
        layer = InputLayer(rng, input_train, input_test, input_label,
                           params['image_shape'],
                           params['label_shape'],
                           params.get('deal_label', False),
                           params.get('angle', 0),
                           params.get('translation', 0),
                           params.get('zoom', 1),
                           params.get('magnitude', 0),
                           params.get('pflip', 0),
                           params.get('angle', 0),
                           params.get('invert', False),
                           params.get('sigma', 1),
                           params.get('nearest', True))
        return layer

