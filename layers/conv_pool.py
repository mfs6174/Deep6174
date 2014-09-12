"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import cPickle
import gzip
import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import theano.printing as PP
import scipy.io as sio

def mean_filter(kernel_size):
    s = kernel_size ** 2
    x = numpy.repeat(1./s, s).reshape((kernel_size, kernel_size))
    return x

class ConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input,
                 filter_shape, image_shape,
                 poolsize=(2, 2), activation='relu', norm='mean', maxout=0):
        """
        Allocate a ConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        print image_shape, filter_shape
        assert image_shape[1] == filter_shape[1]
        self.input = input
        if type(poolsize) == int:
            poolsize = (poolsize, poolsize)
        self.pool_size = poolsize
        self.norm = norm
        self.maxout = maxout

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=self.input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape,
                               border_mode='full')
        mid = (int(numpy.floor(filter_shape[2] / 2.)),
               int(numpy.floor(filter_shape[3] / 2.)))
        conv_out = conv_out[:, :, mid[0]:-mid[0], mid[1]:-mid[1]]

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        if activation == 'tanh':
            activate_out = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        elif activation == 'relu':
            activate_out = T.maximum(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'), 0.0)
        else:
            assert NotImplementedError("unown activation")

        # in the current implementation,
        # another normal conv layer must follow a maxout-conv layer
        n_filter_out = filter_shape[0]
        if maxout != 0:
            n_filter_out = n_filter_out / maxout
            assert filter_shape[0] % maxout == 0
            maxout_out = activate_out[:, ::maxout]
            for i in range(1, maxout):
                maxout_out = T.maximum(maxout_out, activate_out[:, i::maxout])
            activate_out = maxout_out

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=activate_out,
                                            ds=poolsize, ignore_border=True)

        if norm == 'mean':
            output_img_size = (image_shape[0], n_filter_out,
                               image_shape[2] / poolsize[0],
                               image_shape[3] / poolsize[1])

            # mean substraction normalization, with representation size fixed
            filter_size = 3
            filter_shape = (1, 1, filter_size, filter_size)
            filters = mean_filter(filter_size).reshape(filter_shape)
            filters = theano.shared(numpy.asarray(filters,
                                                  dtype=theano.config.floatX),
                                    borrow=True)
            pooled_out = pooled_out.reshape((output_img_size[0] * output_img_size[1],
                                             1,
                                             output_img_size[2], output_img_size[3]))
            mean = conv.conv2d(pooled_out, filters=filters,
                               filter_shape=filter_shape, border_mode='full')
            mid = int(numpy.floor(filter_size / 2.))

            output = pooled_out - mean[:, :, mid : -mid, mid : -mid]
            self.output = output.reshape(output_img_size)
        else:
            self.output = pooled_out

        # store parameters of this layer
        self.params = [self.W, self.b]

    def get_params(self):
        return {'W': self.W.get_value(borrow=True),
                'b': self.b.get_value(borrow=True),
                'pool_size': self.pool_size,
                'norm': self.norm,
                'maxout': self.maxout}

    def save_params_mat(self, basename):
        """ save params in .mat format
            file name will be built by adding suffix to 'basename'
        """
        params = self.get_params()
        sio.savemat(basename + '.mat', params)
