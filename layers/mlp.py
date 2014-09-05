"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time
from itertools import izip, count, chain
import numpy

import theano
import theano.tensor as T
import scipy.io as sio


from logistic_sgd import LogisticRegression, load_data


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

    def get_params(self):
        return {'W': self.W.get_value(borrow=True),
                'b': self.b.get_value(borrow=True)}

    def save_params_mat(self, basename):
        """ save params in .mat format
            file name will be built by adding suffix to 'basename'
        """
        params = self.get_params()
        sio.savemat(basename + '.mat', params)


def _dropout_from_tensor(rng, input, p):
    """ p is the dropout probability
    """
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=1-p, size=input.shape)

    # The cast is important because int * float32 = float64 which pulls things off the gpu
    output = input * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input,
                 n_in, n_out,
                 activation,
                 dropout_rate,
                 W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation)

        self.output = _dropout_from_tensor(rng, self.output, p=dropout_rate)

class DropoutMLP(object):
    """ A Multilayer Perceptron with dropout support """
    def __init__(self, rng, input,
                 n_in, layer_sizes,
                 dropout_rate,
                 activation=T.tanh):
        """ layer_sizes: list of int
            dropout_rate: float
            activation is default to tanh
        """
        # with mlp, training and testing will use different params!
        self.dropout_layers = []        # layers for train
        self.layers = []                # layers for test
        self.layer_sizes = layer_sizes
        self.n_in = n_in
        self.dropout_rate = dropout_rate

        next_input = input
        next_dropout_input = _dropout_from_tensor(rng, input, p=dropout_rate)
        layer_sizes.insert(0, n_in)

        for n_in, n_out in izip(layer_sizes, layer_sizes[1:]):
            next_dropout_layer = DropoutHiddenLayer(rng,
                                                    next_dropout_input,
                                                    n_in, n_out,
                                                    activation,
                                                    dropout_rate)
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_input = next_dropout_layer.output

            # for test, layer needs to reuse params in the dropout with a scaling
            next_layer = HiddenLayer(rng, next_input,
                                     n_in, n_out,
                                     W=next_dropout_layer.W * (1 - dropout_rate),
                                     b=next_dropout_layer.b,
                                     activation=activation)
            self.layers.append(next_layer)
            next_input = next_layer.output

        # provide two kinds of output for next layer to use
        self.dropout_output = next_dropout_input
        self.output = next_input

        self.params = list(chain.from_iterable(
            [x.params for x in self.dropout_layers]))

    def get_params(self):
        ret = {'Ws': [x.W.get_value(borrow=True) for x in self.dropout_layers],
               'bs': [x.b.get_value(borrow=True) for x in self.dropout_layers],
               'dropout_rate': self.dropout_rate,
               'layer_sizes': self.layer_sizes
              }
        return ret

    def set_params(self, Ws, bs):
        assert len(Ws) == len(self.dropout_layers)
        for k, W, b in izip(count(), Ws, bs):
            self.dropout_layers[k].W.set_value(W.astype(theano.config.floatX))
            self.dropout_layers[k].b.set_value(b.astype(theano.config.floatX))

    def save_params_mat(self, basename):
        """ save params in .mat format
            file name will be built by adding suffix to 'basename'
        """
        return
        params = self.get_params()
        sio.savemat(basename + '-hidden1.mat', params)
