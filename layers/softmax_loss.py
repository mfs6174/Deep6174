#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: LR.py
# Date: Sun Nov 30 22:43:30 2014 +0800
# Author: Xi-Jin Zhang <zhangxijin91@gmail.com>
import numpy
import theano
import theano.tensor as T
import theano.printing as PP

from common import Layer

class SoftmaxLoss(Layer):
    """Softmax Loss Class

    Computes the multinomial logistic loss for a one-of-many classification task, 
    passing real-valued predictions through a softmax to get a probability distribution over classes.
    support one patch per sample mode (instead of one label per sample) using s_out
    """

    NAME = 'sl'

    def __init__(self, input_train, input_test,
                 input_shape, n_out,s_out):
        """ Initialize the parameters of the logistic regression

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        super(SoftmaxLoss, self).__init__(None, input_train, input_test)
        self.input_shape = input_shape
        n_in = numpy.prod(self.input_shape[1:])
        self.total_out = n_out*s_out*s_out
        assert n_in==self.total_out, 'n_in != n_out, softmax can not be apply directly'
        self.n_out = n_out
        self.s_out = s_out
        #we do not use W and b for direct softmax
        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.random.normal(0.0, 0.02,
                                                         (n_in, n_out)).astype(theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)
        """
        assert self.n_out*self.s_out**2 == n_in, 'self.n_out*self.s_out**2 != n_in'
        self.input_test=self.input_test.reshape((self.input_shape[0],self.n_out,self.s_out,self.s_out) )
        self.input_test=self.input_test.dimshuffle((1,0,2,3) ).flatten(2).dimshuffle((1,0))
        
        # compute vector of class-membership probabilities in symbolic form
        #self.p_y_given_x = T.nnet.softmax(T.dot(self.input_test, self.W) + self.b)
        #we do not use W and b for direct softmax
        self.p_y_given_x = T.nnet.softmax(self.input_test)
        

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        if self.has_dropout_input:
            self.input_train=self.input_train.reshape((self.input_shape[0],self.n_out,self.s_out,self.s_out) )
            self.input_train=self.input_train.dimshuffle((1,0,2,3) ).flatten(2).dimshuffle((1,0))
            #we do not use W and b for direct softmax
            #self.p_y_given_x = T.nnet.softmax(T.dot(self.input_train, self.W) + self.b)
            self.p_y_given_x = T.nnet.softmax(self.input_train)

        # parameters of the model
        #self.params = [self.W, self.b]
        #we do not have params in direct softmax

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.

        ty = y.flatten()
        return -T.mean(T.log(self.p_y_given_x)[T.arange(ty.shape[0]), ty])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        ty=y.flatten()
        if ty.ndim != self.y_pred.ndim:
            raise TypeError('ty should have the same shape as self.y_pred',
                ('ty', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if ty.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, ty))
        else:
            raise NotImplementedError()

    def get_params(self):
        return {'input_shape': self.input_shape,
                'n_out': self.n_out,
                's_out': self.s_out}

    @staticmethod
    def build_layer_from_params(params, rng, input_train, input_test=None):
        layer = SoftmaxLoss(input_train, input_test,
                            params['input_shape'],
                            params['n_out'],
                            params.get('s_out',1))
        #if 'W' in params:
        #    layer.W.set_value(params['W'].astype(theano.config.floatX))
        #    layer.b.set_value(params['b'].astype(theano.config.floatX))
        return layer
