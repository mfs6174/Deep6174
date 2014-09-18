#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: shared_dataio.py
# Date: Thu Sep 18 10:32:40 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from dataio import read_data
import theano
import numpy as np
from IPython.core.debugger import Tracer
import theano.tensor as T
from itertools import chain

class SharedDataIO(object):
    """ provide api to generate theano shared variable of datasets
    """

    def __init__(self, dataset_filename, share_all, trainer):
        """ share_all: if True, will build shared variable for the whole
            dataset, this is likely to fail when running a large dataset on gpu.
        """
        self.share_all = share_all
        self.batch_size = trainer.batch_size
        self.max_len = trainer.max_len
        self.multi_output = trainer.multi_output
        self.filename = dataset_filename

        if self.share_all:
            self.dataset = read_data(dataset_filename)
            self.shared_dataset = [self.share_dataset(k) for k in
                                   self.dataset]
        else:
            n_in = np.prod(trainer.input_shape[1:])
            self.shared_Xs = [theano.shared(np.zeros((self.batch_size, n_in),
                                                    dtype=theano.config.floatX),
                                           borrow=True) for _ in range(3)]
            if not self.multi_output:
                # numpy.int label
                self.shared_ys = [theano.shared(np.zeros((self.batch_size, ),
                                                        dtype='int32')) for _ in range(3)]
            else:
                assert self.max_len != 0
                # TODO currently FixedLengthLayer is not supported
                seq_len = self.max_len + 1
                self.shared_ys = [theano.shared(np.zeros((self.batch_size,
                                                          seq_len),
                                                         dtype='int32')) for _ in range(3)]

    def get_dataset_size(self):
        """ return a tuple (l1, l2, l3)"""
        return (len(self.dataset[0][0]), len(self.dataset[1][0]),
                len(self.dataset[2][0]))

    def _get_with_batch_index(self, dataset, index):
        """ dataset is 0, 1 or 2 indicating train, valid, test
            index is the batch index
            will set the shared variables to correct value
            return a tuple (X, y) of shared variables
        """
        assert self.share_all == False

        data_x, data_y = self.dataset[dataset]
        data_x = data_x[index * self.batch_size:
                                   (index + 1) * self.batch_size]
        data_y = data_y[index * self.batch_size: (index + 1) * self.batch_size]
        data_x, data_y = self.process_pair(data_x, data_y)
        self.shared_Xs[dataset].set_value(data_x, borrow=True)
        self.shared_ys[dataset].set_value(data_y, borrow=True)
        return (self.shared_Xs[dataset], self.shared_ys[dataset])

    def process_pair(self, X, y):
        if type(X) == list:
            X = np.asarray(X, dtype='float32')
        if len(X[0].shape) != 1:
            X = X.reshape(X.shape[0], -1)       # flatten each image
        if self.max_len > 0:
            y = [list(chain.from_iterable((
                [len(k) - 1],
                k,
                [-1] * (self.max_len - len(k))))) for k in y]
            for k in y:
                assert len(k) == self.max_len + 1
                assert k[0] + 2 <= len(k)
        return (X, np.asarray(y, dtype='int32'))

    def share_dataset(self, data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = self.process_pair(*data_xy)
        shared_x = theano.shared(data_x, borrow=borrow)
        shared_y = theano.shared(data_y, borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    def get_train(self, index):
        return self._get_with_batch_index(0, index)
    def get_valid(self, index):
        return self._get_with_batch_index(1, index)
    def get_test(self, index):
        return self._get_with_batch_index(2, index)

    def read_delay(self):
        """ Read after initialization.
        Will save memory for trainer to compile"""
        assert self.share_all == False
        self.dataset = read_data(self.filename)

if __name__ == '__main__':
    dataset = read_data('./mnist.pkl.gz')

    io = SharedDataIO(dataset, 180, True)
    print io.get_dataset_size()
