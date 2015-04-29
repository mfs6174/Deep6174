#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: shared_dataio.py
# Date: Thu Sep 18 10:32:40 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Author: Xi-Jin Zhang <zhangxijin91@gmail.com>

from dataio import read_data
import dataio
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

class SharedImagesIO(SharedDataIO):
    """ provide api to generate theano shared variable of datasets (with image patches)
    """

    def __init__(self, dataset_base_path, load_all, trainer, stride, pre_proc):
        """ share_all: if True, will build shared variable for the whole
            dataset, this is likely to fail when running a large dataset on gpu.
        """
        self.load_all = load_all
        self.batch_size = trainer.batch_size
        self.multi_output = trainer.multi_output
        self.patch_output = trainer.patch_output
        self.basepath = dataset_base_path
        #deal with pre_proc below
        if pre_proc is None:
            self.need_proc = False
            self.pre_proc ={}
        else:
            self.need_proc = pre_proc.get("proc",False)
            self.pre_proc = pre_proc
        self.proc_y = self.pre_proc.get('proc_y',False)
        self.full_proc = self.pre_proc.get('full_proc',False)
        #self.meanvar = self.pre_proc.get('meanvar',False)
        #self.localval = self.pre_proc.get('localvar',False)
        
        self.patch_size = trainer.input_shape[1:]
        assert (self.patch_output if self.proc_y  else True)
        n_in = np.prod(self.patch_size)
        if type(stride)==int:
            self.stride = (stride,stride)
        else:
            self.stride = stride
        self.shared_Xs = [theano.shared(np.zeros((self.batch_size, n_in),
                                                 dtype=theano.config.floatX),
                                        borrow=True) for _ in range(3)]
        if not self.patch_output:
            # numpy.int label
            raise NotImplementedError()
            """self.shared_ys = [theano.shared(np.zeros((self.batch_size, ),
                                                     dtype='int32')) for _ in range(3)]
            """
        else:
            assert self.stride[0]<=self.patch_size[1] and self.stride[1]<=self.patch_size[2]
            self.shared_ys = [theano.shared(np.zeros((self.batch_size,
                                                      np.prod(stride)),
                                                     dtype='int32')) for _ in range(3)]
        self.data_path= tuple((self.basepath+"/"+i+'/' for i in ["train","valid","test"]))
        self.image_list = dataio.get_image_list(self.data_path)
        self.image_num = tuple( (len(i) for i in self.image_list) )
        self.img_idx_now = [0,0,0]
        self.dataset = [self._read_with_index(i,0) for i in range(3)]
        self.image_size= self.dataset[0][0].shape
        self.patch_num_2d = tuple( ( (self.image_size[1+i]-self.patch_size[1+i])/self.stride[i]+1 for i in range(2) ) )
        self.patch_num = np.prod(self.patch_num_2d)
        print 'patch per image'
        print self.patch_num
        assert self.patch_num%self.batch_size == 0
        self.batch_per_image = self.patch_num / self.batch_size
        self.data_size = tuple( (self.image_num[i]*self.patch_num for i in range(3) ) )
        
    def get_dataset_size(self):
        """ return a tuple (l1, l2, l3)"""
        return self.data_size

    def _read_with_index(self,dataset,imidx):
        return dataio.read_raw_image_label(self.data_path[dataset],self.image_list[dataset][imidx])

    def _get_with_batch_index(self, dataset, index):
        """ dataset is 0, 1 or 2 indicating train, valid, test
            index is the batch index
            will set the shared variables to correct value
            return a tuple (X, y) of shared variables
        """
        imgidx = index/self.batch_per_image
        if self.load_all:
            datasetNow = self.dataset[dataset][imgidx]
        else:
            if not imgidx == self.img_idx_now[dataset]:
                self.img_idx_now[dataset] = imgidx
                self.dataset[dataset] = self._read_with_index(dataset,imgidx)
            datasetNow = self.dataset[dataset]
        data_x=np.ndarray((self.batch_size,self.patch_size[0],self.patch_size[1],self.patch_size[2]),dtype=theano.config.floatX)
        if self.patch_output:
            data_y=np.ndarray((self.batch_size,self.stride[0],self.stride[1]),dtype=theano.config.floatX)
        else:
            raise NotImplementedError()

        if self.need_proc:
            raise NotImplementedError()

        insideIdx = index-imgidx*self.batch_per_image
        if (not self.need_proc) or (dataset != 0):
            for i in range(self.batch_size):
                j=i+insideIdx
                data_x[i,:,:,:] = datasetNow[0][:,j/self.patch_num_2d[1]*self.stride[0]:j/self.patch_num_2d[1]*self.stride[0]+self.patch_size[1],
                                                j%self.patch_num_2d[1]*self.stride[1]:j%self.patch_num_2d[1]*self.stride[1]+self.patch_size[2]]
        else:
            if self.full_proc:
                pass
            else:
                raise NotImplementedError()

        offset = ((self.patch_size[1]-self.stride[0])/2,(self.patch_size[2]-self.stride[1])/2)
        if (not self.proc_y) or (dataset !=0):
            if self.patch_output:
                for i in range(self.batch_size):
                    j=i+insideIdx
                    data_y[i,:,:] = datasetNow[1][j/self.patch_num_2d[1]*self.stride[0]+offset[0]:j/self.patch_num_2d[1]*self.stride[0]+self.patch_size[1]-offset[0],
                                                j%self.patch_num_2d[1]*self.stride[1]+offset[1]:j%self.patch_num_2d[1]*self.stride[1]+self.patch_size[2]-offset[1]]
            else:
                pass
        else:
            if self.full_proc:
                pass
            else:
                raise NotImplementedError()
        
        data_x, data_y = self.process_pair(data_x, data_y)
        self.shared_Xs[dataset].set_value(data_x, borrow=True)
        self.shared_ys[dataset].set_value(data_y, borrow=True)
        return (self.shared_Xs[dataset], self.shared_ys[dataset])

    def process_pair(self, X, y):
        if type(X) == list:
            X = np.asarray(X, dtype='float32')
        if type(y) == list:
            y = np.asarray(y, dtype='int32')
        if len(X[0].shape) != 1:
            X = X.reshape(X.shape[0], -1)       # flatten each image
        if len(y[0].shape) != 1:
            y = y.reshape(y.shape[0], -1)       # flatten each image
        return (X, np.asarray(y, dtype='int32'))

    def share_dataset(self, data_xy, borrow=True):
        raise NotImplementedError()
    
    def get_train(self, index):
        return self._get_with_batch_index(0, index)
    def get_valid(self, index):
        return self._get_with_batch_index(1, index)
    def get_test(self, index):
        return self._get_with_batch_index(2, index)

    def read_delay(self):
        """ Read after initialization.
        Will save memory for trainer to compile"""
        if self.load_all:
            self.dataset = read_data(self.filename)
        else:
            pass


if __name__ == '__main__':
    dataset = read_data('./mnist.pkl.gz')

    io = SharedDataIO(dataset, 180, True)
    print io.get_dataset_size()
