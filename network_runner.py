#!/usr/bin/env python2
# -*- coding: UTF-8 -*-

import numpy as np
import scipy
import scipy.io as sio
from scipy.misc import imsave, toimage, imread
import theano.tensor as T
import theano
import sys, gzip
import cPickle as pickle
import operator
import itertools
from itertools import count, izip
import time
import dataio

from network_trainer import NNTrainer
from lib.imageutil import tile_raster_images, get_image_matrix
from layers.layers import *

N_OUT = 10

class NetworkRunner(object):
    def __init__(self, input_shape, multi_output, patch_output, stride, n_out=1):
        """ input size in (height, width)"""
        # nn is the underlying neural network object to run with
        self.nn = NNTrainer(input_shape, multi_output,patch_output,stride)
        self.multi_output = multi_output
        self.patch_output = patch_output
        self.stride = stride
        self.n_out = n_out
        
    def get_layer_by_index(self, idx):
        """ return the instance of certain layer.
            idx can be negative to get layers from the end
        """
        return self.nn.layers[idx]

    def set_last_updates(self, last_updates):
        """ set last_updates in trainer, for used in momentum
            last_updates: list of np array for each param
        """
        assert len(self.nn.last_updates) == 0
        for lu in last_updates:
            self.nn.last_updates.append(theano.shared(lu))

    def _set_var_output(self):
        """ check whether this network supports variant length sequence output"""
        if type(self.nn.layers[-1]) == SequenceSoftmax:
            self.var_len_output = True
        else:
            self.var_len_output = False

    def finish(self, only_last=True):
        """ compile the output of each layer as theano function"""
        print "Compiling..."
        self._set_var_output()
        self.funcs = []
        for (idx, layer) in enumerate(self.nn.layers):
            if idx != len(self.nn.layers) - 1 and only_last:
                continue
            if idx == len(self.nn.layers) - 1:
                # the output layer: use likelihood of the label
                f = theano.function([self.nn.x],
                                     layer.p_y_given_x,
                                    allow_input_downcast=True)
            else:
                # layers in the middle: use its output fed into the next layer
                f = theano.function([self.nn.x],
                                   layer.get_output_test(), allow_input_downcast=True)
            self.funcs.append(f)

    def _prepare_img_to_run(self, img):
        assert self.nn.batch_size == 1, \
                "batch_size of runner is not 1, but trying to run against 1 image"
        img = get_image_matrix(img, show=False)
        # shape could be (x, y) or (3, x, y)
        assert img.shape in [self.nn.input_shape[1:], self.nn.input_shape[2:]]
        return img.flatten()

    def run(self, img):
        """ return all the representations after each layer"""
        img = self._prepare_img_to_run(img)
        results = []
        for (idx, layer) in enumerate(self.nn.layers):
            # why [img]?
            # theano needs arguments to be listed, although there is only 1 argument here
            results.append(self.funcs[idx]([img]))
        return results

    def run_only_last(self, img):
        img = self._prepare_img_to_run(img)
        return self.funcs[-1]([img])

    def predict(self, img):
        """ return predicted label (either a list or a digit)"""
        results = [self.run_only_last(img)]
        label = NetworkRunner.get_label_from_result(img, results,
                                                    self.multi_output,
                                                    self.var_len_output)
        return label

    def patch_raw_predict(self,inputData):
        ''' inputData should be (b,c*x*y) return data is (b*s_out*s_out,n_out)'''
        inputData = inputData.reshape(inputData.shape[0], -1)       # flatten each image
        return self.run_only_last(inputData)

    def predict_whole_img(self,img_name):
        img = dataio.read_raw_image_only(img_name)
        image_size= img.shape
        if type(self.stride)==int:
            tstride = (self.stride,self.stride)
        else:
            tstride = stride
        patch_size = self.nn.input_shape[1:]
        patch_num_2d = tuple( ( (image_size[1+i]-patch_size[1+i])/tstride[i]+1 for i in range(2) ) )
        patch_num = np.prod(patch_num_2d)
        print 'patch per image'
        print patch_num
        if patch_num % self.nn.batch_size != 0:
            patch_num = patch_num/self.nn.batch_size*self.nn.batch_size
            print 'drop some data to fit batch_size'
            print patch_num
        assert patch_num%self.nn.batch_size == 0
        batch_per_image = patch_num / self.nn.batch_size
        data_x=np.ndarray((self.nn.batch_size,patch_size[0],patch_size[1],patch_size[2]),dtype=theano.config.floatX)
        retImage = np.ndarray((image_size[1],image_size[2],self.n_out),dtype=theano.config.floatX)
        for index in range(batch_per_image):
            insideIdx = index
            for i in range(self.nn.batch_size):
                j=i+insideIdx
                data_x[i,:,:,:] = img[:,j/patch_num_2d[1]*tstride[0]:j/patch_num_2d[1]*tstride[0]+patch_size[1],
                                                j%patch_num_2d[1]*tstride[1]:j%patch_num_2d[1]*tstride[1]+tpatch_size[2]]
            result = self.patch_raw_predict(data_x)
            result = result.reshape((self.nn.batch_size,tstride[0],tstride[1],self.n_out))
            offset = ((patch_size[1]-tstride[0])/2,(patch_size[2]-tstride[1])/2)
            for i in range(self.nn.batch_size):
                j=i+insideIdx
                retImage[j/patch_num_2d[1]*tstride[0]+offset[0]:j/patch_num_2d[1]*tstride[0]+patch_size[1]-offset[0],
                         j%patch_num_2d[1]*tstride[1]+offset[1]:j%patch_num_2d[1]*tstride[1]+patch_size[2]-offset[1],:]
                = data_y[i,:,:,:]
        return retImage


    @staticmethod
    def get_label_from_result(img, results, multi_output, var_len_output=True):
        """ parse the results and get label
            results: return value of run() or run_only_last()
        """
        if not multi_output:
            # the predicted results for single digit output
            return results[-1].argmax()
        else:
            # predicted results for multiple digit output
            ret = []
            for r in results[-1]:
                ret.append(r[0].argmax())
            if var_len_output:
                # the first element is 'length - 1', make it 'length'
                ret[0] += 1
            return ret

def get_nlayer_from_params(params):
    for nlayer in count():
        layername = 'layer' + str(nlayer)
        if layername not in params:
            return nlayer

def build_nn_with_params(params, batch_size):
    """ build a network and return it
        params: the object load from param{epoch}.pkl.gz file
    """
    input_size = params['layer0']['input_shape']
    if batch_size is None:
        batch_size = input_size[0]
    input_size = (batch_size,) + input_size[1:]
    print "Size={0}".format(input_size)

    nlayer = get_nlayer_from_params(params)
    last_layer = params['layer{0}'.format(nlayer - 1)]
    patch_output = False
    if last_layer['type'] in ['ssm']:
        multi_output = True
    elif last_layer['type'] in ['lr']:
        multi_output = False
    elif last_layer['type'] in ['sl']:
        multi_output = False
        patch_output = True
    else:
        assert False
    stride = last_layer.get('s_out',0)
    n_our = last_layer.get('n_out',1)
    runner = NetworkRunner(input_size, multi_output,patch_output,stride,n_out)

    if 'last_updates' in params:
        runner.set_last_updates(params['last_updates'])

    for idx in range(nlayer):
        layername = 'layer' + str(idx)
        layerdata = params[layername]
        typename = layerdata['type']
        if typename == 'convpool':
            typename = 'conv'
        layer_cls = name_cls_dict[typename]
        print "Layer ", idx, ' is ', layer_cls
        runner.nn.add_layer(layer_cls, layerdata)

    print "Model Loaded."
    return runner

def get_nn(filename, batch_size=1):
    """ get a network from a saved model file
        batch_size is None: will use same batch_size in the model
    """
    with gzip.open(filename, 'r') as f:
        data = pickle.load(f)

    nn = build_nn_with_params(data, batch_size)
    # compile all the functions
    nn.finish()
    nn.nn.print_config()
    return nn

#def save_LR_W_img(W, n_filter):
    #""" save W as images """
    #for l in range(N_OUT):
        #w = W[:,l]
        #size = int(np.sqrt(w.shape[0] / n_filter))
        #imgs = w.reshape(n_filter, size, size)
        #for idx, img in enumerate(imgs):
            #imsave('LRW-label{0}-weight{1}.jpg'.format(l, idx), img)

#def save_convolved_images(nn, results):
    #for nl in xrange(nn.n_conv_layer):
        #layer = results[nl][0]
        #img_shape = layer[0].shape
        #tile_len = int(np.ceil(np.sqrt(len(layer))))
        #tile_shape = (tile_len, int(np.ceil(len(layer) * 1.0 / tile_len)))
        #layer = layer.reshape((layer.shape[0], -1))
        #raster = tile_raster_images(layer, img_shape, tile_shape,
                                    #tile_spacing=(3, 3))
        #imsave('{0}.jpg'.format(nl), raster)

