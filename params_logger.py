#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: params_logger.py
# Date: Fri Aug 29 22:14:41 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
import cPickle as pickle
import gzip
import scipy.io as sio
import os
from itertools import count, izip

from layers.layers import name_dict

class ParamsLogger(object):
    """ class to save network params in each epoch"""

    def __init__(self, rgb_input_shape, logdir='logs'):
        """ logdir: a directory to save params to """
        self.rgb_input_shape = rgb_input_shape
        self.logdir = logdir
        try:
            os.mkdir(self.logdir)
        except:
            pass

    def save_params(self, epoch, layers):
        """ epoch: can be a int, or a string, will be used in the filename
        """
        if not isinstance(epoch, basestring):
            epoch = "{0:03d}".format(epoch)
        fname = os.path.join(self.logdir, "param{0}.pkl.gz".format(epoch))
        res = {}

        for layer, cnt in izip(layers, count()):
            # save layer type
            dic = {'type': name_dict[type(layer)] }
            # save other layer parameters
            dic.update(layer.get_params())

            res['layer' + str(cnt)] = dic
        res['input_shape'] = self.rgb_input_shape

        with gzip.open(fname, 'wb') as f:
            pickle.dump(res, f, -1)

def plot_filters(obj):
    for layer_params in obj:
        if layer_params['type'] == 'convpool':
            W = np.asarray(layer_params['W'])
            print W.shape
            filter_cnt = W.shape[0]
            n_img = W.shape[1]
            filter_size = W.shape[2]
            W = W.reshape((filter_cnt * n_img, filter_size, filter_size))
            tot_filter = filter_cnt * n_img

            # too many filters won't look good in the graph
            if len(W) > 100:
                W = W[:100]
                tot_filter = 100

            plt_num = np.ceil(np.sqrt(tot_filter))
            fig = plt.figure()
            for (idx, filt) in enumerate(W):
                if idx % 50 == 0:
                    print "Progress:", idx
                ax = fig.add_subplot(plt_num, plt_num, idx)
                plt.imshow(filt, interpolation='nearest', cmap=cm.Greys_r)
            plt.show()

def plot_filters_in_epoch(epoch, logdir='logs'):
    obj = json.loads(open(os.path.join(logdir, str(epoch) + '.json')).read())
    plot_filters(obj)

def plot_filters_with_file(filename):
    obj = json.loads(open(filename).read())
    plot_filters(obj)

if __name__ == '__main__':
    plot_filters_in_epoch(39)
