#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: params_logger.py
# Date: Fri Jun 06 04:31:41 2014 +0000
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm

import json
from json import encoder
import scipy.io as sio
encoder.FLOAT_REPR = lambda o: format(o, '.7f')
import os

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer

name_dict = {LeNetConvPoolLayer: 'convpool',
             HiddenLayer: 'hidden',
             LogisticRegression: 'lr'}

class ParamsLogger(object):

    def __init__(self, logdir='logs'):
        self.logdir = logdir
        try:
            os.mkdir(self.logdir)
        except:
            pass

    def save_params(self, epoch, layers, layer_config):
        fname = os.path.join(self.logdir, "{0}.mat".format(epoch))
        res = {}

        layer_params = [x.params for x in layers]
        cnt = 0
        for layer, pair, config in zip(layers, layer_params, layer_config):
            cnt += 1
            W = pair[0].get_value()
            W = W.tolist()
            b = pair[1].get_value().tolist()
            dic = {'type': name_dict[type(layer)],
                   'W': W,
                   'b': b}

            # extra config info to save
            if type(layer) == LeNetConvPoolLayer:
                dic['pool_size'] = config['pool_size']
            res['layer' + str(cnt)] = dic

        sio.savemat(fname, res)
        #with open(fname, 'w') as fout:
            #json.dump(res, fout)


def plot_filters(obj):
    for layer_params in obj:
        if layer_params['type'] == name_dict[LeNetConvPoolLayer]:
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
