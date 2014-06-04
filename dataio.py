#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: dataio.py
# Date: Wed Jun 04 15:00:32 2014 +0000
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import gzip
import cPickle as pickle
#import pickle
import tables
import numpy as np
import os

def read_data_fallback(dataset):
    def get(name):
        f = gzip.open(os.path.join(dataset, '{0}.pkl.gz'.format(name)), 'rb')
        data = pickle.load(f)
        f.close()
        return data
    return (get('train'), get('valid'), get('test'))

def read_data(dataset):
    """ return (train, valid, test)"""
    print ' ... loading data from {0}'.format(dataset)
    if dataset.endswith('.pkl.gz'):
        f = gzip.open(dataset, 'rb')
        train, valid, test = pickle.load(f)
        f.close()
        return (train, valid, test)

    if os.path.isdir(dataset):
        return read_data_fallback(dataset)
    assert False, "Invalid Dataset Filename"

def save_data_fallback(data, basename):
    dirname = basename + '.dump'
    os.mkdir(dirname)
    for idx, name in enumerate(['train', 'valid', 'test']):
        fout = gzip.open(os.path.join(dirname, '{0}.pkl.gz'.format(name)), 'wb')
        pickle.dump(data[idx], fout, -1)
        fout.close()


def save_data(data, basename):
    print 'Writing data to {0}'.format(basename)
    try:
        # first try pickle
        fout = gzip.open(basename + '.pkl.gz', 'wb')
        pickle.dump(data, fout, -1)
        fout.close()
    except:
        print "Pickle failed !"
        save_data_fallback(data, basename)
