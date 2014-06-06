#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: dataio.py
# Date: Fri Jun 06 05:24:07 2014 +0000
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import gzip
import cPickle as pickle
#import pickle
import tables
import numpy as np
import os
import scipy.io as sio

def read_data_fallback(dataset):
    f = gzip.open(dataset, 'rb')
    mat = sio.loadmat(f)
    f.close()
    data = mat['data']
    train, valid, test = data[0], data[1], data[2]
    def transform(d):
        d[1] = d[1][0]
        return d
    return transform(train), transform(valid), transform(test)

def read_data(dataset):
    """ return (train, valid, test)"""
    print ' ... loading data from {0}'.format(dataset)
    if dataset.endswith('.pkl.gz'):
        f = gzip.open(dataset, 'rb')
        train, valid, test = pickle.load(f)
        f.close()
        return (train, valid, test)

    if dataset.endswith('.mat.gz'):
        return read_data_fallback(dataset)
    assert False, "Invalid Dataset Filename"

def save_data_fallback(data, basename):
    fname = basename + '.mat.gz'
    f = gzip.open(fname, 'wb')
    sio.savemat(f, {'data' : data})
    f.close()

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

if __name__ == '__main__':
    t, v, ts = read_data('./mnist.pkl.gz')
    save_data_fallback((t, v, ts), 'testdir')

    tt, vv, ttss = read_data_fallback('testdir.mat.gz')
    print tt[1] == t[1]
