#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: dataio.py
# Date: Wed Jun 04 20:27:55 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cPickle, gzip
import tables

def read_data(dataset):
    """ return (train, valid, test)"""
    print ' ... loading data from {0}'.format(dataset)
    if dataset.endswith('.pkl.gz'):
        f = gzip.open(dataset, 'rb')
        train, valid, test = cPickle.load(f)
        f.close()
        return (train, valid, test)

    if dataset.endswith('.h5'):
        f = tables.openFile(dataset)
        train, valid, test = f.root.data.read()
        f.close()
        return (train, valid, test)

    assert False, "Invalid Dataset Filename"

def save_data(data, basename):
    try:
# first try pickle
        fout = gzip.open(basename + '.pkl.gz', 'wb')
        cPickle.dump(data, fout, -1)
        fout.close()
    except:
        print "Pickle failed !"
# then try h5
        fout = tables.openFile(basename + '.h5', mode='w')
        f.createArray(f.root, 'data', data)
        f.close()
