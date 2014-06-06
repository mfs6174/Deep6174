#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: dataio.py
# Date: Fri Jun 06 15:16:01 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import gzip
import cPickle as pickle
#import pickle
import tables
from IPython.core.debugger import Tracer
from itertools import izip, count
import glob
import numpy as np
import os
import scipy.io as sio

def read_data_fallback(dataset):
    def read(name):
        pat = '{0}/{1}-*.pkl.gz'.format(dataset, name)
        all_imgs = []
        all_labels = []
        for f in sorted(glob.glob(pat)):
            fin = gzip.open(f, 'rb')
            imgs, labels = pickle.load(fin)
            if not len(all_imgs):
                all_imgs = np.vstack([imgs])
                all_labels = labels
            else:
                all_imgs = np.vstack([all_imgs, imgs])
                all_labels = np.concatenate((all_labels, labels))
            fin.close()
        return (all_imgs, all_labels)
    return (read('train'), read('valid'), read('test'))

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
    dirname = basename
    try:
        os.mkdir(basename)
    except:
        pass

    nslice = 5
    def save(dataset, name):
        imgs = np.array_split(dataset[0], nslice)
        labels = np.array_split(dataset[1], nslice)
        for idx, img_slice, label_slice in izip(count(), imgs, labels):
            to_save = (img_slice, label_slice)
            fname = "{0}-{1}.pkl.gz".format(name, idx)
            fout = gzip.open(os.path.join(dirname, fname), 'wb')
            pickle.dump(to_save, fout, -1)
            fout.close()

    for idx, name in enumerate(['train', 'valid', 'test']):
        dataset = data[idx]
        save(dataset, name)

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
    print "Saving..."
    save_data_fallback((t, v, ts), 'testdir')

    tt, vv, ttss = read_data_fallback('testdir')
    print tt[1] == t[1]
