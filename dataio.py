#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: dataio.py
# Date: Mon Jul 21 02:24:03 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import gzip
import cPickle as pickle
import operator
#from IPython.core.debugger import Tracer
import sys
from itertools import izip, count
import glob
import numpy as np
import os
import scipy.io as sio

def _read_data_fallback(dataset):
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
    if os.path.isfile(dataset):
        f = gzip.open(dataset, 'rb')
        train, valid, test = pickle.load(f)
        f.close()
        return (train, valid, test)

    if os.path.isdir(dataset):
        return _read_data_fallback(dataset)
    assert False, "Invalid Dataset Filename"

def _save_data_fallback(data, basename):
    dirname = basename + ".pkl.gz"
    try:
        os.mkdir(dirname)
    except:
        pass

    def save(dataset, name):
        size = reduce(operator.mul, dataset[0].shape)
        nslice = np.ceil(size / (2.5 * (10 ** 8)))
        print nslice
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
    """ param data is (train, valid, test)
        basename doesn't contain .pkl.gz suffix
    """
    print 'Writing data to {0}'.format(basename)
    output = basename + '.pkl.gz'
    assert not os.path.exists(output), "Path exists! " + str(output)

    try:
        # first try pickle
        fout = gzip.open(output, 'wb')
        pickle.dump(data, fout, -1)
        fout.close()
    except:
        print "Pickle failed ! Split the data!"
        os.remove(output)
        _save_data_fallback(data, basename)

def get_dataset_imgsize(dataset):
    train = read_data(dataset)[0]
    shape = train[0][0].shape
    if len(shape) == 1:
        size = int(np.sqrt(shape[0]))
        return size
    else:
        print "Not Square!"
        raise NotImplementedError

def sample_dataset(imgs, labels, cnt):
    """ sample `cnt' images from the dataset (imgs, labels)"""
    assert cnt < len(imgs)
    assert len(imgs) == len(labels)
    idxs = random.sample(range(len(imgs)), cnt)
    imgs = imgs[idxs]
    labels = labels[idxs]
    return (imgs, labels)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        dataset = sys.argv[1]
    else:
        dataset = './mnist.pkl.gz'

    t, v, ts = read_data(dataset)
    print len(t[0]), len(v[0]), len(ts[0])
    #print "Saving..."
    #_save_data_fallback((t, v, ts), 'testdir')

    #tt, vv, ttss = _read_data_fallback('testdir')
    #print tt[1] == t[1]
