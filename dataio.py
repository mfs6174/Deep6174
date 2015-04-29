#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: dataio.py
# Date: Thu Sep 18 10:23:38 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Author: Xi-Jin Zhang <zhangxijin91@gmail.com>

import gzip
import cPickle as pickle
import operator
#from IPython.core.debugger import Tracer
from itertools import izip, count
import glob
import numpy as np
import os, sys
import scipy.io as sio

import cv2
import theano as tn

from lib.utils import memorized

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

@memorized
def read_data(dataset):
    """ return (train, valid, test)"""
    print 'Loading data from {0} ...'.format(dataset)
    if os.path.isfile(dataset):
        f = gzip.open(dataset, 'rb')
        train, valid, test = pickle.load(f)
        f.close()
        return (train, valid, test)

    if os.path.isdir(dataset):
        return _read_data_fallback(dataset)
    print 'Data loaded.'
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

def get_dataset_imgsize(dataset, transform=True):
    train = read_data(dataset)[0]
    shape = train[0][0].shape
    if not transform:
        return shape
    if len(shape) == 1:
        size = int(np.sqrt(shape[0]))
        return (size, size)
    else:
        return shape

def sample_dataset(imgs, labels, cnt):
    """ sample `cnt' images from the dataset (imgs, labels)"""
    assert cnt < len(imgs)
    assert len(imgs) == len(labels)
    idxs = random.sample(range(len(imgs)), cnt)
    imgs = imgs[idxs]
    labels = labels[idxs]
    return (imgs, labels)

@memorized
def read_raw_image_label(ipath,image,label = None,multi = 0):
    """ return (image, label) (not flattened)"""
    #print 'Loading image and label from {0} ...'.format(image)
    if label is None:
        label=ipath+"label/"+image
    image = ipath+image
    if os.path.isfile(image) and os.path.isfile(label):
        im = cv2.imread(image).astype(tn.config.floatX)/255.0
        assert im is not None, "invalid image"
        lb = cv2.imread(label,multi)
        assert lb is not None, "invalid label"
        if (len(im.shape)==2):
            newim = im.reshape((1,im.shape[0],im.shape[1]))
        elif len(im.shape)==3:
            newim = np.ndarray((im.shape[2],im.shape[0],im.shape[1]))
            for i in range(0,im.shape[2]):newim[i,:,:]=im[:,:,i]
        else:
            assert False, "invalid image shape"
        if (len(lb.shape)==2):
            newlb = lb
        elif len(lb.shape)==3:
            newlb = np.ndarray((lb.shape[2],lb.shape[0],lb.shape[1]))
            for i in range(0,lb.shape[2]):newlb[i,:,:]=lb[:,:,i]
        else:
            assert False, "invalid label shape"
        #print 'Data loaded.'
        return (newim,newlb)
    assert False, "Invalid Dataset Filename"



image_type = [".tif",".jpg",".png",".bmp",".pgm"]

def list_images(dir):
    files = os.listdir(dir)
    images = []
    for f in files:
        name,ext=os.path.splitext(f)
        if ext in image_type:
            images.append(f)
    return images

def get_image_list(dir):
    ilist = [list_images(dir[i]) for i in range(3)]
    return ilist

@memorized
def read_image_label(dataset):
    """ return (image, label) (not flattened)"""
    #print 'Loading data from {0} ...'.format(dataset)
    if os.path.isfile(dataset):
        f = gzip.open(dataset, 'rb')
        im,lb = pickle.load(f)
        f.close()
        return (im,lb)

    #if os.path.isdir(dataset):
    #    return _read_data_fallback(dataset)
    #print 'Data loaded.'
    assert False, "Invalid Dataset Filename"


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

