#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gen_seq_data.py
# Date: Sat Jul 26 08:57:48 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from scipy import stats
from scipy.misc import imrotate
import numpy as np
import dataio
from IPython.core.debugger import Tracer
import cPickle as pickle
import gzip
import sys
from itertools import izip

def random_slice(k, N):
    """ randomly return k integers which sum to N"""
    seeds = np.random.random_sample((k, ))
    s = sum(seeds)
    seeds = [x / s * N for x in seeds]
    seeds = np.round(seeds)
    s = sum(seeds)
    if s < N:
        seeds[-1] += N - s
    elif s > N:
        for idx, t in enumerate(seeds):
            if t > s - N:
                seeds[idx] -= s - N
                break
        else:
            return random_slice(k, N)
    seeds = map(int, seeds)
    assert sum(seeds) == N and len(seeds) == k, "{0}, {1}".format(seeds, sum(seeds))
    assert all([lambda x: x >= 0, seeds])
    return seeds

def random_rotate(imgs):
    angles = np.random.randint(-20, 21, (len(imgs), ))
    imgs = [imrotate(img, ang) for img, ang in izip(imgs, angles)]
    return imgs

class SeqDataGenerator(object):

    def __init__(self, len_dist, dataset, max_width=None):
        """ len_dist: a dict containing the distribution of length.
        """
        lens = len_dist.keys()
        self.max_len = max(lens)
        probs = [len_dist[k] for k in lens]
        assert sum(probs) == 1.0

        self.len_rvg = stats.rv_discrete(values=(lens, probs))
        # merge train/valid/test
        #self.dataset = np.concatenate((dataset[0][0], dataset[1][0],
                                       #dataset[2][0])), np.concatenate((dataset[0][1],
                                                                        #dataset[1][1], dataset[2][1]))
        #self.dataset_len = len(self.dataset[0])
        self.dataset = dataset

        self.orig_image_shape = int(np.sqrt(self.dataset[0][0][0].shape[0]))
        if max_width is None:
            max_width = self.orig_image_shape * self.max_len
        self.img_size = (self.orig_image_shape, max_width)
        print "Original dataset size: {0}, {1}, {2}".format(len(dataset[0][0]),
                                                         len(dataset[1][0]),
                                                         len(dataset[2][0]))
        print "Image size: {0}".format(self.img_size)

    def gen_n_samples(self, n, dataset):
        lens = self.len_rvg.rvs(size=n)

        rets = []
        labels = []
        for l in lens:
            img, label = self.select_n_images(l, dataset)
            rets.append(img)
            labels.append(label)
        return rets, labels

    def select_n_images(self, n, dataset):
        index = np.random.choice(len(dataset[0]), n)
        imgs = dataset[0][index]
        imgs = [k.reshape(self.orig_image_shape, self.orig_image_shape) for k in imgs]
        labels = dataset[1][index]

        img = np.concatenate(imgs, axis = 1)

        if img.shape == self.img_size:
            return img, labels
        else:
            paste = self.paste_image(imgs)
            return paste, labels

    def paste_image(self, imgs):
        assert self.img_size[0] == imgs[0].shape[0]
        height = self.img_size[0]

        imgs = random_rotate(imgs)

        n_chunks = len(imgs) + 1
        space_left = self.img_size[1] - len(imgs) * imgs[0].shape[1]
        assert space_left > 0
        chunks = random_slice(n_chunks, space_left)

        ret = np.zeros((height, chunks[0]))
        for idx, k in enumerate(imgs):
            ret = np.hstack((ret, k, np.zeros((height, chunks[idx + 1]))))
        assert ret.shape == self.img_size
        return ret

    def write_dataset(self, n_train, n_valid, n_test, fname):
        train = self.gen_n_samples(n_train, self.dataset[0])
        valid = self.gen_n_samples(n_valid, self.dataset[1])
        test = self.gen_n_samples(n_test, self.dataset[2])
        dataset = (train, valid, test)
        print "Writing to {0}...".format(fname)

        fout = gzip.open(fname, 'wb')
        pickle.dump(dataset, fout, -1)
        fout.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} <output.pkl.gz> <sequence length>"
        sys.exit()
    dataset = dataio.read_data('./data/mnist.pkl.gz')
    fout = sys.argv[1]
    seq_len = int(sys.argv[2])

    generator = SeqDataGenerator({seq_len: 1.0}, dataset, max_width = 100)

    generator.write_dataset(80000, 15000, 15000, fout)




