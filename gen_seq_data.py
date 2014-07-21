#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gen_seq_data.py
# Date: Mon Jul 21 00:36:54 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from scipy import stats
import numpy as np
import dataio
from IPython.core.debugger import Tracer
import cPickle as pickle
import gzip

class SeqDataGenerator(object):

    def __init__(self, len_dist, dataset):
        """ len_dist: a dict containing the distribution of length.
        """
        lens = len_dist.keys()
        self.max_len = max(lens)
        probs = [len_dist[k] for k in lens]
        assert sum(probs) == 1.0

        self.len_rvg = stats.rv_discrete(values=(lens, probs))
        # merge train/valid/test
        self.dataset = np.concatenate((dataset[0][0], dataset[1][0],
                                       dataset[2][0])), np.concatenate((dataset[0][1],
                                                                        dataset[1][1], dataset[2][1]))
        self.dataset_len = len(self.dataset[0])

        self.orig_image_shape = np.sqrt(self.dataset[0][0].shape[0])
        self.img_size = (self.orig_image_shape,
                         self.orig_image_shape * self.max_len)
        print "Dataset total size: {0}".format(self.dataset_len)
        print "Image size: {0}".format(self.img_size)

    def gen_n_samples(self, n):
        lens = self.len_rvg.rvs(size=n)

        rets = []
        labels = []
        for l in lens:
            img, label = self.paste_n_images(l)
            rets.append(img)
            labels.append(label)
        return rets, labels

    def paste_n_images(self, n):
        index = np.random.choice(self.dataset_len, n)
        imgs = self.dataset[0][index]
        imgs = [k.reshape(self.orig_image_shape, self.orig_image_shape) for k in imgs]
        labels = self.dataset[1][index]

        img = np.concatenate(imgs, axis = 1)

        if img.shape == self.img_size:
            return img, labels
        else:
            raise NotImplementedError

    def write_dataset(self, n_train, n_valid, n_test, fname):
        train = self.gen_n_samples(n_train)
        valid = self.gen_n_samples(n_valid)
        test = self.gen_n_samples(n_test)
        dataset = (train, valid, test)

        fout = gzip.open(fname, 'wb')
        pickle.dump(dataset, fout, -1)
        fout.close()

if __name__ == '__main__':
    dataset = dataio.read_data('./mnist.pkl.gz')
    generator = SeqDataGenerator({2: 1.0}, dataset)

    generator.write_dataset(50000, 10000, 10000, 'digits2.pkl.gz')




