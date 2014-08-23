#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gen-seq-data.py
# Date: Sat Aug 23 13:54:32 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from scipy import stats
from scipy.misc import imrotate, imresize
from scipy import ndimage
import numpy as np
import dataio
from IPython.core.debugger import Tracer
import cPickle as pickle
import gzip
import sys
from itertools import izip
from copy import copy

from imageutil import show_img_sync, get_image_matrix
from utils import timeit
from progress import Progressor

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
    # check return results
    assert sum(seeds) == N and len(seeds) == k, "{0}, {1}".format(seeds, sum(seeds))
    assert all([lambda x: x >= 0, seeds])
    return seeds

def random_rotate(imgs):
    """random rotate a list of images"""
    angles = np.random.randint(-20, 21, (len(imgs), ))
    imgs = [imrotate(img, ang) for img, ang in izip(imgs, angles)]
    return imgs

def random_resize(imgs, max_len, digit_shapes=None):
    """ random resize a list of images, and will resize digit_shapes
    correspondingly in place"""
    seeds = np.random.random_sample((len(imgs),))
    def resize(img, idx):
        assert img.shape[0] < max_len
        assert img.shape[0] == img.shape[1]
    # valid interval is [20, max_len]
        LEN_MIN = 20
        assert max_len > 20
        new_size = seeds[idx] * (max_len - 20) + 20
        frac = new_size * 1.0 / img.shape[0]
        ret = imresize(img, frac) / 255.0

        if digit_shapes is not None:
            assert digit_shapes[idx].shape == img.shape
            digit_shapes[idx] = imresize(digit_shapes[idx], frac) / 255.0
            assert digit_shapes[idx].shape == ret.shape, "{0}!={1}".format(digit_shapes[idx].shape, ret.shape)
        return ret
    return [resize(k, idx) for idx, k in enumerate(imgs)]

def random_place(img, frame_size, digit_shape=None):
    """ put img randomly inside a zero frame in frame_size
        return flags and results"""
    if digit_shape is not None:
        assert digit_shape.shape == img.shape
    offsets = np.random.random_sample((2, ))
    offsets = (int(offsets[0] * (frame_size[0] - img.shape[0])),
               int(offsets[1] * (frame_size[1] - img.shape[1])))
    ret = np.zeros(frame_size, dtype='float32')
    flag = np.zeros(frame_size, dtype='float32')
    for x in range(img.shape[0]):
        ret[x + offsets[0]][offsets[1]:offsets[1] + img.shape[1]] = img[x]
        if digit_shape is None:
            flag[x + offsets[0]][offsets[1]:offsets[1] + img.shape[1]] = np.array([1] * img.shape[1])
        else:
            flag[x + offsets[0]][offsets[1]:offsets[1] + img.shape[1]] = digit_shape[x]
    return flag, ret

def fill_vertical_blank(imgs, height):
    """ randomly shift the image vertically to fit the height"""
    seeds = np.random.random_sample((len(imgs), ))
    def fill(img, idx):
        assert img.shape[0] <= height
        if img.shape[0] == height:
            return img
        space = (height - img.shape[0])
        offset = int(space * seeds[idx])
        img = np.vstack((np.zeros((offset, img.shape[1])),
                         img,
                         np.zeros((space - offset, img.shape[1]))
                       ))
        return img
    return [fill(img, idx) for idx, img in enumerate(imgs)]

class SeqDataGenerator(object):

    def __init__(self, len_dist, dataset, max_width=None, max_height=None,
                rotate=False, resize=False, crazy=False, max_dist=None):
        """ len_dist: a dict containing the distribution of length.
            max_dist: maximum distance between digits, to make them closer to each other
        """
        lens = len_dist.keys()
        self.max_len = max(lens)
        probs = [len_dist[k] for k in lens]

        self.do_rotate = rotate
        self.do_resize = resize
        self.crazy = crazy
        self.max_dist = max_dist

        self.len_rvg = stats.rv_discrete(values=(lens, probs))
        # merge train/valid/test
        self.dataset = dataset
        shape = self.dataset[0][0][0].shape

        self.orig_image_shape = int(np.sqrt(shape[0]))
        assert self.orig_image_shape ** 2 == int(shape[0])
        if max_width is None:
            max_width = self.orig_image_shape * self.max_len
        if max_height is None:
            max_height = self.orig_image_shape
        self.img_size = (max_height, max_width)
        print "Original dataset size: {0}, {1}, {2}".format(len(dataset[0][0]),
                                                         len(dataset[1][0]),
                                                         len(dataset[2][0]))
        print "Image size: {0}".format(self.img_size)

    def gen_n_samples(self, n, dataset):
        """ generate n pairs of images/labels from dataset"""
        # generate a random length
        lens = self.len_rvg.rvs(size=n)

        rets = []
        labels = []
        prgs = Progressor(n)
        for idx, l in enumerate(lens):
            img, label = self.select_n_digits(l, dataset)
            rets.append(np.asarray(img, dtype='float32'))
            labels.append(label)

            if idx > 0 and idx % 1000 == 0:
                prgs.report(1000, True)
        return rets, labels

    def select_n_digits(self, n, dataset):
        """ select n digits to build a image"""
        index = np.random.choice(len(dataset[0]), n)
        imgs = dataset[0][index]
        imgs = [k.reshape(self.orig_image_shape, self.orig_image_shape) for k in imgs]
        labels = dataset[1][index]
        if len(dataset) == 3:
            shapes = [dataset[2][k] for k in index]
        else:
            shapes = None

        if self.crazy:
            if len(dataset) == 3:
                # has shape information
                return self.crazy_paste_image(imgs, labels, shapes)
            else:
                return self.crazy_paste_image(imgs, labels)

        else:
            return self.paste_image(imgs), labels

    def paste_image(self, imgs):
        """ directly paste images together, with some random shift and scaling. (deprecated)"""
        orig_imgs = imgs
        max_height = self.img_size[0]
        if self.do_rotate:
            imgs = random_rotate(imgs)
        if self.do_resize:
            imgs = random_resize(imgs, min(self.img_size))

        imgs = fill_vertical_blank(imgs, max_height)

        n_chunks = len(imgs) + 1
        space_left = self.img_size[1] - sum([k.shape[1] for k in imgs])
        if space_left <= 0:
            return self.paste_image(orig_imgs)
        assert space_left > 0
        chunks = random_slice(n_chunks, space_left)

        ret = np.zeros((max_height, chunks[0]), dtype='float32')
        for idx, k in enumerate(imgs):
            ret = np.hstack((ret, k, np.zeros((self.img_size[0], chunks[idx + 1]))))
        assert ret.shape == self.img_size
        return ret

    def crazy_paste_image(self, imgs, labels, shapes=None):
        """ pasge images so that their shape don't overlap"""
        cnt = 0
        while True:
            orig_shapes = copy(shapes)
            ret = self._do_crazy_paste_image(imgs, labels, orig_shapes)
            if ret != False:
                return ret
            cnt += 1
            if cnt % 200 == 0:
                print "Trying {0}...".format(cnt)

    def _do_crazy_paste_image(self, imgs, labels, shapes=None):
        """ return False when test failed.
            will modify shapes in place
        """
        if self.do_rotate:
            imgs = random_rotate(imgs)
        if self.do_resize:
            max_resize =  min(self.img_size)
            max_resize = min([70, max_resize])
            imgs = random_resize(imgs, max_resize, shapes)

        if shapes is None:
            frames = [random_place(k, self.img_size) for k in imgs]
        else:
            frames = [random_place(k, self.img_size, shape)
                      for k, shape in izip(imgs, shapes)]
        flags = [x[0] for x in frames]
        centers = [ndimage.measurements.center_of_mass(f) for f in flags]

        # check overlap digits
        n_overlap = np.sum(sum(flags) > 1.0)
        if n_overlap > 1 or not self._dist_ok(centers):
            return False

        # sort the labels by certer of mass
        labels = sorted(enumerate(labels),
                        key=lambda tp: centers[tp[0]][1] * 1000 + centers[tp[0]][0])
        labels = np.asarray([k[1] for k in labels], dtype='int32')

        ret = sum([x[1] for x in frames])
        return ret, labels

    def _dist_ok(self, centers):
        """ check whether images are close enough"""
        if self.max_dist is None or len(centers) == 1:
            return True
        for k in centers:
            for k2 in centers:
                if k == k2: continue
                dist = (k[0] - k2[0]) ** 2 + (k[1] - k2[1]) ** 2
                if dist < self.max_dist ** 2:
                    break
            else:
                return False
        return True

    def write_dataset(self, n_train, n_valid, n_test, fname):
        """ generate and write dataset to file"""
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
    dataset = dataio.read_data('./data/mnist.shaped.pkl.gz')
    fout = sys.argv[1]
    seq_len = int(sys.argv[2])

    lengths = range(1, 7)
    generator = SeqDataGenerator(
        dict([(k, 1.0 / len(lengths)) for k in lengths]),
        dataset, max_width=180, max_height=70,
        rotate=False, resize=True, crazy=True, max_dist=50)

    generator.write_dataset(100000, 10000, 10000, fout)




