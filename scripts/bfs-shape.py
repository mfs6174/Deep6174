#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: bfs-shape.py
# Date: Thu Sep 18 15:43:27 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
from scipy.misc import imread, toimage
from collections import deque, Counter
import os, sys

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
from dataio import read_data, save_data
from lib.imageutil import get_image_matrix

def find_shape(img):
    ZERO_THRES = 0.1
    """ return a ndarray of boolean"""
    label = {}
    now_label = 20

    def bfs(i, j, lb):
        q = deque()
        q.append((i, j))

        def valid(x, y):
            return x >= 0 and x < img.shape[0] and y >= 0 \
                    and y < img.shape[1] and img[x][y] > ZERO_THRES and (x, y) not in label

        def add(x, y):
            if valid(x, y):
                q.append((x, y))

        while len(q) > 0:
            top = q.pop()
            label[top] = lb
            add(top[0] - 1, top[1])
            add(top[0] + 1, top[1])
            add(top[0], top[1] - 1)
            add(top[0], top[1] + 1)
            add(top[0] - 1, top[1] - 1)
            add(top[0] - 1, top[1] + 1)
            add(top[0] + 1, top[1] - 1)
            add(top[0] + 1, top[1] + 1)


    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] <= ZERO_THRES:
                continue
            if (i, j) in label:
                continue
            bfs(i, j, now_label)
            now_label += 1

    # find largest label
    c = Counter(label.values())
    most_common = c.most_common()[0][0]

    x = np.zeros(img.shape)
    for k, v in label.iteritems():
        if v == most_common:
            x[k] = 1
    return x

train, valid, test = read_data("mnist.pkl.gz")

def work_dataset(X, y):
    Z = []
    for idx, img in enumerate(X):
        img = get_image_matrix(img)
        Z.append(find_shape(img))
        if idx % 1000 == 0:
            print "Progress: {0} / {1}".format(idx, len(X))
    Z = np.asarray(Z)
    return (X, y, Z)

train = work_dataset(*train)
valid = work_dataset(*valid)
test = work_dataset(*test)
save_data((train, valid, test), "mnist.shaped")
