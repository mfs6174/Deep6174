#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test-model.py
# Date: Thu Aug 14 12:21:39 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import sys
from itertools import izip

from network_runner import NetworkRunner, label_match, get_nn
from dataio import read_data, save_data, get_dataset_imgsize
from imageutil import get_image_matrix

def get_an_image(dataset, label):
    """ get an image with label=number"""
    train_set, valid_set, test_set = read_data(dataset)
    for idx, img in enumerate(test_set[0]):
        ys = test_set[1][idx]
        if hasattr(ys, '__iter__'):
            ys = ys[0]
        if int(ys) != label:
            continue
        img = get_image_matrix(img)


if __name__ == '__main__':
    params_file = sys.argv[1]
    nn = get_nn(params_file)

    train, valid, test = read_data(sys.argv[2])
    corr, tot = 0, 0
    if nn.var_len_output:
        len_tot, len_corr = 0, 0
    for img, label in izip(test[0], test[1]):
        pred = nn.predict(img)
        if nn.multi_output and hasattr(pred, '__iter__'):
            if nn.var_len_output:
                seq_len = pred[0]
                tot += seq_len
                #corr += sum([1 for i, j in izip(pred[1:1 + seq_len], label)
                             #if i == j])
                corr += len(set(pred[1:]) & set(label))
                print pred, label

                len_tot += 1
                len_corr += pred[0] == len(label)
                if len_tot % 1000 == 0:
                    print "Length predict accuracy: {0}".format(len_corr * 1.0 / len_tot)
            elif len(label) == len(pred) - 1:
                tot += len(label)
                corr += len(set(label) & set(pred[1:]))
                #corr += len([k for k, _ in izip(pred, label) if k == _])
            else:
                tot += 1
                corr += label_match(pred, label)
        else:
            tot += 1
            corr += label == pred

        if tot % 1000 == 0:
            print "Rate: {0}".format(corr * 1.0 / tot)


# Usage ./test_model.py param_file.pkl.gz dataset.pkl.gz
