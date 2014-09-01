#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test-model.py
# Date: Mon Sep 01 14:58:05 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import sys
from itertools import izip

from network_runner import get_nn
from dataio import read_data, save_data
from imageutil import get_image_matrix
from accuracy import AccuracyRecorder

if len(sys.argv) != 3:
    print "Usage: {0} <model file> <dataset.pkl.gz>".format(sys.argv[0])
    sys.exit()

def label_match(l1, l2):
    """ for two sequences with different length,
    return whether the short one matches the long one(is a substring) """
    l1, l2 = list(l1), list(l2)
    if len(l1) > len(l2):
        l1, l2 = l2, l1
    now = -1
    for k in l1:
        try:
            now = l2.index(k, now + 1)
        except:
            return False
    return True

# get the network
params_file = sys.argv[1]
nn = get_nn(params_file)

train, valid, test = read_data(sys.argv[2])
digit_accu = AccuracyRecorder('digit')
abs_accu = AccuracyRecorder('abs')

len_sep_accu = []
for l in [1, 2, 3, 4, 5]:
    len_sep_accu.append(AccuracyRecorder('l{0}-abs'.format(l)))
if nn.var_len_output:
    # also record the accuracy for length
    length_accu = AccuracyRecorder('length')

for img, label in izip(test[0], test[1]):
    pred = nn.predict(img)
    #print pred, label
    if nn.multi_output and hasattr(pred, '__iter__'):
        if nn.var_len_output:
            seq_len = pred[0]
            if len(label) <= len(len_sep_accu):
                len_sep_accu[len(label) - 1].update(1,
                               list(label) == list(pred[1:1+seq_len]))
            # digit level accuracy
            digit_accu.update(seq_len,
                              sum([1 for i, j in izip(pred[1:1 + seq_len], label) if i == j]))
            # absolute accuracy
            abs_accu.update(1,
                            list(label) == list(pred[1:1+seq_len]))

            # length accuracy
            length_accu.update(1, pred[0] == len(label))
        elif nn.multi_output:
            # Fixed length output
            digit_accu.update(len(label),
                              len([k for k, _ in izip(pred, label) if k == _]))
        else:
            digit_accu.update(1, label_match(pred, label))
    else:
        abs_accu.update(1, label == pred)
digit_accu.log()
abs_accu.log()
length_accu.log()
for accu in len_sep_accu:
    accu.log()


# Usage ./test_model.py param_file.pkl.gz dataset.pkl.gz
