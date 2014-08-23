#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test-model.py
# Date: Fri Aug 22 23:30:36 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import sys
from itertools import izip

from network_runner import get_nn
from dataio import read_data, save_data
from imageutil import get_image_matrix

#def get_an_image(dataset, label):
    #""" get an image with label=number"""
    #train_set, valid_set, test_set = read_data(dataset)
    #for idx, img in enumerate(test_set[0]):
        #ys = test_set[1][idx]
        #if hasattr(ys, '__iter__'):
            #ys = ys[0]
        #if int(ys) != label:
            #continue
        #img = get_image_matrix(img)

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
corr, tot = 0, 0
if nn.var_len_output:
    # also record the precision for length
    len_tot, len_corr = 0, 0
for img, label in izip(test[0], test[1]):
    pred = nn.predict(img)
    #print pred, label
    if nn.multi_output and hasattr(pred, '__iter__'):
        if nn.var_len_output:
            seq_len = pred[0]
            # per-digit accuracy
            if len(label) != 5:
                continue
            tot += seq_len
            corr += sum([1 for i, j in izip(pred[1:1 + seq_len], label) if i == j])
            # absolute accuracy
            #tot += 1
            #corr += list(label) == list(pred[1:1+seq_len])

            len_tot += 1
            len_corr += pred[0] == len(label)
            if len_tot % 1000 == 0:
                print "Length predict accuracy: {0}".format(len_corr * 1.0 / len_tot)
        elif nn.multi_output:
            # Fixed length output
            tot += len(label)
            corr += len([k for k, _ in izip(pred, label) if k == _])
        else:
            tot += 1
            corr += label_match(pred, label)
    else:
        tot += 1
        corr += label == pred

    if tot % 1000 == 0 and tot > 0:
        print "Rate: {0}".format(corr * 1.0 / tot)
print "Rate: {0}".format(corr * 1.0 / tot)


# Usage ./test_model.py param_file.pkl.gz dataset.pkl.gz
