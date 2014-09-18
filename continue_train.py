#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: continue_train.py
# Date: Wed Sep 17 23:02:26 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from network_runner import build_nn_with_params

import cPickle as pickle
import gzip
import sys
import operator

if len(sys.argv) != 3:
    print "Usage: {0} <model file to continue on> dataset.pkl.gz".format(sys.argv[0])
    sys.exit()

# get the network from saved params
model = sys.argv[1]
with gzip.open(model, 'r') as f:
    data = pickle.load(f)
nn_runner = build_nn_with_params(data, None)
nn = nn_runner.nn

input_size = nn_runner.input_shape
load_all = reduce(operator.mul, input_size[1:]) < 100 ** 2

dataset = sys.argv[2]
nn.work(init_learning_rate=0.01,
        dataset_file=dataset,
        load_all_data=load_all)
