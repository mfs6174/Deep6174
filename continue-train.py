#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: continue-train.py
# Date: Sun Sep 21 17:46:11 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# Author: Xi-Jin Zhang <zhangxijin91@gmail.com>

from network_runner import build_nn_with_params

import cPickle as pickle
import gzip
import sys
import operator

if len(sys.argv) < 3:
    print "Usage: {0} <model file to continue on> dataset.pkl.gz [output directory]".format(sys.argv[0])
    sys.exit()

# get the network from saved params
model = sys.argv[1]
with gzip.open(model, 'r') as f:
    data = pickle.load(f)
nn_runner = build_nn_with_params(data, None)
nn = nn_runner.nn

input_size = nn.input_shape
#load_all = reduce(operator.mul, input_size[1:]) < 100 ** 2
load_all=False
dataset = sys.argv[2]

if len(sys.argv) == 4:
    output_directory = sys.argv[3]
else:
    output_directory = dataset + '-output-continue'
nn.work(0.001, dataset, load_all, output_directory)
