#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: continue_train.py
# Date: Mon Aug 11 11:25:58 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from run_network import build_nn_with_params

import cPickle as pickle
import gzip
import sys

if len(sys.argv) != 3:
    print "Usage: {0} <model file to continue on> dataset.pkl.gz".format(sys.argv[0])
    sys.exit()

model = sys.argv[1]
with gzip.open(model, 'r') as f:
    data = pickle.load(f)
nn_runner = build_nn_with_params(data, 500)
nn = nn_runner.nn

input_size = nn_runner.input_size
load_all = input_size[0] * input_size[1] < 100 ** 2

dataset = sys.argv[2]
nn.work(init_learning_rate=0.04, dataset_file=dataset, n_epochs=1000,
       load_all_data=load_all)
