#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: scale-dataset.py
# Date: Thu Sep 18 15:44:12 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import sys
from scipy.misc import imresize
import numpy as np

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
from dataio import read_data, save_data

dataset = sys.argv[1]
factor = float(sys.argv[2])

t, v, ts = read_data(dataset)

def process(ds):
    return ([imresize(img, factor) for img in ds[0]], ds[1])

newname = dataset[:-7] + '-factor{0}'.format(factor)
newdata = (process(t), process(v), process(ts))
save_data(newdata, newname)

# Usage: ./scale-dataset.py input.pkl.gz 0.6
# will generate 'input-factor0.6.pkl.gz',
