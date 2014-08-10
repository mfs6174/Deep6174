#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: scale-dataset.py
# Date: Sun Aug 10 15:30:31 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import sys
from scipy.misc import imresize
import numpy as np

from dataio import read_data, save_data

dataset = sys.argv[1]
factor = float(sys.argv[2])

t, v, ts = read_data(dataset)

def process(ds):
    return ([imresize(img, factor) for img in ds[0]], ds[1])

newname = dataset[:-6] + '-factor{0}.pkl.gz'.format(factor)
newdata = (process(t), process(v), process(ts))
save_data(newdata, newname)

# Usage: ./scale-dataset.py input.pkl.gz 0.6
# will generate 'input-factor0.6.pkl.gz',
