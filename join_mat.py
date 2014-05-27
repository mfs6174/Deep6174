#!/usr/bin/env python2
# -*- coding: UTF-8 -*-

import scipy
import scipy.io as sio
import glob
import sys
import os

try:
    logdir = sys.argv[1]
except:
    logdir = '.'

params = {}
for f in glob.glob(logdir + '/*.mat'):
    basename = os.path.basename(f)
    epoch = basename[:-4]
    mat = sio.loadmat(f)
    params['epoch' + str(epoch)] = mat

sio.savemat(os.path.join(logdir, 'all_params.mat'), params)
