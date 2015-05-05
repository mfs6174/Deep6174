#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: test-char.py
# Date: Thu Dec 11 00:19:50 2014 +0800
# Author: Xi-Jin Zhang <zhangxijin91@gmail.com>

import numpy as np
import sys,os
import cv2
import gzip
import cPickle as pickle

if len(sys.argv) <3:
    sys.exit("Usage: {} <model> <picture> <output-dir>".format(sys.argv[0]))

from network_runner import get_nn

params_file = sys.argv[1]
img_name = sys.argv[2]
nn = get_nn(params_file, 101)
output_dir = (sys.argv[3] if len(sys.argv)==4 else "./predicted/")
try:
    os.mkdir(output_dir)
except:
    pass
assert os.path.isdir(output_dir), "cannot create directory " + output_dir

pred = nn.predict_whole_img(img_name)
to_show = (pred*255.0).astype('uint8')
label = np.zeros_like(to_show)
indic = pred.argmax(-1)
for i in range(label.shape[0]):
    for j in range(label.shape[1]):
        label[i,j,indic[i,j]] = (255 if indic[i,j] != 2 else 0)
cv2.imwrite(output_dir+'/'+img_name+'_predicted.png',to_show)
cv2.imwrite(output_dir+'/'+img_name+'_labeled.png',label)
fout = gzip.open(output_dir+'/'+img_name+'_predicted.pkl.gz', 'wb')
pickle.dump(pred, fout, -1)
fout.close()
