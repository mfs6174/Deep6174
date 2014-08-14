#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: paste_image.py
# Date: Sat May 31 00:29:45 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

# Paste images of the same size

import sys
import glob
import argparse
import scipy.misc as smisc
import Image
import numpy as np

def get_args():
    desc = 'Paste images of the same size'
    parser = argparse.ArgumentParser(description = desc)

    parser.add_argument('-i', '--input',
                        help='input files wildcard, must be passed quoted', required=True)

    parser.add_argument('-o', '--output',
                        help='output files, default is "output.jpg"',
                        required=False)

    ret = parser.parse_args()
    return ret

global args

args = get_args()

print args.input

files = glob.glob(args.input)
img = smisc.imread(files[0])
size = int(img.shape[0])
print "{0} images with size = {1}".format(len(files), size)

width = int(np.ceil(np.sqrt(len(files))))

frame = Image.new("L", (width * size, width * size), "white")
print frame.size
for idx, f in enumerate(files):
    row, col = idx / width, idx % width
    print row, col
    img = Image.open(f)
    frame.paste(img, (int(col * size), int(row * size)))

if args.output:
    output = args.output
else:
    output = "output.jpg"
frame.save(output)
