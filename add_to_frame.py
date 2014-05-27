#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: add_to_frame.py\2
# Date: Sun May 18 17:48:40 2014 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cPickle as pickle
#import pickle
import gzip, numpy
import random
import argparse

def get_args():
    desc = 'add img into a larger frame'
    parser = argparse.ArgumentParser(description = desc)

    parser.add_argument('-i', '--input',
                        help='input file of name "*.pkl.gz" ', required=True)
    parser.add_argument('-s', '--size',
                        help='frame size', required=True)
    parser.add_argument('-p', '--place',
                       help='place of the image. either "random" or "(x, y)"')

    ret = parser.parse_args()
    return ret


def add_img_to_frame(img, frame, offset):
    """put a smaller matrix into a larger frame,
    starting at a specific offset"""
    img = img.reshape((orig_size, orig_size))
    for x in xrange(orig_size):
        frame[x + offset[0]][offset[1]: offset[1] + orig_size] = img[x]

def add_frame(dataset):
    """ process a dataset consisting of a list of imgs"""
    if args.place != 'random':
        offset = eval(args.place)
        assert type(offset) == tuple and len(offset) == 2

    Xs = dataset[0]
    newX = []
    for (idx, k) in enumerate(Xs):
        if args.place == 'random':
            # generate a random offset
            offset = (random.randint(0, frame_size - orig_size),
                         random.randint(0, frame_size - orig_size))
        frame = numpy.zeros((frame_size, frame_size), dtype=numpy.float32)
        add_img_to_frame(k, frame, offset)
        newX.append(numpy.ndarray.flatten(frame))
    return (numpy.asarray(newX), dataset[1])

# prepare params
args = get_args()
input = args.input
frame_size = int(args.size)
output = input[:-6] + "frame{0}.pkl.gz".format(frame_size)

# read data
f = gzip.open(input, 'rb')
train_set, valid_set, test_set = pickle.load(f)
print len(train_set[0]), len(valid_set[0]), len(test_set[0])
orig_size = int(numpy.sqrt(len(train_set[0][0])))
f.close()

assert frame_size > orig_size, "frame size must be larger than original image"

# add to frame
train_set = add_frame(train_set)
valid_set = add_frame(valid_set)
test_set = add_frame(test_set)

print "Writing..."
data = (train_set, valid_set, test_set)
fout = gzip.open(output, 'wb')
pickle.dump(data, fout, -1)
fout.close()

#usage: add_to_frame.py [-h] -i INPUT -s SIZE [-p PLACE]
#optional arguments:
  #-h, --help            show this help message and exit
  #-i INPUT, --input INPUT
                        #input file of name "*.pkl.gz"
  #-s SIZE, --size SIZE  frame size
  #-p PLACE, --place PLACE
                        #place of the image. either "random" or "(x, y)"
# output filename is 'input.frameXX.pkl.gz'
