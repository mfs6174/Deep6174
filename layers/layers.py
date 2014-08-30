#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: layers.py
# Date: Fri Aug 29 22:14:14 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer
from fixed_length_softmax import FixedLengthSoftmax
from sequence_softmax import SequenceSoftmax

name_dict = {LeNetConvPoolLayer: 'convpool',
             HiddenLayer: 'hidden',
             LogisticRegression: 'lr',
             FixedLengthSoftmax: 'fl-sm',
             SequenceSoftmax: 'ssm'}
