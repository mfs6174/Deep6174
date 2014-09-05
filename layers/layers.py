#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: layers.py
# Date: Fri Sep 05 14:23:19 2014 +0000
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from logistic_sgd import LogisticRegression
from mlp import HiddenLayer, DropoutMLP
from conv_pool import ConvPoolLayer
from fixed_length_softmax import FixedLengthSoftmax
from sequence_softmax import SequenceSoftmax

name_dict = {ConvPoolLayer: 'convpool',
             HiddenLayer: 'hidden',
             DropoutMLP: 'mlp',
             LogisticRegression: 'lr',
             FixedLengthSoftmax: 'fl-sm',
             SequenceSoftmax: 'ssm'}
