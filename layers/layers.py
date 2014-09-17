#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: layers.py
# Date: Tue Sep 16 23:31:37 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


#from logistic_sgd import LogisticRegression
#from fixed_length_softmax import FixedLengthSoftmax

from common import Layer

from conv import ConvLayer
from pool import PoolLayer
from sub import MeanSubtractLayer
from maxout import MaxoutLayer
from fc import FullyConnectedLayer

from sequence_softmax import SequenceSoftmax

name_dict = {ConvLayer: 'convpool',
             FullyConnectedLayer: 'fc',
             SequenceSoftmax: 'ssm',
             PoolLayer: 'pool',
             MeanSubtractLayer: 'sub',
             MaxoutLayer: 'maxout'
            }

