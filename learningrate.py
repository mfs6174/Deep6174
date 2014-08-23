#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: learningrate.py
# Date: Fri Aug 22 22:58:34 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

class LearningRateProvider(object):
    """ file based learning rate provider.
        we can then modify the file to control learning rate
    """

    def __init__(self, filename, init_value):
        self.filename = filename
        with open(filename, 'w') as f:
            f.write('{0}\n'.format(init_value))

    def get_rate(self, epoch):
        """ read learning rate from the file"""
        with open(self.filename, 'r') as f:
            for idx, line in enumerate(f):
                if idx == epoch - 1:
                    return float(line)
        # return the last line if line is not enough
        return float(line)

if __name__ == '__main__':
    import time
    lrp = LearningRateProvider('/tmp/test.lr', 0.1)
    time.sleep(5)
    print lrp.get_rate(5)

