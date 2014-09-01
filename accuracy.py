#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: accuracy.py
# Date: Mon Sep 01 14:06:23 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

class AccuracyRecorder(object):
    def __init__(self, name, update_interv=1000):
        self.tot = 0
        self.corr = 0
        self.cnt = 0
        self.update_interv = update_interv
        self.name = name

    def update(self, tot, corr):
        self.tot += tot
        self.corr += corr
        self.cnt += 1
        if self.cnt % self.update_interv == 0:
            self.log()

    def log(self):
        if self.cnt == 0:
            return
        print "{0} after {1}, accu={2}".format(self.name, self.cnt, self.corr * 1.0/ self.tot)
