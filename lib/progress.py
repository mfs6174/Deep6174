#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: progress.py
# Date: Wed Sep 17 17:00:42 2014 -0700
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import time

class Progressor(object):
    """ progress logger"""
    def __init__(self, total, prompt=''):
        self.total = total
        self.prompt = prompt
        self.start_time = time.time()
        self.last = 0.0

    def report(self, cnt, incremental=False):
        if incremental:
            self.last += cnt
            cnt = self.last
        elapsed = time.time() - self.start_time

        if self.total:
            prgs = float(cnt) / self.total
            if prgs == 0:
                eta = float('inf')
            else:
                eta = elapsed / prgs - elapsed
                print "{} {} of {} done. elapsed: {:.2f}s ETA: {:.2f}s".format(
                    self.prompt,cnt, self.total, elapsed, eta)
        else:
            # no total. run forever. only report current time
            print "{} {} done. elapsed : {:.2f}s".format(
                self.prompt, cnt, elapsed)
