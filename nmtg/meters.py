# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import time


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        return self.sum / self.count

    def update(self, val=1, n=1):
        self.val = val
        self.sum += val
        self.count += n


class TimeMeter:
    """Computes the average occurrence of some event per second"""
    def __init__(self, init=0):
        self.init = init
        self.start_time = None
        self.n = 0

    def start(self):
        self.start_time = time.time()
        self.n = 0

    def reset(self, init=0, n=0):
        self.init = init
        self.n = n
        self.start_time = None

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        if self.start_time is None:
            return self.init
        return self.init + (time.time() - self.start_time)


class StopwatchMeter:
    """Computes the sum/avg duration of some event in seconds"""
    def __init__(self):
        self.start_time = None
        self.sum = 0
        self.n = 0
        self.val = 0

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None
            self.val = delta

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None
        self.val = 0

    @property
    def avg(self):
        if self.n == 0:
            return 0.0
        return self.sum / self.n
