# -*- coding: utf-8 -*-
# @Time    : 2021/1/30 15:33
# @Author  : -----↖(^?^)↗
# @FileName: utils.py
# -------------------------start-----------------------------
import torch
from matplotlib import pyplot as plt
import numpy as np


def construct_maps(keys):
    d = dict()
    keys = sorted(list(set(keys)))
    for k in keys:
        if k not in d:
            d[k] = len(list(d.keys()))
    return d


def draw(outs, labels, path):
    figure = plt.figure()
    plt.xlabel('pred')
    plt.ylabel('gt')
    plt.scatter(outs, labels)
    low = min(torch.min(outs), torch.min(labels)) - 0.01
    up = max(torch.max(outs), torch.max(labels)) + 0.01
    plt.plot(np.arange(low, up, 0.01), np.arange(low, up, 0.01))
    plt.savefig(path)
    return figure


def draw_rank(gt_rank, pre_rank, path):
    figure = plt.figure()
    plt.xlabel('gt rank')
    plt.ylabel('pre rank')
    plt.scatter(gt_rank, pre_rank)
    plt.savefig(path)
    return figure


def leeway(outs, labels):
    leeway = [0.01, 0.02, 0.05, 0.1]
    correct = [0, 0, 0, 0]
    for i, l in enumerate(leeway):
        for out, lab in zip(outs, labels):
            # if (1 - l) * out <= lab <= (1 + l) * out:  # (1 - l) * lab <= out <= (1 + l) * lab
            if (1 - l) * lab <= out <= (1 + l) * lab:
                correct[i] += 1
    return [i / len(outs) for i in correct]


# https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)
