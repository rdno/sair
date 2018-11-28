#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
from scipy.optimize import curve_fit


def read_searchfile(filename):
    steps = []
    misfits = []
    with open(filename) as f:
        for line in f:
            step, misfit = line.split()
            steps.append(float(step))
            misfits.append(float(misfit))
    return np.array(steps), np.array(misfits)


def parabola(x, a, b, c):
    return abs(a)*x**2 + b*x + c


def next_step_from_linefitting(steps, misfits, step_size):
    degree = max([len(steps) - 1, 2])
    dots = np.linspace(min(steps), max(steps)+step_size, 1000)
    # c, _ = curve_fit(parabola, steps, misfits)
    z = np.polyfit(steps, misfits, degree)
    p = np.poly1d(z)
    zp = p(dots)
    # zp = parabola(dots, *c)
    # print("PARABOLA")
    x = dots[zp.argmin()]
    if x == 0:
        min_step = min(steps[1:])
        return min_step/2
    return x


def get_next_step(conf, linesearch_file):
    steps, misfits = read_searchfile(linesearch_file)
    if len(steps) == 1:
        return conf.linesearch.first
    else:
        if len(steps) >= 3:
            return next_step_from_linefitting(steps, misfits,
                                              conf.linesearch.step)
        elif misfits[-1] <= misfits[-2]:  # Increase the step length
            return steps[-1] + conf.linesearch.step
        else:
            return steps[-1] / 2


class LinesearchFailed(Exception):
    pass


def get_best(conf, linesearch_file):
    steps, misfits = read_searchfile(linesearch_file)
    old_misfit = misfits[0]
    min_misfit = min(misfits)
    min_step = steps[misfits.argmin()]

    if min_step < conf.linesearch.min_step or \
       (1 - min_misfit/old_misfit) < conf.linesearch.min_improvement:
        raise LinesearchFailed
    else:
        return min_step, min_misfit
