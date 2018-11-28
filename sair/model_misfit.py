#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import argparse
import numpy as np
import glob

from .specfem_io import read_all_from_folder


def rms(data_one, data_two):
    return np.sqrt(np.sum((data_one-data_two)**2)/len(data_one))


def load_model(first_model_file):
    files = sorted(glob.glob(first_model_file.replace("000000", "*")))
    data = np.loadtxt(files[0])
    for file_ in files[1:]:
        new_data = np.loadtxt(file_)
        data = np.append(data, new_data,
                         axis=0)
    return data


def print_difference(model_one, model_two):
    indices = [2, 3, 4]
    labels = ["rho", "Vp", "Vs"]
    misfit = {}
    for i, label in zip(indices, labels):
        misfit[label] = rms(model_one[:, i], model_two[:, i])

    for label, value in misfit.iteritems():
        print("{:3s} {:.5f}".format(label, value))


def calc_model_misfit(model_one, model_two, model_parameter):
    one = read_all_from_folder(model_one, model_parameter)
    two = read_all_from_folder(model_two, model_parameter)
    return rms(one, two)


def main():
    parser = argparse.ArgumentParser(
        description='RMS difference between two models')
    parser.add_argument('model_one')
    parser.add_argument('model_two')
    parser.add_argument('model_parameter')

    args = parser.parse_args()
    print(calc_model_misfit(args.model_one, args.model_two,
                            args.model_parameter))


if __name__ == "__main__":
    main()
