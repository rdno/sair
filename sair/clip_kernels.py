# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import argparse

import numpy as np
from .utils import tqdm

from itertools import chain

from .specfem_io import read_all_from_folder
from .specfem_io import read_data_from_folder
from .specfem_io import write_data_to_folder
from .config import load_config

import matplotlib.pyplot as plt

from . import mpi


def get_auto_limits(kernel_folder, kernelname, plot=False):
    all_data = read_all_from_folder(kernel_folder, kernelname)
    all_data[all_data < 0] = -all_data[all_data < 0]
    bins = np.logspace(np.log10(all_data.min()) - 1,
                       np.log10(all_data.max()) + 1,
                       10001)
    hist, edges = np.histogram(all_data, bins=bins)
    mids = np.array([(e-b)+b for b, e in zip(edges, edges[1:])])
    mids = np.hstack((-np.flip(mids, 0), mids))
    hist = np.hstack((np.flip(hist, 0), hist))

    # calc moments
    m0 = np.sum(hist)
    mu = np.sum(hist*mids)/m0
    var = np.sum(hist*(mids-mu)**2)/m0
    std = np.sqrt(var)
    limit = 10*std

    if plot:
        d = np.exp(-(mids-mu)**2/(2*var))
        fig, ax = plt.subplots()
        ax.plot(mids, hist/hist.max(), "b", label="data")
        ax.plot(mids, d, "g", label="fit")
        ylim = ax.get_ylim()
        ax.plot([mu - limit, mu - limit], ylim, "r--")
        ax.plot([mu + limit, mu + limit], ylim, "r--")
        ax.set_title(kernelname)
        ax.legend()

        all_data = read_all_from_folder(kernel_folder, kernelname)
        out_of_region = np.logical_or(all_data > mu + limit,
                                      all_data < mu - limit)

        print("{}: {} will be removed out of {} ({}%).".format(
            kernelname,
            len(all_data[out_of_region]), len(all_data),
            len(all_data[out_of_region])/len(all_data)))

    return mu - limit, mu + limit


def clip_kernel(conf, kernel_folder, analyze=False):
    pp = conf.postprocessing
    if pp.clip_kernels is False and False:
        print("Clipping is set to false.")
        return

    preconds = pp.preconditioners
    kernels = [k+"_kernel" for k in
               chain(conf.simulation.kernels, preconds)]

    for kernelname in kernels:
        if analyze and mpi.is_main_rank:
            get_auto_limits(kernel_folder, kernelname, plot=True)
        else:
            if pp.clip_minval == "auto" or pp.clip_maxval == "auto":
                min_val, max_val = get_auto_limits(kernel_folder, kernelname)
            else:
                min_val, max_val = pp.clip_minval, pp.clip_maxval
            data = read_data_from_folder(kernel_folder, kernelname, mpi.rank)
            write_data_to_folder(data, kernel_folder, kernelname+"_unclipped",
                                 mpi.rank)
            data[data > max_val] = max_val
            data[data < min_val] = min_val
            write_data_to_folder(data, kernel_folder, kernelname, mpi.rank)

    if analyze and mpi.is_main_rank:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Clip the kernels")
    parser.add_argument('config_file')
    parser.add_argument('kernel_folder',
                        help="file that contain the kernel folders")
    parser.add_argument('--analyze', "-a", action="store_true",
                        help="analyze values for the kernels")

    args = parser.parse_args()
    conf = load_config(args.config_file)
    clip_kernel(conf, args.kernel_folder, args.analyze)


if __name__ == '__main__':
    main()
