#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


from sair import specfem_io as sp
from sair.config import load_config
from sair.smooth_kernels import smooth_kernel
from sair import mpi

from itertools import chain

import numpy as np
import argparse


def find_src_ispec(ex, ez, xs, zs, nspec, ngllz, ngllx):
    for ispec in range(nspec):
        if xs[ispec, 0, 0] <= ex <= xs[ispec, 0, ngllx-1]:
            if zs[ispec, 0, 0] <= ez <= zs[ispec, ngllz-1, 0]:
                return ispec
    return None


def assign_smoothing(data, ex, ez, xs, zs, ibool,
                     maxdist):
    sigma = maxdist/3.0
    # src_ispec = find_src_ispec(ex, ez, xs, zs, *ibool.shape)
    # src_iglobs = []
    # for j in range(ngllz):
    #     for i in range(ngllx):
    #         src_iglobs.append(ibool[src_ispec, j, i])

    distances = np.sqrt((xs - ex)**2 + (zs - ez)**2)
    data[(distances <= maxdist).reshape(data.shape)] = sigma

    # data[] = sigma
    # for ispec in range(nspec):
    #     for j in range(ngllz):
    #         for i in range(ngllx):
    #             dist = np.sqrt((xs[ispec, j, i] - ex)**2 +
    #                            (zs[ispec, j, i] - ez)**2)
    #             if dist <= dist:
    #                 n = i+j*ngllx+ispec*ngllx*ngllz
    #                 data[n] = sigma
    return data


def find_typical_spec_size(xs, zs):
    nspec, ngllz, ngllx = xs.shape
    sizes = []
    for ispec in range(nspec):
        width = xs[ispec, 0, ngllx-1] - xs[ispec, 0, 0]
        height = zs[ispec, ngllz-1, 0] - zs[ispec, 0, 0]
        diag = np.sqrt(width**2 + height**2)
        sizes.append(diag)

    return np.mean(sizes)


def create_source_smoothing(conf, kernel_folder):
    ibool = sp.read_ibool_from_folder(kernel_folder, mpi.rank)
    nspec, ngllz, ngllx = ibool.shape
    xs = sp.read_data_matrix_from_folder(kernel_folder, "x",
                                         ibool.shape, mpi.rank)
    zs = sp.read_data_matrix_from_folder(kernel_folder, "z",
                                         ibool.shape, mpi.rank)
    typical_size = find_typical_spec_size(xs, zs)
    data = np.ones(nspec*ngllz*ngllx)*-1
    event_locs = conf.get_event_locs()
    for event, (ex, ez) in event_locs.iteritems():
        if conf.postprocessing.smooth_around_sources:
            assign_smoothing(data, ex, ez, xs, zs, ibool, typical_size)
        if conf.postprocessing.smooth_around_stations:
            for sx, sz in conf.get_stations(event, with_comps=False).values():
                assign_smoothing(data, sx, sz,
                                 xs, zs, ibool, typical_size)

    sp.write_data_to_folder(data, kernel_folder, "src_smoothing", mpi.rank)


def smooth_around_sources(conf, kernel_folder):
    xs = sp.read_data_from_folder(kernel_folder, "x", mpi.rank)
    zs = sp.read_data_from_folder(kernel_folder, "z", mpi.rank)
    sigma = sp.read_data_from_folder(kernel_folder,
                                     "src_smoothing", mpi.rank)

    kernels = [k+"_kernel" for k in
               chain(conf.simulation.kernels,
                     conf.postprocessing.preconditioners)]
    for kernelname in kernels:
        k = sp.read_data_from_folder(kernel_folder, kernelname, mpi.rank)
        sp.write_data_to_folder(k, kernel_folder,
                                kernelname+"_unsmoothed_sources",
                                mpi.rank)
        data = smooth_kernel(kernel_folder, kernelname, xs, zs,
                             sigma, sigma)
        sp.write_data_to_folder(data, kernel_folder, kernelname,
                                mpi.rank)


def main():
    parser = argparse.ArgumentParser(
        description="Smooth around sources/stations")
    parser.add_argument('conf_file')
    parser.add_argument('kernel_folder',
                        help="file that contain the kernel folders")

    args = parser.parse_args()
    conf = load_config(args.conf_file)
    create_source_smoothing(conf, args.kernel_folder)
    smooth_around_sources(conf, args.kernel_folder)


if __name__ == "__main__":
    main()
