# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import argparse

import numpy as np
from .utils import tqdm

from itertools import chain

from .specfem_io import read_all_with_coord
from .specfem_io import write_data_to_folder
from .specfem_io import read_all_from_folder
from .specfem_io import read_data_from_folder
from .specfem_io import read_ibool_from_folder
from .specfem_io import get_glj_weights

from .config import load_config

from . import mpi


def smooth_kernel(kernel_folder, kernelname, xs, zs, sigmah, sigmav,
                  all_jac=None, all_ibool=None, all_wgllsq=None):
    all_x, all_z, all_data = read_all_with_coord(kernel_folder, kernelname)
    if all_jac is None:
        all_jac = read_all_from_folder(kernel_folder, "jacobian")
        all_ibool = np.zeros(all_jac.shape)
        all_wgllsq = np.zeros(all_jac.shape)
        rstart = 0
        for r in range(mpi.size):
            ibool = read_ibool_from_folder(kernel_folder, r)
            nspec, ngllz, ngllx = ibool.shape
            weights = get_glj_weights(ngllx)
            for ispec in range(nspec):
                for j in range(ngllz):
                    for i in range(ngllx):
                        n = i+j*ngllx+ispec*ngllx*ngllz + rstart
                        all_ibool[n] = rstart + ibool[ispec, j, i]
                        all_wgllsq[n] = weights[i]*weights[j]
            rstart += nspec*ngllx*ngllz

    varh2 = 2*sigmah**2
    varv2 = 2*sigmav**2
    varh2_inv = 1 / varh2
    varv2_inv = 1 / varv2
    sigma3h = 3*sigmah
    sigma3v = 3*sigmav

    data = read_data_from_folder(kernel_folder, kernelname, mpi.rank)
    for i, (x, z) in tqdm(enumerate(zip(xs, zs)), total=len(xs)):
        if sigmah[i] <= 0 or sigmav[i] <= 0:
            continue
        test = np.logical_and(np.sqrt((all_x-x)**2) < sigma3h[i],
                              np.sqrt((all_z-z)**2) < sigma3v[i])
        close_x = all_x[test]
        close_z = all_z[test]
        close_data = all_data[test]
        close_jac = all_jac[test]
        close_wgllsq = all_wgllsq[test]
        close_fac = close_jac*close_wgllsq
        close_ibool = all_ibool[test]

        G = np.exp(- (x - close_x)**2*varh2_inv[i]
                   - (z - close_z)**2*varv2_inv[i])
        data[i] = 0
        seen = []
        total_weight = 0
        for ib, fac, g, d in zip(close_ibool, close_fac, G, close_data):
            if ib not in seen:
                weight = g*fac
                data[i] += weight*d
                total_weight += weight
                seen.append(ib)
        data[i] /= total_weight
    return data


def smooth_kernels(conf, kernel_folder):
    xs = read_data_from_folder(kernel_folder, "x", mpi.rank)
    zs = read_data_from_folder(kernel_folder, "z", mpi.rank)

    all_jac = read_all_from_folder(kernel_folder, "jacobian")
    all_ibool = np.zeros(all_jac.shape)
    all_wgllsq = np.zeros(all_jac.shape)
    rstart = 0
    for r in range(mpi.size):
        ibool = read_ibool_from_folder(kernel_folder, r)
        nspec, ngllz, ngllx = ibool.shape
        weights = get_glj_weights(ngllx)
        for ispec in range(nspec):
            for j in range(ngllz):
                for i in range(ngllx):
                    n = i+j*ngllx+ispec*ngllx*ngllz + rstart
                    all_ibool[n] = rstart + ibool[ispec, j, i]
                    all_wgllsq[n] = weights[i]*weights[j]
        rstart += nspec*ngllx*ngllz

    preconds = conf.postprocessing.preconditioners

    kernels = [k+"_kernel" for k in
               chain(conf.simulation.kernels, preconds)]

    # all_x = read_all_from_folder(kernel_folder, "x")
    # all_z = read_all_from_folder(kernel_folder, "z")

    if conf.postprocessing.smooth_adaptive:
        pp = conf.postprocessing
        # data should be between 0-1
        adaptive_data = read_all_from_folder(
            pp.smooth_adaptive_data_folder, pp.smooth_adaptive_filename)
        if np.min(adaptive_data) < 0:
            raise Exception("Data should not have negative values"
                            " for for adaptive smoothing.")
        adaptive_data /= np.max(adaptive_data)
        adaptive_data = 1 - adaptive_data
        sigmav = adaptive_data*(pp.max_sigmav-pp.min_sigmav) + pp.min_sigmav
        sigmah = adaptive_data*(pp.max_sigmah-pp.min_sigmah) + pp.min_sigmah
        # if mpi.rank == 0:
        #     from sair.plot import plot_bin
        #     import matplotlib.pyplot as plt
        #     all_x = read_all_from_folder(kernel_folder, "x")
        #     all_z = read_all_from_folder(kernel_folder, "z")

        #     plot_bin(all_x, all_z, sigmav, show=False, title="$\sigma_v$")
        #     plot_bin(all_x, all_z, sigmah, show=False, title="$\sigma_h$")
        #     plt.show()
        # raise Exception("Halt!")
    else:
        sigmav = np.ones(all_jac.shape)*conf.postprocessing.sigmav
        sigmah = np.ones(all_jac.shape)*conf.postprocessing.sigmah

    for kernelname in kernels:
        data = smooth_kernel(kernel_folder, kernelname, xs, zs, sigmah, sigmav,
                             all_jac, all_ibool, all_wgllsq)
        write_data_to_folder(data, kernel_folder,
                             kernelname+"_smooth", mpi.rank)


def main():
    parser = argparse.ArgumentParser(
        description="Smooth kernels")
    parser.add_argument('conf_file')
    parser.add_argument('kernel_folder',
                        help="file that contain the kernel folders")

    args = parser.parse_args()
    conf = load_config(args.conf_file)
    print("args.kernels_folder", args.kernel_folder)
    smooth_kernels(conf, args.kernel_folder)


if __name__ == '__main__':
    main()
