# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import argparse

import numpy as np
from .utils import tqdm

from .specfem_io import read_data_from_folder
from .specfem_io import write_data_to_folder
from .config import load_config

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_kernel_folders(filename):
    with open(filename) as f:
        folders = [line.replace("\n", "")
                   for line in f.readlines()]
    return folders


def sum_kernels(kernel_name, folders, output_folder, absolute=False):
    result = read_data_from_folder(folders[0], kernel_name, rank)
    if absolute:
        result = np.abs(result)
    for folder in folders[1:]:
        kernel = read_data_from_folder(folder, kernel_name, rank)
        if absolute:
            kernel = np.abs(kernel)
        result += kernel

    write_data_to_folder(result, output_folder, kernel_name, rank)


def get_kernelname_and_modifiers(kernelname):
    data = kernelname.split(":")
    name = data[0]
    modifiers = {}
    if len(data) > 1:
        for m in data[1:]:
            modifiers[m] = True
    return name, modifiers


def create_source_mask(xs, zs, sources, source_rad):
    mask = np.ones_like(xs)
    sigma = source_rad / 3 # source_rad = 3*sigma
    var_inv = 1.0 / sigma**2
    # If there is more than one rank, filter the sources
    if size > 0:
        print("Filtering sources...")
        filtered_sources = []
        xmin, xmax = min(xs), max(xs)
        zmin, zmax = min(zs), max(zs)
        for s_loc in sources:
            if xmin <= s_loc[0] <= xmax and zmin <= s_loc[1] <= zmax:
                filtered_sources.append(s_loc)
        sources = filtered_sources

    xz = np.array(zip(xs, zs))
    for s_loc in tqdm(sources):
        dists = np.sum((xz-s_loc)**2, 1)
        mask *= (1.0 - np.exp(-dists*var_inv))
    return mask


def mask_kernel(conf, kernel_folder, kernelnames):
    x = read_data_from_folder(kernel_folder, "x", rank)
    z = read_data_from_folder(kernel_folder, "z", rank)
    mask = np.ones_like(x)

    if conf.postprocessing.mask_sources:
        sources = conf.get_event_locs().values()
        source_mask = create_source_mask(x, z, sources,
                                         conf.postprocessing.mask_radius)
        mask *= source_mask

    if conf.postprocessing.mask_stations:
        all_stations = conf.get_stations(conf.get_event_names(),
                                         add_event_label=True).values()
        unique_stations = [np.array(b)
                           for b in set([tuple(a) for a in all_stations])]
        receiver_mask = create_source_mask(x, z, unique_stations,
                                           conf.postprocessing.mask_radius)
        mask *= receiver_mask

    write_data_to_folder(mask, kernel_folder, "mask", rank)

    for kernelname in kernelnames:
        data = read_data_from_folder(kernel_folder, kernelname, rank)
        write_data_to_folder(data, kernel_folder, kernelname+"_unmasked", rank)
        write_data_to_folder(data*mask, kernel_folder, kernelname, rank)


def main():
    parser = argparse.ArgumentParser(
        description="Mask the kernel")
    parser.add_argument('config_file')
    parser.add_argument('kernel_folder',
                        help="file that contain the kernel folders")
    parser.add_argument('kernelnames',
                        nargs="+",
                        help="Kernels To mask.")

    args = parser.parse_args()
    conf = load_config(args.config_file)
    mask_kernel(conf, args.kernel_folder, args.kernelnames)


if __name__ == '__main__':
    main()
