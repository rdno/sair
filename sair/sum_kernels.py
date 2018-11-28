# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import argparse

import numpy as np
from .utils import tqdm

from .specfem_io import read_data_from_folder
from .specfem_io import write_data_to_folder

from .config import load_config

from . import mpi


def get_kernel_folders(filename):
    with open(filename) as f:
        folders = [line.replace("\n", "")
                   for line in f.readlines()]
    return folders


def sum_kernels_old(kernel_name, folders, output_folder, absolute=False):
    result = read_data_from_folder(folders[0], kernel_name, mpi.rank)
    if absolute:
        result = np.abs(result)
    for folder in folders[1:]:
        kernel = read_data_from_folder(folder, kernel_name, mpi.rank)
        if absolute:
            kernel = np.abs(kernel)
        result += kernel

    write_data_to_folder(result, output_folder, kernel_name, mpi.rank)


def sum_kernels(kernel_name, conf, misfit_type, it,
                output_folder, absolute=False):
    events = conf.get_event_names()
    folders = [conf.get_event_kernel_folder(e, misfit_type, it)
               for e in events]
    src_weights = conf.get_src_weights()
    weights = [src_weights[e] for e in events]
    result = read_data_from_folder(folders[0], kernel_name, mpi.rank)
    if absolute:
        result = np.abs(result)
    result *= weights[0]
    for i, folder in enumerate(folders[1:], 1):
        kernel = read_data_from_folder(folder, kernel_name, mpi.rank)
        if absolute:
            kernel = np.abs(kernel)
        result += weights[i]*kernel

    write_data_to_folder(result, output_folder, kernel_name, mpi.rank)


def get_kernelname_and_modifiers(kernelname):
    data = kernelname.split(":")
    name = data[0]
    modifiers = {}
    if len(data) > 1:
        for m in data[1:]:
            modifiers[m] = True
    return name, modifiers


def main_old():
    parser = argparse.ArgumentParser(
        description="Sum the kernels")
    parser.add_argument('kernel_folders',
                        help="file that contain the kernel folders")
    parser.add_argument('output_folder',
                        help="folder to output")
    parser.add_argument('kernelnames',
                        nargs="+",
                        help="""Kernels To Sum.
Add modifier after colon (e.g. Hessian2_kernel:absolute)""")

    args = parser.parse_args()
    folders = get_kernel_folders(args.kernel_folders)
    for kernelname in tqdm(args.kernelnames, desc="Sum Kernels"):
        name, modifiers = get_kernelname_and_modifiers(kernelname)
        sum_kernels(name, folders, args.output_folder, **modifiers)


def main():
    parser = argparse.ArgumentParser(
        description="Sum the kernels")
    parser.add_argument('conf_file',
                        help="Config filename")
    parser.add_argument('misfit_type',
                        help="Misfit type")
    parser.add_argument('iteration_no', type=int,
                        help="Config filename")
    parser.add_argument('output_folder',
                        help="folder to output")
    parser.add_argument('kernelnames',
                        nargs="+",
                        help="""Kernels To Sum.
Add modifier after colon (e.g. Hessian2_kernel:absolute)""")

    args = parser.parse_args()
    conf = load_config(args.conf_file)
    for kernelname in tqdm(args.kernelnames, desc="Sum Kernels"):
        name, modifiers = get_kernelname_and_modifiers(kernelname)
        sum_kernels(name, conf, args.misfit_type, args.iteration_no,
                    args.output_folder, **modifiers)


if __name__ == '__main__':
    main()
