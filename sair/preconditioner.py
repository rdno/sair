#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import sys

import numpy as np
import argparse

from .specfem_io import read_data_from_folder
from .specfem_io import write_data_to_folder
from .config import load_config

from .utils import tqdm
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_precond(folder, preconditioners):
    precond = read_data_from_folder(folder, preconditioners[0], rank)
    for p in preconditioners[1:]:
        precond += read_data_from_folder(folder, p, rank)
    return precond


def precond_kernel(conf, kernel_folder, kernelname=None, kerneldata=None):
    preconditioners = [p+"_kernel"
                       for p in conf.postprocessing.preconditioners]
    if conf.postprocessing.smooth:
        preconditioners = [p+"_smooth" for p in preconditioners]

    precond = get_precond(kernel_folder, preconditioners)
    precondmax = comm.allreduce(np.max(abs(precond)), op=MPI.MAX)
    norm_precond = precond/precondmax
    wtr = conf.postprocessing.precond_wtr
    norm_precond[norm_precond < wtr] += wtr

    if kerneldata is not None:
        kname = ""
        kernel = kerneldata
    elif kernelname is not None:
        kname = kernelname + "_kernel"
        if conf.postprocessing.smooth:
            kname = kname + "_smooth"
        kernel = read_data_from_folder(kernel_folder, kname, rank)
    else:
        raise Exception("Kernel information is not given.")
    return kname+"_precond", kernel / precond


def main():
    parser = argparse.ArgumentParser(description='Generate Model Update')
    parser.add_argument('kernel_folder',
                        help="Folder that contains the kernels")
    parser.add_argument('conf_file',
                        help="Config file")
    args = parser.parse_args()

    conf = load_config(args.conf_file)

    if conf.postprocessing.precondition:
        for kernel in tqdm(conf.simulation.kernels, desc="Precondition"):
            kname, p_kernel = precond_kernel(conf, args.kernel_folder,
                                             kernelname=kernel)
            write_data_to_folder(p_kernel, args.kernel_folder, kname, rank)
    else:
        print("Preconditioning is off.")


if __name__ == '__main__':
    main()
