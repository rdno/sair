#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import argparse

from .specfem_io import read_data_from_folder
from .specfem_io import write_data_to_folder


from . import config
from . import io_utils as io

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def main():
    parser = argparse.ArgumentParser(description='Model Update')
    parser.add_argument('conf_file')
    parser.add_argument('misfit_type')
    parser.add_argument('iteration', type=int)
    parser.add_argument('step_length', type=float,
                        help="Step Length parameter")

    args = parser.parse_args()

    conf = config.load_config(args.conf_file)
    alpha = args.step_length
    updates = {"rhop_update": "rho",
               "alpha_update": "vp",
               "beta_update": "vs"}

    misfit_type = args.misfit_type
    it = args.iteration
    old_name = conf.get_model_folder(misfit_type, it-1)
    new_name = conf.get_model_folder(misfit_type, it)
    update_folder = conf.get_update_folder(misfit_type, it)

    old_model = conf.get_model_folder(misfit_type, it-1)
    new_model = conf.get_model_folder(misfit_type, it)
    if rank == 0:
        io.copy_folder(old_model, new_model, clean=True)
    comm.barrier()

    for name in conf.simulation.kernels:
        uname = name + "_update"
        mname = updates[uname]
        print("Updating {}".format(mname))
        old_model = read_data_from_folder(old_name, mname, rank)
        d = read_data_from_folder(update_folder, uname, rank)

        # if conf.inversion.optimization == "L-BFGS" and it > 1:
        # new_model = old_model + alpha*d
        # else:
        new_model = old_model*np.exp(alpha*d)
        write_data_to_folder(new_model, new_name, mname, rank)


if __name__ == "__main__":
    main()
