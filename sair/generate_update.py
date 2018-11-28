#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import argparse

from .specfem_io import read_data_from_folder
from .specfem_io import write_data_to_folder
from .specfem_io import read_all_from_folder

from . import io_utils as io
from . import config
from . import preconditioner

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_beta(g_new, g_old=None):
    """Polak-Ribiere"""
    if g_old is None:
        return 0
    top = np.sum(g_new*(g_new-g_old))
    bot = np.sum(g_old*g_old)
    return top/bot


def nlcg(conf, d, misfit_type, kernel_folder, iteration, kname, uname):
    """Non-linear Conjugate Gradient"""
    old_kernel_folder = conf.root_work_dir / "kernels_{}_{}".format(
        misfit_type, iteration-1)
    old_update = conf.root_work_dir / "update_{}_{}".format(
        misfit_type, iteration-1)
    d_old = read_data_from_folder(old_update, uname, rank)
    new_misfit = read_all_from_folder(kernel_folder, kname)
    new_misfit /= comm.allreduce(np.max(np.abs(new_misfit)), op=MPI.MAX)
    old_misfit = read_all_from_folder(old_kernel_folder, kname)
    old_misfit /= comm.allreduce(np.max(np.abs(old_misfit)), op=MPI.MAX)

    guard = np.abs(np.dot(new_misfit, old_misfit) / np.dot(new_misfit, new_misfit))  # NOQA
    if guard > conf.inversion.loss_of_conjugacy:
        if rank == 0:
            print("Loss of conjugacy {:.5f}.\n".format(guard))
        else:
            if rank == 0:
                print("Conjugacy is Fine {:.5f}.\n".format(guard))
            beta = get_beta(new_misfit, old_misfit)
            d += beta*d_old
    return d


def mpi_dot(a, b):
    return comm.allreduce(np.dot(a, b), MPI.SUM)


def lbfgs(conf, misfit_type, kernel_folder, iteration,
          kname, uname, model_name):
    cur_model_folder = conf.root_work_dir / "models" / "DATA_{}_{}".format(  # NOQA
        misfit_type, iteration - 1)
    old_model_folder = conf.root_work_dir / "models" / "DATA_{}_{}".format(  # NOQA
        misfit_type, iteration - 2)
    old_kernel_folder = conf.root_work_dir / "kernels_{}_{}".format(
        misfit_type, iteration-1)
    lbfgs_folder = conf.root_work_dir / "LBFGS"

    if rank == 0:
        io.makedir(lbfgs_folder)

    m = conf.inversion.lbfgs_memory

    info_file = lbfgs_folder / "info.npy"
    memory_used = 0
    if io.is_a_file(info_file):
        memory_used, = np.load(info_file)

    cur_model = read_data_from_folder(cur_model_folder, model_name,
                                      rank)
    old_model = read_data_from_folder(old_model_folder, model_name,
                                      rank)

    s = np.zeros((m, len(cur_model)))
    s_file = lbfgs_folder / "proc{:06d}_S.npy".format(rank)
    if io.is_a_file(s_file):
        s = np.load(s_file)

    y = np.zeros((m, len(cur_model)))
    y_file = lbfgs_folder / "proc{:06d}_Y.npy".format(rank)
    if io.is_a_file(y_file):
        y = np.load(y_file)

    s[1:, :] = s[:-1, :]
    s[0, :] = cur_model - old_model

    new_misfit = read_data_from_folder(kernel_folder, kname, rank)
    old_misfit = read_data_from_folder(old_kernel_folder, kname, rank)

    y[1:, :] = y[:-1, :]
    y[0, :] = new_misfit - old_misfit

    alpha = np.zeros(m)
    rho = np.zeros(m)
    q = new_misfit

    memory_used = min([memory_used + 1, m])

    for i in range(memory_used):
        rho[i] = 1 / mpi_dot(y[i, :], s[i, :])
        alpha[i] = rho[i]*mpi_dot(s[i, :], q)
        q = q - alpha[i]*y[i, :]

    if conf.postprocessing.precondition:
        _, q = preconditioner.precond_kernel(conf, kernel_folder, kerneldata=q)

    top = mpi_dot(y[0, :], s[0, :])
    bot = mpi_dot(y[0, :], y[0, :])

    r = top / bot * q

    for i in range(memory_used-1, -1, -1):
        beta = rho[i]*mpi_dot(y[i, :], r)
        r = r + s[i, :] * (alpha[i] - beta)

    d = -r
    d /= comm.allreduce(np.max(np.abs(d)), MPI.MAX)

    if mpi_dot(new_misfit, d) / mpi_dot(new_misfit, new_misfit) > 0:
        print("restarting L-BFGS")
        memory_used = 0
        d = -new_misfit
        d /= comm.allreduce(np.max(np.abs(d)), MPI.MAX)
        s = np.zeros((m, len(cur_model)))
        y = np.zeros((m, len(cur_model)))

    np.save(s_file, s)
    np.save(y_file, y)
    np.save(info_file, np.array([memory_used]))

    return d


def main():
    parser = argparse.ArgumentParser(description='Generate Model Update')
    parser.add_argument('conf_file',
                        help="Config file")
    parser.add_argument('misfit_type',
                        help="Misfit Type")
    parser.add_argument('iteration', type=int,
                        help="Iteration Number")

    args = parser.parse_args()
    conf = config.load_config(args.conf_file)
    kernel_folder = conf.root_work_dir / "kernels_{}_{}".format(
        args.misfit_type, args.iteration)
    output_folder = conf.root_work_dir / "update_{}_{}".format(
        args.misfit_type, args.iteration)

    params = conf.simulation.kernels
    for param in params:
        kname = param + "_kernel"
        uname = param + "_update"
        if conf.postprocessing.smooth:
            kname += "_smooth"
        if conf.postprocessing.precondition:
            kname += "_precond"

        # Steepest Descent
        misfit = read_data_from_folder(kernel_folder, kname, rank)
        misfit /= comm.allreduce(np.max(np.abs(misfit)), op=MPI.MAX)
        d = -1*misfit

        if args.iteration > 1:
            if conf.inversion.optimization == "NLCG":
                d = nlcg(conf, d, args.misfit_type,
                         kernel_folder,
                         args.iteration,
                         kname, uname)
            if conf.inversion.optimization == "L-BFGS":
                # Use the not preconditioned kernel
                kname = kname.replace("_precond", "")
                d = lbfgs(conf, args.misfit_type,
                          kernel_folder, args.iteration,
                          kname, uname,
                          conf.get_model_name(param))

        write_data_to_folder(d, output_folder, uname, rank)


if __name__ == '__main__':
    main()
