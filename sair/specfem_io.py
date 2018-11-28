# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import glob
import os
import numpy as np
from scipy.io import FortranFile


def read_data(filename):
    nbytes = os.path.getsize(filename)
    with open(filename, 'rb') as f:
        f.seek(0)
        n = np.fromfile(f, dtype='int32', count=1)[0]

        if n == nbytes-8:
            f.seek(4)
            data = np.fromfile(f, dtype='float32')
            return data[:-1]
        else:
            f.seek(0)
            data = np.fromfile(f, dtype='float32')
    return data


def write_data(data, filename):
    n = np.array([4*len(data)], dtype='int32')
    data = np.array(data, dtype='float32')

    with open(filename, 'wb') as f:
        n.tofile(f)
        data.tofile(f)
        n.tofile(f)


def read_data_from_folder(folder, param, rank=0):
    return read_data(data_filename(folder, param, rank))


def write_data_to_folder(data, folder, param, rank=0):
    return write_data(data, data_filename(folder, param, rank))


def data_filename(folder, param, rank=0):
    return os.path.join(folder,
                        "proc{:06d}_{}.bin".format(rank, param))


def get_nproc(folder):
    return len(glob.glob(os.path.join(folder, "proc*_x.bin")))


def read_all_from_folder(folder, param, nproc=None):
    if nproc is None:
        nproc = get_nproc(folder)
    if nproc == 0:
        raise Exception("folder does not have data: {}".format(folder))

    for iproc in range(nproc):
        if iproc == 0:
            data = read_data_from_folder(folder, param, iproc)
        else:
            new_data = read_data_from_folder(folder, param, iproc)
            data = np.append(data, new_data, axis=0)
    return data


def read_all_with_coord(folder, param, nproc=None):
    x = read_all_from_folder(folder, "x", nproc)
    z = read_all_from_folder(folder, "z", nproc)
    data = read_all_from_folder(folder, param, nproc)
    return x, z, data


def read_ibool_from_folder(folder, rank=0, ngll=5):
    filename = data_filename(folder, "NSPEC_ibool", rank)
    f = FortranFile(filename, "r")
    nspec = f.read_ints()[0]
    d = f.read_record("({},{},{})i4".format(nspec, ngll, ngll))
    f.close()
    return d


def read_data_matrix_from_folder(folder, param, shape, rank=0):
    filename = data_filename(folder, param, rank)
    f = FortranFile(filename, "r")
    # nspec = f.read_ints()[0]
    d = f.read_record("({},{},{})f4".format(*shape))
    f.close()
    return d


def get_glj_points(n=5):
    if n != 5:
        raise NotImplementedError
    return np.array([-1., -0.65465367,  0., 0.65465367,  1])


def get_glj_weights(n=5):
    if n != 5:
        raise NotImplementedError
    return np.array([0.1, 0.54444444, 0.71111111, 0.54444444, 0.1])
