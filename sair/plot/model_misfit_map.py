#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


from ..specfem_io import read_all_from_folder
from ..specfem_io import read_all_with_coord
from . import utils

import numpy as np
import argparse


def rms(data_one, data_two):
    a = np.sqrt(np.sum((data_one-data_two)**2)/len(data_one))
    return a


def get_model_misfit_map_values(x, y, data_model, data_target,
                                chunk_size_x=None, chunk_size_z=None):
    if chunk_size_x is None or chunk_size_z is None:
        data = np.sqrt((data_target-data_model)**2)
    else:
        data = np.zeros_like(data_model)
        left, right = min(x), max(x)
        bottom, top = min(y), max(y)
        xi = np.arange(left, right+chunk_size_x, chunk_size_x)
        yi = np.arange(bottom, top+chunk_size_z, chunk_size_z)
        for i in range(len(xi)-1):
            for j in range(len(yi)-1):
                test_x = np.logical_and(x >= xi[i], x <= xi[i+1])
                test_y = np.logical_and(y >= yi[j], y <= yi[j+1])
                test = np.logical_and(test_x, test_y)
                data[test] = rms(data_model[test], data_target[test])
    return data


def plot_model_misfit_map(x, y, data_model, data_target,
                          normalized=True, logarithmic=True,
                          limits=None,
                          nproc=None,
                          chunk_size_x=None,
                          chunk_size_z=None,
                          plot_stations=False, plot_pairs=False,
                          title="", output_file="", cmap="gray_r",
                          colorlabel="Model Misfit",
                          is_global=False, **kwargs):

    data = get_model_misfit_map_values(x, y, data_model, data_target,
                                       chunk_size_x, chunk_size_z)

    if logarithmic:
        data = np.log(data)

    if normalized:
        data /= max(data)

    if limits is None:
        if normalized:
            limits = [0, 1]
        else:
            limits = [min(data), max(data)]

    return utils.plot_bin(x, y, data,
                          plot_stations_on=plot_stations,
                          plot_pairs_on=plot_pairs,
                          title=title, output_file=output_file,
                          vmax=limits[0], vmin=limits[1],
                          cmap=cmap,
                          colorlabel=colorlabel,
                          is_global=is_global,
                          **kwargs)


def read_and_plot_model_misfit_map(model_folder, target_folder,
                                   model_paramater,
                                   normalized=True, logarithmic=True,
                                   limits=None,
                                   nproc=None,
                                   chunk_size_x=None,
                                   chunk_size_z=None,
                                   plot_stations=False, plot_pairs=False,
                                   title="", output_file="", cmap="gray_r",
                                   colorlabel="Model Misfit",
                                   is_global=False, **kwargs):

    x, y, data_model = read_all_with_coord(model_folder,
                                           model_paramater,
                                           nproc)

    data_target = read_all_from_folder(target_folder,
                                       model_paramater,
                                       nproc)

    return plot_model_misfit_map(x, y, data_model, data_target,
                                 normalized, logarithmic, limits, nproc,
                                 chunk_size_x, chunk_size_z,
                                 plot_stations, plot_pairs,
                                 title, output_file, cmap,
                                 colorlabel, is_global, **kwargs)
