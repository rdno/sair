#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


from ..config import load_config
from ..specfem_io import read_all_from_folder
from ..specfem_io import read_all_with_coord
from . import utils

import numpy as np
import argparse


def plot_improvement(x, y, data_target, data_init, data_model,
                     nproc=None, conf=None, level=None,
                     plot_stations=False, plot_pairs=False,
                     title="", output_file="", cmap=None,
                     is_global=False, **kwargs):
    if cmap is None:
        cmap = utils.read_colormap_file("red_green")

    pert_init = np.log(data_init/data_target)
    pert_model = np.log(data_model/data_target)
    pert = np.abs(pert_init) - np.abs(pert_model)

    if level is None:
        level = max(abs(pert))

    return utils.plot_bin(x, y, pert, conf,
                          plot_stations_on=plot_stations,
                          plot_pairs_on=plot_pairs,
                          title=title, output_file=output_file,
                          vmax=-level, vmin=level,
                          cmap=cmap,
                          colorlabel="abs(ln(init/target)) - abs(ln(model/target))",  # NOQA
                          is_global=is_global,
                          **kwargs)


def read_and_plot_improvement(target_folder, init_folder, model_folder,
                              model_parameter,
                              nproc=None, conf=None, level=None,
                              plot_stations=False, plot_pairs=False,
                              title="", output_file="", cmap=None,
                              is_global=False, **kwargs):
    x, y, data_target = read_all_with_coord(target_folder, model_parameter,
                                            nproc)
    data_model = read_all_from_folder(model_folder, model_parameter, nproc)
    data_init = read_all_from_folder(init_folder, model_parameter, nproc)

    plot_improvement(x, y, data_target, data_init, data_model,
                     nproc, conf, level,
                     plot_stations, plot_pairs,
                     title, output_file, cmap,
                     is_global, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description='Plot perturbation')
    parser.add_argument('data_target_folder')
    parser.add_argument('data_init_folder')
    parser.add_argument('data_model_folder')
    parser.add_argument('argument', choices=("vp", "vs", "rho"))
    parser.add_argument('nproc', type=int, default=None, nargs="?")
    parser.add_argument('-c', '--config-file', default=None,
                        help="Config file")
    parser.add_argument('-l', '--plot-level',
                        default=0.7, type=float)
    parser.add_argument('-s', '--stations',
                        action="store_true")
    parser.add_argument('-p', '--pairs',
                        action="store_true")
    parser.add_argument('-o', '--output-file',
                        default="",
                        type=str)
    parser.add_argument('-t', '--title',
                        default="",
                        type=str)
    parser.add_argument('-g', '--is-global',
                        action="store_true", help="Global Model")

    args = parser.parse_args()

    conf = None
    if args.config_file is not None:
        conf = load_config(args.config_file)

    read_and_plot_improvement(args.data_target_folder, args.data_init_folder,
                              args.data_model_folder, args.argument,
                              args.nproc,
                              conf, args.plot_level,
                              args.stations, args.pairs,
                              args.title, args.output_file,
                              args.is_global)


if __name__ == "__main__":
    main()
