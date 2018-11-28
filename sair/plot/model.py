#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from ..config import load_config
from ..specfem_io import read_all_with_coord
from . import utils

import numpy as np
import argparse


def plot_model(x, y, data, nproc=None,
               conf=None, config_file=None, limits=None,
               plot_stations=False, plot_pairs=False,
               title="", output_file="",
               is_global=False,
               **kwargs):
    if config_file is not None:
        conf = load_config(config_file)

    total_max = np.max(data)
    total_min = np.min(data)

    if limits is not None:
        total_max = max(limits)
        total_min = min(limits)

    return utils.plot_bin(x, y, data, conf,
                          plot_stations_on=plot_stations,
                          plot_pairs_on=plot_pairs,
                          title=title, output_file=output_file,
                          vmax=total_max, vmin=total_min,
                          colorlabel="S-Wave speed (m/s)",
                          is_global=is_global,
                          **kwargs)


def read_and_plot_model(data_folder, model_param,
                        nproc=None,
                        conf=None, config_file=None, limits=None,
                        plot_stations=False, plot_pairs=False,
                        title="", output_file="",
                        is_global=False,
                        **kwargs):

    x, y, data = read_all_with_coord(data_folder, model_param, nproc)
    return plot_model(x, y, data, nproc,
                      conf, config_file, limits,
                      plot_stations, plot_pairs,
                      title, output_file,
                      is_global, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description='Plot the model')
    parser.add_argument('data_folder')
    parser.add_argument('argument', choices=("vp", "vs", "rho"))
    parser.add_argument('nproc', type=int, default=None, nargs="?")
    parser.add_argument('-c', '--config-file', default=None,
                        help="Config file")
    parser.add_argument('-s', '--stations',
                        action="store_true")
    parser.add_argument('-p', '--pairs',
                        action="store_true")
    parser.add_argument('-o', '--output-file',
                        default="",
                        type=str)
    parser.add_argument('-l', '--limits', nargs=2,
                        default=None, type=float)
    parser.add_argument('-t', '--title',
                        default="",
                        type=str)
    parser.add_argument('-g', '--is-global',
                        action="store_true", help="Global Model")
    args = parser.parse_args()

    read_and_plot_model(args.data_folder, args.argument, args.nproc,
                        config_file=args.config_file,
                        limits=args.limits,
                        plot_stations=args.stations,
                        plot_pairs=args.pairs,
                        title=args.title,
                        output_file=args.output_file,
                        is_global=args.is_global)


if __name__ == "__main__":
    main()
