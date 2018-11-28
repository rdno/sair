#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from ..config import load_config
from ..specfem_io import read_all_with_coord
from . import utils

import numpy as np
import argparse


def plot_kernel(x, y, data, nproc=None,
                conf=None, config_file=None, level=0.1,
                max_value=None,
                plot_stations=False, plot_pairs=False,
                title="", output_file="",
                is_global=False,
                is_positive=False,
                **kwargs):

    if config_file is not None:
        conf = load_config(config_file)

    if max_value is None:
        max_value = np.max(abs(data))
    plot_data = data / max_value

    return utils.plot_bin(x, y, plot_data, conf,
                          plot_stations_on=plot_stations,
                          plot_pairs_on=plot_pairs,
                          title=title, output_file=output_file,
                          vmax=level,
                          vmin=-level if not is_positive else 0,
                          colorlabel="Normalized kernel",
                          is_global=is_global,
                          **kwargs)


def read_and_plot_kernel(data_folder, kernelname, nproc=None,
                         conf=None, config_file=None, level=0.1,
                         max_value=None,
                         plot_stations=False, plot_pairs=False,
                         title="", output_file="",
                         is_global=False,
                         is_positive=False,
                         **kwargs):

    x, y, data = read_all_with_coord(data_folder, kernelname, nproc)

    return plot_kernel(x, y, data, nproc,
                       conf, config_file, level,
                       max_value, plot_stations, plot_pairs,
                       title, output_file, is_global, is_positive,
                       **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description='Plot the model')
    parser.add_argument('data_folder')
    parser.add_argument('kernelname')
    parser.add_argument('nproc', type=int, default=None, nargs="?")
    parser.add_argument('-l', '--plot-level',
                        default=0.1, type=float)
    parser.add_argument('-c', '--config-file', default=None,
                        help="Config file")
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
    parser.add_argument('-P', '--is-positive',
                        action="store_true", help="Kernel values are positive")

    args = parser.parse_args()

    read_and_plot_kernel(args.data_folder, args.kernelname, args.nproc,
                         config_file=args.config_file,
                         level=args.plot_level,
                         plot_stations=args.stations, plot_pairs=args.pairs,
                         title=args.title, output_file=args.output_file,
                         is_global=args.is_global,
                         is_positive=args.is_positive)


if __name__ == "__main__":
    main()
