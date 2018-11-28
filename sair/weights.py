#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import argparse

import numpy as np

from . import utils
from . import io_utils as io
from .config import load_config

from collections import defaultdict

import matplotlib.pyplot as plt


class ReceiverWeights(object):
    def __init__(self, conf, events, measurement_type_cat_on=False):
        self.conf = conf
        self.events = conf.get_event_names()
        self.weights = defaultdict(lambda: defaultdict(lambda: 1))
        self.alphas = {}
        self.cat_weights = {}
        self.measurement_type_cat_on = measurement_type_cat_on
        if self.measurement_type_cat_on:
            self.measurement_types = ["single", "dd"]
        else:
            self.measurement_types = ["all"]
        if self.conf.adjoint.receiver_weights:
            self.load()

    def load(self):
        folder = io.Path(self.conf.adjoint.receiver_weights)
        for event in self.events:
            data = utils.load_json(folder / "{}.json".format(event))
            for station, weight in data.iteritems():
                comp = station[-1]
                self.weights[(event, comp, "all")][station] = weight["weight"]
                if self.conf.adjoint.use_receiver_weights_for_single:
                    self.weights[(event, comp, "single")][station] = weight["weight"]  # NOQA
                if self.conf.adjoint.use_receiver_weights_for_dd:
                    self.weights[(event, comp, "dd")][station] = weight["weight"]  # NOQA

    def find_norm_factors(self, dd_weights, single_windows):
        total_weights = defaultdict(int)
        windows = defaultdict(set)
        weight_single = self.conf.adjoint.weight_single
        weight_dd = self.conf.adjoint.weight_dd

        for event in self.events:
            for station in self.conf.get_stations(event, with_comps=True):
                comp = station.split(".")[-1][-1]

                if self.measurement_type_cat_on:
                    cat = (event, comp, "dd")
                else:
                    cat = (event, comp, "all")
                total_weights[cat] += 0  # create the category
                rec_weight = self.weights[cat][station]
                for no, weight in dd_weights.get_by_event(
                        event, station).iteritems():
                    windows[cat].add((station, no))
                    total_weights[cat] += rec_weight*weight_dd*weight

                if self.measurement_type_cat_on:
                    cat = (event, comp, "single")
                else:
                    cat = (event, comp, "all")
                total_weights[cat] += 0  # Create the category
                rec_weight = self.weights[cat][station]
                for no, weight in single_windows.get(
                        (event, station), {}).iteritems():
                    windows[cat].add((station, no))
                    total_weights[cat] += rec_weight*weight_single*weight

        ncat = len(total_weights)
        for cat, total_weight in total_weights.iteritems():
            if total_weight == 0:
                self.alphas[cat] = 0
                self.cat_weights[cat] = 0
            else:
                self.alphas[cat] = len(windows[cat])/total_weights[cat]
                self.alphas[cat] = 5.74948665298
                self.cat_weights[cat] = 1/ncat*1/len(windows[cat])

    def analyze(self, dd_weights, single_windows):
        ncat = len(self.cat_weights)
        for cat in self.cat_weights:
            if self.alphas[cat] == 0:
                continue
            print("Category:", cat)
            print("alpha:", self.alphas[cat])
            print("cat_weight:", self.cat_weights[cat])
            print("n_windows:", 1/self.cat_weights[cat]/ncat)
            total_weight = 0

            weight_single = self.conf.adjoint.weight_single
            weight_dd = self.conf.adjoint.weight_dd

            n_windows = 0
            for station in self.conf.get_stations(cat[0], with_comps=True):
                if not station.endswith(cat[1]):
                    continue
                rec_weight = self.get(cat, station)
                if cat[2] == "all" or cat[2] == "dd":
                    for no, weight in dd_weights.get_by_event(
                            cat[0], station).iteritems():
                        total_weight += rec_weight*weight_dd*weight
                        n_windows += 1

                if cat[2] == "all" or cat[2] == "single":
                    for no, weight in single_windows.get(
                            (cat[0], station), {}).iteritems():
                        total_weight += rec_weight*weight_single*weight
                        n_windows += 1
            print("Total weight:", total_weight)

    def get(self, cat, station, misfit_type=None):
        """Returns Normalized receiver weight"""
        if len(cat) == 2:
            if self.measurement_type_cat_on:
                cat = (cat[0], cat[1], misfit_type)
            else:
                cat = (cat[0], cat[1], "all")
        return self.cat_weights[cat]*self.alphas[cat]*self.get_raw(cat, station)  # NOQA

    def get_from_stationname(self, station, misfit_type):
        net, name, comp, event = station.split(".")
        return self.get((event, comp[-1]),
                        ".".join([net, name, comp]),
                        misfit_type)

    def get_raw(self, cat, station):
        """Returns receiver weights"""
        return self.weights[cat][station]


class DDWeights(object):
    def __init__(self, pairs={}, weights_on=True):
        self.pairs = pairs
        self.weights_on = weights_on
        self.weights = defaultdict(dict)
        self._prepare()

    def _prepare(self):
        for pair in self.pairs:
            sta_i, n_i = pair["window_id_i"].split(":")
            sta_j, n_j = pair["window_id_j"].split(":")
            if self.weights_on:
                self.weights[sta_i][n_i] = pair["weight_i"]
                self.weights[sta_j][n_j] = pair["weight_j"]
            else:
                self.weights[sta_i][n_i] = 1
                self.weights[sta_j][n_j] = 1

    def get(self, station):
        return self.weights[station]

    def get_by_event(self, event, station):
        return self.get(".".join([station, event]))


def distance(s1, s2):
    return np.sqrt(np.linalg.norm(s1-s2))


def distance_matrix(stations):
    station_names = stations.keys()
    nsta = len(station_names)
    distances = np.zeros((nsta, nsta))
    for i, sta_i in enumerate(station_names):
        for j, sta_j in enumerate(station_names[i+1:], i+1):
            d = np.linalg.norm(stations[sta_i]-stations[sta_j])
            distances[i, j] = d
            distances[j, i] = d
    return station_names, distances


def calc_dist_weights(distances, delta):
    nsta = distances.shape[0]
    weights = np.zeros(nsta)
    for i in range(nsta):
        weights[i] = np.sum(np.exp(-(distances[i, :] / delta)**2))
    return weights


def search_delta(distances, start, interval, ratio, drop,
                 plot=False):
    cur_delta = start
    cond_numbers = []
    deltas = []
    pbar = utils.tqdm(desc="Searching for delta")
    while True:
        w = calc_dist_weights(distances, cur_delta)
        cond_number = max(w)/min(w)
        if len(cond_numbers) > 1 and \
           cond_number < max(cond_numbers)*(1 - drop):
            break
        else:
            pbar.update()
            cond_numbers.append(cond_number)
            deltas.append(cur_delta)
            cur_delta += interval

    cond_numbers = np.array(cond_numbers)
    target_cond_number = (max(cond_numbers)-min(cond_numbers))*ratio
    print("target_cond_number: ", target_cond_number+min(cond_numbers))
    min_i = ((cond_numbers-target_cond_number-min(cond_numbers))**2).argmin()
    cond = cond_numbers[min_i]
    delta = deltas[min_i]
    print("Selected cond number:", cond)
    print("Selected delta:", delta)

    weights = calc_dist_weights(distances, delta)

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(deltas, cond_numbers, 20, color=None, edgecolor="blue")
        ax.plot(deltas, cond_numbers, "g-")
        ax.scatter([delta], [cond], 100, marker="*", color="red",
                   zorder=10)
        xlim = ax.get_xlim()
        ax.plot(xlim, [cond, cond], color="orange")

        ax.set_xlabel("Reference Distance")
        ax.set_ylabel("Condition Number")
        plt.show()

    return delta, weights


def calc_rec_weights():
    parser = argparse.ArgumentParser(
        description="Create receiver weights")
    parser.add_argument("conf_file")
    parser.add_argument("--water-level", "-w", type=float,
                        default=None)
    parser.add_argument("--search-ratio", "-s", type=float,
                        default=0.35)
    parser.add_argument("--search-start", "-b", type=float,
                        default=1)
    parser.add_argument("--search-inverval", "-i", type=float,
                        default=100)
    parser.add_argument("--search-stop-drop", "-d", type=float,
                        default=0.1)
    parser.add_argument("--plot-search", "-p", action="store_true")
    args = parser.parse_args()

    conf = load_config(args.conf_file)
    if conf.adjoint.receiver_weights is None:
        raise Exception("Adjoint receiver weights folder is not set.")
    weights_folder = io.Path(conf.adjoint.receiver_weights)

    events = conf.get_event_names()
    for event in events:
        names, distances = distance_matrix(conf.get_stations(event,
                                                             with_comps=False))
        if args.water_level:
            weights = calc_dist_weights(distances, args.water_level)
        else:
            delta, weights = search_delta(distances, args.search_start,
                                          args.search_inverval,
                                          args.search_ratio,
                                          args.search_stop_drop,
                                          args.plot_search)
        rec_weights = {}
        for i, name in enumerate(names):
            for comp in conf.simulation.comps:
                rec_weights["{}.{}".format(name, comp)] = {
                    "weight": 1.0/weights[i]
                }

        utils.write_json(weights_folder / "{}.json".format(event),
                         rec_weights)


def calc_src_weights():
    parser = argparse.ArgumentParser(
        description="Create source weights")
    parser.add_argument("conf_file")
    parser.add_argument("--water-level", "-w", type=float,
                        default=None)
    parser.add_argument("--search-ratio", "-s", type=float,
                        default=0.35)
    parser.add_argument("--search-start", "-b", type=float,
                        default=1)
    parser.add_argument("--search-inverval", "-i", type=float,
                        default=100)
    parser.add_argument("--search-stop-drop", "-d", type=float,
                        default=0.1)
    parser.add_argument("--plot-search", "-p", action="store_true")
    args = parser.parse_args()

    conf = load_config(args.conf_file)
    if conf.adjoint.source_weights is None:
        raise Exception("Adjoint source weights file is not set.")

    event_locs = conf.get_event_locs()
    events, distances = distance_matrix(event_locs)
    if args.water_level:
        weights = calc_dist_weights(distances, args.water_level)
    else:
        delta, weights = search_delta(distances,
                                      args.search_start,
                                      args.search_inverval,
                                      args.search_ratio,
                                      args.search_stop_drop,
                                      args.plot_search)

    print("Total Weight is", sum(weights))
    print("Number of events:", len(weights))
    fac = len(weights)/sum(weights)
    print("Normalization factor:", fac)
    weights = [fac*w for w in weights]
    print("Total weight after normalization:", sum(weights))
    print("Distance between max and min value:", max(weights)-min(weights))
    source_weights = {e: 1.0/weights[i] for i, e in enumerate(events)}
    utils.write_json(conf.adjoint.source_weights, source_weights)


def _plot_rec_weights(conf, event, dd_weights=False,
                      single_for_non_paired=False, single_for_all=False,
                      title="", is_global=False, output_file=False,
                      **kwargs):
    xs = []
    ys = []
    weights = []
    events = [event]
    r = ReceiverWeights(conf, events)
    normalized = dd_weights or single_for_non_paired or single_for_all
    pairs = {}
    single_windows = {}

    if dd_weights:
        from .dd_pair import read_pairs
        pairs = read_pairs(conf, events)

    single_stanames = []
    if single_for_non_paired:
        from .adjoint import read_pair_files
        _, _, _, _, single_stanames = read_pair_files(conf, events, pairs)

    if single_for_all:
        single_stanames = conf.get_stations(events, add_event_label=True)

    for staname in single_stanames:
        net, name, comp, ev = staname.split(".")
        staname = ".".join([net, name, comp])
        single_windows[(ev, staname)] = {"0": 1}

    dd_weights = DDWeights(pairs)
    r.find_norm_factors(dd_weights, single_windows)

    for sta, (x, y) in conf.get_stations(events).iteritems():
        if normalized:
            w = r.get((events[0], sta[-1]), sta)
        else:
            w = r.get_raw((events[0], sta[-1]), sta)
        xs.append(x)
        ys.append(y)
        d = dd_weights.get_by_event(events[0], sta)
        mw = 0
        if len(d) > 0:
            for v in d.values():
                mw += v
        for v in single_windows[(events[0], sta)].values():
            mw += v
        w = w*mw
        weights.append(w)

    return plot_weights(conf, xs, ys, weights, "Receiver Weights",
                        title=title, is_global=is_global,
                        output_file=output_file, **kwargs)


def plot_rec_weights():
    parser = argparse.ArgumentParser(
        description="Plot Receiver Weights")
    parser.add_argument("conf_file")
    parser.add_argument("event")
    parser.add_argument('-t', '--title', default="", type=str)
    parser.add_argument('-o', '--output-file',
                        default="", type=str)
    parser.add_argument('-g', '--is-global',
                        action="store_true", help="Global Model")
    parser.add_argument('-p', '--dd-weights',
                        action="store_true", help="Add DD weights")
    parser.add_argument('-c', '--single-for-non-paired',
                        action="store_true",
                        help="Consider SINGLE measurements for non-paired")
    parser.add_argument('-C', '--single-for-all',
                        action="store_true",
                        help="Consider single all stations")

    args = parser.parse_args()

    conf = load_config(args.conf_file)
    _plot_rec_weights(conf, args.event,
                      args.dd_weights, args.single_for_non_paired,
                      args.single_for_all,
                      args.title, args.is_global, args.output_file)


def _plot_src_weights(conf, title="",
                      is_global=False, output_file=None,
                      **kwargs):
    events = conf.get_event_names()
    event_locs = conf.get_event_locs()
    src_weights = conf.get_src_weights()
    xs = [event_locs[e][0] for e in events]
    ys = [event_locs[e][1] for e in events]
    weights = [src_weights[e] for e in events]

    return plot_weights(conf, xs, ys, weights, "Source Weights",
                        marker="*",
                        title=title,
                        is_global=is_global,
                        output_file=output_file,
                        **kwargs)


def plot_src_weights():
    parser = argparse.ArgumentParser(
        description="Plot Source Weights")
    parser.add_argument("conf_file")
    parser.add_argument('-t', '--title', default="", type=str)
    parser.add_argument('-o', '--output-file',
                        default="", type=str)
    parser.add_argument('-g', '--is-global',
                        action="store_true", help="Global Model")

    args = parser.parse_args()

    conf = load_config(args.conf_file)
    _plot_src_weights(conf, args.title,
                      args.is_global, args.output_file)


def _plot_dd_weights(conf, event, number_of_pairs=False,
                     title="", is_global=False, output_file=False,
                     **kwargs):
    from .dd_pair import read_pairs

    xs = []
    ys = []
    weights = []
    events = [event]
    pairs = read_pairs(conf, events)
    dd_weights = DDWeights(pairs)

    stations = conf.get_stations(events, add_event_label=True)
    for sta, (x, y) in stations.iteritems():
        xs.append(x)
        ys.append(y)
        try:
            weights.append(dd_weights.get(sta)["0"])
        except KeyError:
            weights.append(1)  # no pairs

    label = "DD Weights"
    if number_of_pairs:
        weights = [round(1.0/w) - 1 for w in weights]
        label = "Number of DD Pairs"

    return plot_weights(conf, xs, ys, weights, label,
                        title=title, is_global=is_global,
                        output_file=output_file, **kwargs)


def plot_dd_weights():
    parser = argparse.ArgumentParser(
        description="Plot DD Weights")
    parser.add_argument("conf_file")
    parser.add_argument("event")
    parser.add_argument('-t', '--title', default="", type=str)
    parser.add_argument('-o', '--output-file',
                        default="", type=str)
    parser.add_argument('-g', '--is-global',
                        action="store_true", help="Global Model")
    parser.add_argument('-n', '--number-of-pairs',
                        action="store_true",
                        help="Show the number of pairs")
    args = parser.parse_args()
    conf = load_config(args.conf_file)
    _plot_dd_weights(conf, args.event, args.number_of_pairs,
                     args.title, args.is_global, args.output_file)


def plot_weights(conf, xs, ys, weights, weights_label,
                 title="", is_global=False, marker="v",
                 output_file="", fig=None, ax=None,
                 **kwargs):

    from .plot import utils as plot_utils
    from .specfem_io import read_all_from_folder

    x = read_all_from_folder(conf.init_model, "x")
    y = read_all_from_folder(conf.init_model, "z")
    fig, ax = plot_utils.plot_bin(x, y, np.zeros(len(x)),
                                  show=False,
                                  colorbar=False,
                                  is_global=is_global,
                                  title=title,
                                  output_file="",
                                  cmap="gray_r",
                                  fig=fig, ax=ax,
                                  **kwargs)

    sc = ax.scatter(xs, ys, c=weights, marker=marker, cmap="jet")
    cbar = plt.colorbar(sc, fraction=0.024, pad=0.02, ax=ax)
    cbar.set_label(weights_label)

    if not output_file:
        plt.show()
    else:
        fig.savefig(output_file)

    return fig, ax
