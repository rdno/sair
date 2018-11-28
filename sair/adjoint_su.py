# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import numpy as np

import argparse

import pyadjoint
from pyadjoint.config import ConfigCrossCorrelation
from pyadjoint.config import ConfigExponentiatedPhase
from pyadjoint.config import ConfigWaveForm
from pyadjoint.config import ConfigDoubleDifferenceCrossCorrelation
from pyadjoint.config import ConfigDoubleDifferenceWaveForm


from functools import partial

from . import mpi
from . import utils
from . import config
from . import weights
from . import dd_pair

from .windows import WindowGetter
from .utils import tqdm

from collections import defaultdict

CALC_MISFIT = 1
CALC_ADJOINT = 2
CALC_BOTH = CALC_MISFIT | CALC_ADJOINT


def read_files(conf, events=None, stations=None):
    if stations is None:
        stations = conf.get_stations(events, with_comps=True,
                                     add_event_label=True)
        stations = []
        for event in events:
            for station in conf.get_stations(event, with_comps=True,
                                             add_event_label=True):
                stations.append(station)

    obsds = utils.SUReader(conf, events, "obs")
    synts = utils.SUReader(conf, events, "syn")
    windows = WindowGetter(conf)

    return stations, obsds, synts, windows


def read_pair_files(conf, events, all_pairs):
    stanames = {}
    for pair in all_pairs:
        sta_i = pair["window_id_i"].split(":")[0]
        sta_j = pair["window_id_j"].split(":")[0]
        stanames[sta_i] = sta_i
        stanames[sta_j] = sta_j

    single_stanames = []
    for staname in conf.get_stations(events, add_event_label=True,
                                     with_comps=True):
        if staname not in stanames:
            stanames[staname] = staname
            single_stanames.append(staname)

    obsds = utils.SUReader(conf, events, "obs")
    synts = utils.SUReader(conf, events, "syn")
    windows = WindowGetter(conf)

    return stanames.keys(), obsds, synts, windows, single_stanames


def single_misfit(filename,
                  obsds, synts, windows,
                  adjoint_misfit_type,
                  adjoint_config,
                  calc_type, conf,
                  rec_weights):
    """Compute an adjoint source/misfit for one trace"""

    obsd = obsds[filename]
    synt = synts[filename]
    adj_windows = windows[(filename, "single")]

    rec_weight = rec_weights.get_from_stationname(filename, "single")

    p = pyadjoint.calculate_adjoint_source(adjoint_misfit_type,
                                           obsd, synt,
                                           adjoint_config,
                                           adj_windows,
                                           adjoint_src=bool(calc_type & CALC_ADJOINT),  # NOQA
                                           plot=False)

    res = {filename: {}}
    weight = conf.adjoint.weight_single*rec_weight
    if calc_type & CALC_MISFIT:
        res[filename]["misfit"] = weight*p.misfit

    if calc_type & CALC_ADJOINT:
        res[filename]["src"] = weight*p.adjoint_source[::-1]

    return res


def calc_single(calc_type, events, conf,
                adjoint_misfit_type, adjoint_config,
                stations=None):
    stations, obsds, synts, windows = read_files(conf, events, stations)

    single_windows = {}
    for station in stations:
        net, name, comp, event = station.split(".")
        sta = ".".join([net, name, comp])
        wins = windows[(station, "single")]
        if len(wins) > 0:
            single_windows[(event, sta)] = {i: 1 for i in range(len(wins))}

    rec_weights = weights.ReceiverWeights(conf, events)
    rec_weights.find_norm_factors(weights.DDWeights(), single_windows)
    rec_weights.analyze(weights.DDWeights(), single_windows)

    calc = partial(single_misfit,
                   obsds=obsds,
                   synts=synts,
                   windows=windows,
                   calc_type=calc_type,
                   adjoint_misfit_type=adjoint_misfit_type,
                   adjoint_config=adjoint_config,
                   conf=conf,
                   rec_weights=rec_weights)

    results = mpi.run_with_mpi(calc, stations, title="Single")

    if calc_type & CALC_MISFIT:
        def write_misfits():
            filename = conf.get_misfit_folder() / "misfits"
            if len(events) == 1:
                filename += "_" + events[0]
            with open(filename, "w") as f:
                total_misfit = 0
                for sta in sorted(results.keys()):
                    misfit = results[sta]["misfit"]
                    f.write("{:100s} {:.15f}\n".format(sta, misfit))
                    total_misfit += misfit
                f.write("{:.15f}\n".format(total_misfit))

        mpi.run_on_main_rank(write_misfits)

    if calc_type & CALC_ADJOINT:
        def write_srcs():
            times = obsds[stations[0]].times()
            dt = times[1]-times[0]

            data = {}
            for sta in results.keys():
                data[sta] = results[sta]["src"]
            utils.write_su_adjoint(conf, data, dt, events)

        mpi.run_on_main_rank(write_srcs)


def calc_cc(calc_type, events, conf, stations=None):
    adj_config = ConfigCrossCorrelation(1/conf.adjoint.freq_max,
                                        1/conf.adjoint.freq_min,
                                        conf.adjoint.taper_type,
                                        conf.adjoint.taper_percentage)
    return calc_single(calc_type, events, conf,
                       "cc_traveltime_misfit", adj_config,
                       stations)


def calc_ep(calc_type, events, conf, stations=None):
    adj_config = ConfigExponentiatedPhase(1.0/conf.adjoint.freq_max,
                                          1.0/conf.adjoint.freq_min,
                                          conf.adjoint.taper_type,
                                          conf.adjoint.taper_percentage,
                                          wtr_env=0.2)
    return calc_single(calc_type, events, conf,
                       "exponentiated_phase_misfit", adj_config,
                       stations)


def calc_norm_waveform(calc_type, events, conf, stations=None):
    adj_config = ConfigWaveForm(1/conf.adjoint.freq_max,
                                1/conf.adjoint.freq_min,
                                conf.adjoint.taper_type,
                                conf.adjoint.taper_percentage)
    return calc_single(calc_type, events, conf,
                       "norm_waveform_misfit", adj_config,
                       stations)


def calc_waveform(calc_type, events, conf, stations=None):
    adj_config = ConfigWaveForm(1/conf.adjoint.freq_max,
                                1/conf.adjoint.freq_min,
                                conf.adjoint.taper_type,
                                conf.adjoint.taper_percentage)
    return calc_single(calc_type, events, conf,
                       "waveform_misfit", adj_config,
                       stations)


def single_dd_misfit(pair_id,
                     pairs, obsds, synts, windows,
                     adjoint_misfit_type,
                     adjoint_config,
                     calc_type, conf,
                     rec_weights, dd_weights):
    """Compute an DD adjoint source/misfit for one pair"""
    pair = pairs[pair_id]
    sta_i, n_i = pair["window_id_i"].split(":")
    sta_j, n_j = pair["window_id_j"].split(":")

    adj_i, adj_j = pyadjoint.calculate_adjoint_source_DD(
        adjoint_misfit_type,
        obsds[sta_i], synts[sta_i],
        obsds[sta_j], synts[sta_j],
        adjoint_config,
        [windows[sta_i][int(n_i)]],
        [windows[sta_j][int(n_j)]],
        plot=False,
        adjoint_src=bool(calc_type & CALC_ADJOINT))

    net_i, name_i, comp_i, ev_i = sta_i.split(".")
    net_j, name_j, comp_j, ev_j = sta_j.split(".")

    rw_i = rec_weights.get_from_stationname(sta_i, "dd")
    rw_j = rec_weights.get_from_stationname(sta_j, "dd")

    w_i = conf.adjoint.weight_dd*rw_i*dd_weights.get(sta_i)[n_i]
    w_j = conf.adjoint.weight_dd*rw_j*dd_weights.get(sta_j)[n_j]

    key_i = ":".join([sta_i, str(pair_id)])
    key_j = ":".join([sta_j, str(pair_id)])
    res = {key_i: {}, key_j: {}}

    if calc_type & CALC_MISFIT:
        res[key_i]["misfit"] = w_i*adj_i.misfit
        res[key_j]["misfit"] = w_j*adj_j.misfit

    if calc_type & CALC_ADJOINT:
        res[key_i]["src"] = w_i*adj_i.adjoint_source[::-1]
        res[key_j]["src"] = w_j*adj_j.adjoint_source[::-1]

    return res


def calc_dd(calc_type, events, conf,
            adjoint_misfit_type,
            adjoint_config):
    all_pairs = dd_pair.read_pairs(conf, events)
    stanames, obsds, synts, windows, _ = read_pair_files(conf, events,
                                                         all_pairs)

    dd_weights = weights.DDWeights(all_pairs,
                                   conf.dd.pair_wise_weighting)
    rec_weights = weights.ReceiverWeights(conf, events)
    rec_weights.find_norm_factors(dd_weights,
                                  single_windows={})
    if mpi.is_main_rank:
        rec_weights.analyze(dd_weights, {})

    calc = partial(single_dd_misfit,
                   pairs=all_pairs,
                   obsds=obsds,
                   synts=synts,
                   windows=windows,
                   adjoint_misfit_type=adjoint_misfit_type,
                   adjoint_config=adjoint_config,
                   calc_type=calc_type,
                   conf=conf,
                   dd_weights=dd_weights,
                   rec_weights=rec_weights)

    results = mpi.run_with_mpi(calc, list(range(len(all_pairs))),
                               title="DD")

    if calc_type & CALC_MISFIT:
        def write_misfits():
            total_misfit = 0
            # misfits = defaultdict(float)
            for stap, v in results.iteritems():
                sta, p = stap.split(":")
                net, sta, comp, event = sta.split(".")
                # TODO: Station based misfit
                # staname = ".".join([net, sta, comp])
                # misfits[staname] = v["misfit"]
                total_misfit += v["misfit"]
            filename = conf.get_misfit_folder() / "misfits"
            if len(events) == 1:
                filename += "_" + events[0]
            with open(filename, "w") as f:
                f.write("{:.15f}\n".format(total_misfit))
        mpi.comm.barrier()
        mpi.run_on_main_rank(write_misfits)

    if calc_type & CALC_ADJOINT:
        def write_srcs():
            times = obsds[stanames[0]].times()
            dt = times[1] - times[0]
            adjs = {sta: np.zeros(len(times)) for sta in stanames}

            # Sum over pairs
            for stap, adj in tqdm(results.iteritems()):
                sta, p = stap.split(":")
                adjs[sta] += adj["src"]

            utils.write_su_adjoint(conf, adjs, dt, events)

        mpi.comm.barrier()
        mpi.run_on_main_rank(write_srcs)


def calc_cc_dd(calc_type, events, conf):
    adjoint_config = ConfigDoubleDifferenceCrossCorrelation(
        1.0/conf.adjoint.freq_max, 1.0/conf.adjoint.freq_min,
        conf.adjoint.taper_type, conf.adjoint.taper_percentage)
    adjoint_misfit_type = "cc_traveltime_misfit_DD"
    return calc_dd(calc_type, events, conf,
                   adjoint_misfit_type, adjoint_config)


def calc_wf_dd(calc_type, events, conf):
    adjoint_config = ConfigDoubleDifferenceWaveForm(
        1.0/conf.adjoint.freq_max, 1.0/conf.adjoint.freq_min,
        conf.adjoint.taper_type, conf.adjoint.taper_percentage)
    adjoint_misfit_type = "waveform_misfit_DD"
    return calc_dd(calc_type, events, conf,
                   adjoint_misfit_type, adjoint_config)


def calc_combined(calc_type, events, conf,
                  single_misfit_type, single_adjoint_config,
                  dd_misfit_type, dd_adjoint_config,
                  compute_single_on_all=False):
    """Calculate Traditional and DD misfits together"""
    all_pairs = dd_pair.read_pairs(conf, events)
    stanames, obsds, synts, windows, single_stanames = read_pair_files(
        conf, events, all_pairs)

    windows.set_pairs(all_pairs)
    windows.allow_singles_to_use_paired_windows = compute_single_on_all

    dd_weights = weights.DDWeights(all_pairs,
                                   conf.dd.pair_wise_weighting)

    if compute_single_on_all:
        single_stanames = stanames

    single_windows = {}
    for staname in stanames:
        net, name, comp, ev = staname.split(".")
        sta = ".".join([net, name, comp])
        wins = windows[(staname, "single")]
        if len(wins) > 0:
            single_windows[(ev, sta)] = {i: 1 for i in range(len(wins))}

    rec_weights = weights.ReceiverWeights(
        conf, events,
        measurement_type_cat_on=compute_single_on_all)
    rec_weights.find_norm_factors(dd_weights,
                                  single_windows=single_windows)

    if mpi.is_main_rank:
        rec_weights.analyze(dd_weights, single_windows)

    calc_dd = partial(single_dd_misfit,
                      pairs=all_pairs,
                      obsds=obsds,
                      synts=synts,
                      windows=windows,
                      calc_type=calc_type,
                      adjoint_misfit_type=dd_misfit_type,
                      adjoint_config=dd_adjoint_config,
                      conf=conf,
                      dd_weights=dd_weights,
                      rec_weights=rec_weights)

    calc_single = partial(single_misfit,
                          obsds=obsds,
                          synts=synts,
                          windows=windows,
                          calc_type=calc_type,
                          adjoint_misfit_type=single_misfit_type,
                          adjoint_config=single_adjoint_config,
                          conf=conf,
                          rec_weights=rec_weights)

    results_dd = mpi.run_with_mpi(calc_dd, list((range(len(all_pairs)))),
                                  title="DD")

    results_single = mpi.run_with_mpi(calc_single, single_stanames,
                                      title="Single")

    if calc_type & CALC_ADJOINT:
        def write_srcs():
            times = obsds[stanames[0]].times()
            dt = times[1] - times[0]
            adjs = {sta: np.zeros(len(times)) for sta in stanames}

            # Sum over pairs
            for stap, adj in tqdm(results_dd.iteritems()):
                sta, p = stap.split(":")
                adjs[sta] += adj["src"]

            for sta, adj in tqdm(results_single.iteritems()):
                adjs[sta] += adj["src"]

            utils.write_su_adjoint(conf, adjs, dt, events)

        mpi.comm.barrier()
        mpi.run_on_main_rank(write_srcs)

    if calc_type & CALC_MISFIT:
        def write_misfits():
            total_single_misfit = 0
            total_dd_misfit = 0
            for stap, v in results_dd.iteritems():
                sta, p = stap.split(":")
                _, _, comp, event = sta.split(".")
                total_dd_misfit += v["misfit"]
            for sta, v in results_single.iteritems():
                _, _, comp, event = sta.split(".")
                total_single_misfit += v["misfit"]

            filename = conf.get_misfit_folder() / "misfits"
            if len(events) == 1:
                filename += "_" + events[0]
            total_misfit = total_single_misfit + total_dd_misfit
            with open(filename, "w") as f:
                f.write("Single: {:.15f}\n".format(total_single_misfit))
                f.write("DD: {:.15f}\n".format(total_dd_misfit))
                f.write("{:.15f}\n".format(total_misfit))
            print("Single: {:.15f}".format(total_single_misfit))
            print("DD: {:.15f}".format(total_dd_misfit))
            print("{:.15f}".format(total_misfit))

        mpi.comm.barrier()
        mpi.run_on_main_rank(write_misfits)


def calc_cc_plus_dd(calc_type, events, conf,
                    compute_single_on_all=False):
    single_adjoint_config = ConfigCrossCorrelation(
        1/conf.adjoint.freq_max, 1/conf.adjoint.freq_min,
        conf.adjoint.taper_type, conf.adjoint.taper_percentage)
    dd_adjoint_config = ConfigDoubleDifferenceCrossCorrelation(
        1.0/conf.adjoint.freq_max, 1.0/conf.adjoint.freq_min,
        conf.adjoint.taper_type, conf.adjoint.taper_percentage)
    return calc_combined(calc_type, events, conf,
                         "cc_traveltime_misfit", single_adjoint_config,
                         "cc_traveltime_misfit_DD", dd_adjoint_config,
                         compute_single_on_all)


def calc_wf_plus_dd(calc_type, events, conf,
                    compute_single_on_all=False):
    single_adjoint_config = ConfigWaveForm(1/conf.adjoint.freq_max,
                                           1/conf.adjoint.freq_min,
                                           conf.adjoint.taper_type,
                                           conf.adjoint.taper_percentage)
    dd_adjoint_config = ConfigDoubleDifferenceWaveForm(
        1.0/conf.adjoint.freq_max, 1.0/conf.adjoint.freq_min,
        conf.adjoint.taper_type, conf.adjoint.taper_percentage)
    return calc_combined(calc_type, events, conf,
                         "waveform_misfit", single_adjoint_config,
                         "waveform_misfit_DD", dd_adjoint_config,
                         compute_single_on_all)


def single_one(filename, obsds, synts, windows, calc_type, conf):
    from pyadjoint.utils import window_taper
    from scipy.integrate import simps

    # obsd = obsds[filename]
    synt = synts[filename]
    adj_windows = windows[filename]

    nlen_t = len(synt.data)

    misfit = 1
    p = np.zeros(nlen_t)
    q = np.zeros(nlen_t)
    dt = synt.stats.delta

    for left, right in adj_windows:
        left_sample = int(np.floor(left / dt)) + 1
        nlen = int(np.floor((right-left) / dt)) + 1
        right_sample = left_sample + nlen
        s = synt.data[left_sample:right_sample]
        window_taper(s,
                     taper_percentage=conf.adjoint.taper_percentage,
                     taper_type=conf.adjoint.taper_type)

        dsdt = np.gradient(s, dt)
        nnorm = simps(y=dsdt*dsdt, dx=dt)
        p[left_sample:right_sample] = misfit*dsdt/nnorm

        mnorm = simps(y=s*s, dx=dt)
        q[left_sample:right_sample] = misfit*s/mnorm

    res = {filename: {}}
    if calc_type & CALC_MISFIT:
        res[filename]["misfit"] = misfit

    if calc_type & CALC_ADJOINT:
        res[filename]["src"] = p

    return res


def calc_one(calc_type, events, conf, filenames=None):
    stations, obsds, synts, windows = read_files(conf, events, filenames)

    calc = partial(single_one,
                   obsds=obsds,
                   synts=synts,
                   windows=windows,
                   calc_type=calc_type,
                   conf=conf)

    results = mpi.run_with_mpi(calc, stations, title="One")

    if calc_type & CALC_MISFIT:
        def write_misfits():
            filename = conf.get_misfit_folder() / "misfits"
            if len(events) == 1:
                filename += "_" + events[0]
            with open(filename, "w") as f:
                total_misfit = 0
                for sta in sorted(results.keys()):
                    misfit = results[sta]["misfit"]
                    f.write("{:100s} {:.15f}\n".format(sta, misfit))
                    total_misfit += misfit
                f.write("{:.15f}\n".format(total_misfit))

        mpi.run_on_main_rank(write_misfits)

    if calc_type & CALC_ADJOINT:
        def write_srcs():
            times = obsds[stations[0]].times()
            dt = times[1]-times[0]

            data = {}
            for sta in results.keys():
                data[sta] = results[sta]["src"]
            utils.write_su_adjoint(conf, data, dt, events)

        mpi.run_on_main_rank(write_srcs)


adjs = {
    "cc": calc_cc,
    "waveform": calc_waveform,
    "dd": calc_cc_dd,
    "waveform_dd": calc_wf_dd,
    "cc+dd": calc_cc_plus_dd,
    "cc_and_dd": partial(calc_cc_plus_dd, compute_single_on_all=True),
    "one": calc_one,
    "ep": calc_ep,
    "norm_wf": calc_norm_waveform,
    "wf+dd": calc_wf_plus_dd,
    "wf_and_dd": partial(calc_wf_plus_dd, compute_single_on_all=True),
}
calc_types = {
    "adj": CALC_ADJOINT,
    "misfit": CALC_MISFIT,
    "both": CALC_BOTH
}


def main():
    parser = argparse.ArgumentParser(description='Calculate Adjoint')
    parser.add_argument('adj_type', choices=adjs.keys())
    parser.add_argument('calc_type', choices=calc_types.keys())
    parser.add_argument('config_file', help="Config file")
    parser.add_argument('event', nargs="?", type=str, default=None)

    args = parser.parse_args()

    conf = config.load_config(args.config_file)

    if args.event is None:
        events = conf.get_event_names()
    else:
        events = [args.event]

    adjs[args.adj_type](calc_types[args.calc_type],
                        events,
                        conf)


if __name__ == '__main__':
    main()
