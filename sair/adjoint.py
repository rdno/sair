# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import numpy as np

import argparse
import sys

import pyadjoint
from pyadjoint.config import ConfigCrossCorrelation
from pyadjoint.config import ConfigDoubleDifferenceCrossCorrelation


from functools import partial

from . import mpi
from . import utils
from . import config
from . import weights
from . import dd_pair

from .windows import WindowGetter
from .utils import tqdm


CALC_MISFIT = 1
CALC_ADJOINT = 2
CALC_BOTH = CALC_MISFIT | CALC_ADJOINT


def read_files(conf, events=None, filenames=None):
    if filenames is None:
        filenames = []
        for event in events:
            for station in conf.get_stations(event):
                filenames.append(conf.get_trace_filename(event, station,
                                                         "obs"))

    obsd_filenames = dict([(f, f) for f in filenames])
    synt_filenames = dict([(f, f.replace("_obs", "_syn")) for f in filenames])

    obsds = utils.AsciiReader(obsd_filenames)
    synts = utils.AsciiReader(synt_filenames)
    windows = WindowGetter(obsd_filenames, conf=conf)

    return filenames, obsds, synts, windows


def read_pair_files(conf, events, all_pairs):
    filenames = {}
    synt_filenames = {}
    windows = {}
    for pair in all_pairs:
        sta_i, filename_i = conf.get_pair_filename(pair["window_id_i"])
        sta_j, filename_j = conf.get_pair_filename(pair["window_id_j"])
        for s, f in [(sta_i, filename_i), (sta_j, filename_j)]:
            if f not in filenames:
                filenames[s] = f
                synt_filenames[s] = f.replace("_obs", "_syn")

    cc_stanames = []
    for staname in conf.get_stations(events, add_event_label=True,
                                     with_comps=True):
        if staname not in filenames:
            _, fname = conf.get_pair_filename(staname)
            filenames[staname] = fname
            synt_filenames[staname] = fname.replace("_obs", "_syn")
            cc_stanames.append(staname)

    obsds = utils.AsciiReader(filenames)
    synts = utils.AsciiReader(synt_filenames)
    windows = WindowGetter(filenames, conf)

    return filenames, obsds, synts, windows, cc_stanames


def single_cc(filename, obsds, synts, windows,
              calc_type,
              conf,
              rec_weights,
              write_now=False):
    """Compute CC Adjoint for one trace
    """
    config = ConfigCrossCorrelation(conf.adjoint.freq_min,
                                    conf.adjoint.freq_max,
                                    conf.adjoint.taper_type,
                                    conf.adjoint.taper_percentage)
    obsd = obsds[filename]
    synt = synts[filename]
    adj_windows = windows[filename]

    try:
        info = conf.get_info_from_filename(filename)
        sta = info["station"]
        cat = (info["event"], sta[-1])
    except AttributeError:
        net, name, comp, ev = filename.split(".")
        sta = ".".join([net, name, comp])
        cat = (ev, comp[-1])
    rec_weight = rec_weights.get(cat, sta)

    p = pyadjoint.calculate_adjoint_source("cc_traveltime_misfit",
                                           obsd, synt,
                                           config, adj_windows,
                                           adjoint_src=bool(calc_type & CALC_ADJOINT),  # NOQA
                                           plot=False)
    try:
        _, event, _, basefilename = filename.split("/")
        net, sta, _, _ = basefilename.split(".")
    except ValueError:
        net, sta, _, event = filename.split(".")

    res = {filename: {}}
    weight = conf.adjoint.weight_cc*rec_weight
    if calc_type & CALC_MISFIT:
        res[filename]["misfit"] = weight*p.misfit

    if calc_type & CALC_ADJOINT:
        times = obsd.times()
        adj_data = np.zeros((len(times), 2))
        adj_data[:, 0] = times
        adj_data[:, 1] = weight*p.adjoint_source[::-1]
        if write_now:
            np.savetxt(
                filename.replace("DATA_obs", "SEM").replace("semd", "adj"),
                adj_data)
        else:
            res[filename]["src"] = adj_data[:, 1]

    return res


def calc_cc(calc_type, events, conf, filenames=None):
    filenames, obsds, synts, windows = read_files(conf, events, filenames)

    cc_windows = {}
    for filename in filenames:
        info = conf.get_info_from_filename(filename)
        # Assumes there is only one window
        cc_windows[(info["event"], info["station"])] = {"0": 1}

    rec_weights = weights.ReceiverWeights(conf, events)
    rec_weights.find_norm_factors(weights.DDWeights(), cc_windows)

    calc = partial(single_cc,
                   obsds=obsds,
                   synts=synts,
                   windows=windows,
                   calc_type=calc_type,
                   conf=conf,
                   rec_weights=rec_weights,
                   write_now=True)

    results = mpi.run_with_mpi(calc, filenames, title="CC")

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


def single_cc_dd(pair_id, pairs, obsds, synts, windows,
                 calc_type, conf,
                 rec_weights, dd_weights):

    config = ConfigDoubleDifferenceCrossCorrelation(
        conf.adjoint.freq_min, conf.adjoint.freq_max,
        conf.adjoint.taper_type, conf.adjoint.taper_percentage)
    pair = pairs[pair_id]
    sta_i, n_i = pair["window_id_i"].split(":")
    sta_j, n_j = pair["window_id_j"].split(":")

    adj_i, adj_j = pyadjoint.calculate_adjoint_source_DD(
        "cc_traveltime_misfit_DD",
        obsds[sta_i], synts[sta_i],
        obsds[sta_j], synts[sta_j],
        config, windows[sta_i], windows[sta_j],
        plot=False,
        adjoint_src=bool(calc_type & CALC_ADJOINT))

    net_i, name_i, comp_i, ev_i = sta_i.split(".")
    net_j, name_j, comp_j, ev_j = sta_j.split(".")
    cat_i = (ev_i, comp_i[-1])
    cat_j = (ev_j, comp_j[-1])

    rw_i = rec_weights.get(cat_i, ".".join([net_i, name_i, comp_i]))
    rw_j = rec_weights.get(cat_j, ".".join([net_j, name_j, comp_j]))

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


def calc_cc_dd(calc_type, events, conf):
    all_pairs = dd_pair.read_pairs(conf, events)
    filenames, obsds, synts, windows, _ = read_pair_files(conf, events,
                                                          all_pairs)

    dd_weights = weights.DDWeights(all_pairs,
                                   conf.dd.pair_wise_weighting)
    rec_weights = weights.ReceiverWeights(conf, events)
    rec_weights.find_norm_factors(dd_weights,
                                  cc_windows={})
    if mpi.is_main_rank:
        rec_weights.analyze(dd_weights, {})

    calc = partial(single_cc_dd,
                   pairs=all_pairs,
                   obsds=obsds,
                   synts=synts,
                   windows=windows,
                   calc_type=calc_type,
                   conf=conf,
                   dd_weights=dd_weights,
                   rec_weights=rec_weights)

    results = mpi.run_with_mpi(calc, list(range(len(all_pairs))),
                               title="DD")

    if calc_type & CALC_MISFIT:
        def write_misfits():
            total_misfit = 0
            for stap, v in results.iteritems():
                sta, p = stap.split(":")
                _, _, comp, event = sta.split(".")
                total_misfit += v["misfit"]
            filename = conf.get_misfit_folder() / "misfits"
            if len(events) == 1:
                filename += "_" + events[0]
            with open(filename, "w") as f:
                f.write("{:.15f}\n".format(total_misfit))
        mpi.run_on_main_rank(write_misfits)

    if calc_type & CALC_ADJOINT:
        def write_srcs():
            times = obsds[filenames.keys()[0]].times()
            adjs = {sta: np.zeros(len(times)) for sta in filenames.keys()}

            # Sum over pairs
            for stap, adj in tqdm(results.iteritems()):
                sta, p = stap.split(":")
                _, _, comp, event = sta.split(".")
                adjs[sta] += adj["src"]

            for staname in tqdm(adjs, desc="Writing"):
                adj_data = np.zeros((len(times), 2))
                adj_data[:, 0] = times
                adj_data[:, 1] = adjs[staname]
                filename = filenames[staname]
                np.savetxt(filename.replace("DATA_obs", "SEM").replace("semd", "adj"),  # NOQA
                           adj_data)
        mpi.run_on_main_rank(write_srcs)


def calc_cc_plus_dd(calc_type, events, conf,
                    compute_cc_on_all=False):
    all_pairs = dd_pair.read_pairs(conf, events)
    filenames, obsds, synts, windows, cc_stanames = read_pair_files(conf,
                                                                    events,
                                                                    all_pairs)

    dd_weights = weights.DDWeights(all_pairs,
                                   conf.dd.pair_wise_weighting)

    if compute_cc_on_all:
        cc_stanames = filenames.keys()

    cc_windows = {}
    for staname in cc_stanames:
        net, name, comp, ev = staname.split(".")
        staname = ".".join([net, name, comp])
        cc_windows[(ev, staname)] = {"0": 1}

    rec_weights = weights.ReceiverWeights(conf, events)
    rec_weights.find_norm_factors(dd_weights,
                                  cc_windows=cc_windows)
    calc_dd = partial(single_cc_dd,
                      pairs=all_pairs,
                      obsds=obsds,
                      synts=synts,
                      windows=windows,
                      calc_type=calc_type,
                      conf=conf,
                      dd_weights=dd_weights,
                      rec_weights=rec_weights)

    calc_cc = partial(single_cc,
                      obsds=obsds,
                      synts=synts,
                      windows=windows,
                      calc_type=calc_type,
                      conf=conf,
                      rec_weights=rec_weights,
                      write_now=False)

    results_dd = mpi.run_with_mpi(calc_dd, list((range(len(all_pairs)))),
                                  title="DD")

    results_cc = mpi.run_with_mpi(calc_cc, cc_stanames,
                                  title="CC")

    if calc_type & CALC_ADJOINT:
        def write_srcs():
            times = obsds[filenames.keys()[0]].times()
            adjs = {sta: np.zeros(len(times)) for sta in filenames.keys()}

            # Sum over pairs
            for stap, adj in tqdm(results_dd.iteritems()):
                sta, p = stap.split(":")
                _, _, comp, event = sta.split(".")
                adjs[sta] += adj["src"]

            for sta, adj in tqdm(results_cc.iteritems()):
                _, _, comp, event = sta.split(".")
                adjs[sta] += adj["src"]

            for staname in tqdm(adjs, "Writing"):
                adj_data = np.zeros((len(times), 2))
                adj_data[:, 0] = times
                adj_data[:, 1] = adjs[staname]
                filename = filenames[staname]
                np.savetxt(filename.replace("DATA_obs", "SEM").replace("semd", "adj"),  # NOQA
                           adj_data)
        mpi.run_on_main_rank(write_srcs)

    if calc_type & CALC_MISFIT:
        def write_misfits():
            total_cc_misfit = 0
            total_dd_misfit = 0
            for stap, v in results_dd.iteritems():
                sta, p = stap.split(":")
                _, _, comp, event = sta.split(".")
                total_dd_misfit += v["misfit"]
            for sta, v in results_cc.iteritems():
                _, _, comp, event = sta.split(".")
                total_cc_misfit += v["misfit"]

            filename = conf.get_misfit_folder() / "misfits"
            if len(events) == 1:
                filename += "_" + events[0]
            total_misfit = total_cc_misfit + total_dd_misfit
            with open(filename, "w") as f:
                f.write("CC: {:.15f}\n".format(total_cc_misfit))
                f.write("DD: {:.15f}\n".format(total_dd_misfit))
                f.write("{:.15f}\n".format(total_misfit))
        mpi.run_on_main_rank(write_misfits)


def single_one(filename, obsds, synts, windows,
               calc_type,
               conf,
               write_now=False):
    from pyadjoint.utils import window_taper
    from scipy.integrate import simps

    obsd = obsds[filename]
    synt = synts[filename]
    adj_windows = windows[filename]

    nlen_t = len(obsd.data)

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
        times = obsd.times()
        adj_data = np.zeros((len(times), 2))
        adj_data[:, 0] = times
        adj_data[:, 1] = p
        if write_now:
            np.savetxt(
                filename.replace("DATA_obs", "SEM").replace("semd", "adj"),
                adj_data)
        else:
            res[filename]["src"] = adj_data[:, 1]

    return res


def calc_one(calc_type, events, conf, filenames=None):
    filenames, obsds, synts, windows = read_files(conf, events, filenames)

    calc = partial(single_one,
                   obsds=obsds,
                   synts=synts,
                   windows=windows,
                   calc_type=calc_type,
                   conf=conf,
                   write_now=True)

    results = mpi.run_with_mpi(calc, filenames, title="One")

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


adjs = {
    "cc": calc_cc,
    "dd": calc_cc_dd,
    "cc+dd": calc_cc_plus_dd,
    "cc_and_dd": partial(calc_cc_plus_dd, compute_cc_on_all=True),
    "one": calc_one,
    "ep": NotImplemented,
    "waveform": NotImplemented,
    "norm_wf": NotImplemented,
    "waveform_dd": NotImplemented,
    "convolved": NotImplemented,
    "wf+dd": NotImplemented,
    "wf_and_dd": NotImplemented,
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
