#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from pytomo3d.doubledifference import pairing
from pytomo3d.doubledifference import utils as pair_utils

import argparse

import numpy as np

from . import config
from . import utils
from . import io_utils as io
from . import mpi
from functools import partial


def pair_from_current_events(pair, events):
    def event_of(x):
        return x.split(":")[0].split(".")[-1]
    for event in events:
        this_event = event_of(pair["window_id_i"]) == event \
                     or event_of(pair["window_id_j"]) == event
        if this_event:
            return True
    return False


def read_pairs(conf, events):
    pairs = utils.load_json(conf.root_work_dir / "pairs.json")

    all_pairs = []
    for comp in pairs:
        all_pairs.extend(pairs[comp])

    all_pairs = list(filter(partial(pair_from_current_events,
                                    events=events), all_pairs))
    return all_pairs


def get_distance(loc1, loc2):
    return np.sqrt(np.sum((loc1-loc2)**2))


def close_pairs(stations, threshold, old_pairs):
    def pair_by_distance(pair_i, pair_j):
        distance = get_distance(stations[pair_i.split(":")[0]],
                                stations[pair_j.split(":")[0]])
        is_paired = distance < threshold
        return distance, is_paired

    return pairing._reduce_pairs(pair_by_distance, "distance", old_pairs,
                                 allow_self_pairing=True)


def event_pairs(allow_event_pairing, allow_cross_pairing, allow_rec_pairing,
                old_pairs):
    def pair_by_events(pair_i, pair_j):
        net_i, sta_i, _, ev_i = pair_i.split(":")[0].split(".")
        net_j, sta_j, _, ev_j = pair_j.split(":")[0].split(".")
        if ev_i == ev_j:
            if allow_rec_pairing:
                return "Same Event", True
            else:
                return "Same Event", False
        elif allow_event_pairing:
            if net_i == net_j and sta_i == sta_j:
                return "Different Event Same Station", True
            elif allow_cross_pairing:
                return "Different Event Different Station", True
            else:
                return "Different Event Different Station", False
        else:
            return "Different Event Not Allowed", False
    return pairing._reduce_pairs(pair_by_events, "event", old_pairs,
                                 allow_self_pairing=True)


def close_event_pairs(event_locations, threshold, old_pairs):
    def pair_by_event_closeness(pair_i, pair_j):
        _, comp_i, _ = pair_utils.deconstruct_winname(pair_i)
        _, comp_j, _ = pair_utils.deconstruct_winname(pair_j)
        net_i, sta_i, _, ev_i = comp_i.split(".")
        net_j, sta_j, _, ev_j = comp_j.split(".")
        if ev_i == ev_j:
            return 0, True
        else:
            loc_i = event_locations[ev_i]
            loc_j = event_locations[ev_j]
            dist = get_distance(loc_i, loc_j)
            return dist, dist < threshold
    return pairing._reduce_pairs(pair_by_event_closeness,
                                 "event_dist", old_pairs,
                                 allow_self_pairing=True)


def azimuth_pairs(event_locs, station_locs, threshold, old_pairs):
    def pair_by_azimuth(pair_i, pair_j):
        sta_i, comp_i, _ = pair_utils.deconstruct_winname(pair_i)
        sta_j, comp_j, _ = pair_utils.deconstruct_winname(pair_j)
        _, _, _, ev_i = comp_i.split(".")
        _, _, _, ev_j = comp_j.split(".")
        if ev_i == ev_j:
            ev = event_locs[ev_i]
            s_loc_i = station_locs[pair_i.split(":")[0]]
            s_loc_j = station_locs[pair_j.split(":")[0]]
            az_i = np.arctan2(s_loc_i[1]-ev[1], s_loc_i[0]-ev[0])
            az_j = np.arctan2(s_loc_j[1]-ev[1], s_loc_j[0]-ev[0])
            az_diff = np.abs(az_i-az_j)*180/np.pi
            return az_diff, az_diff < threshold
        else:  # Can be implemented later
            return -1, None
    return pairing._reduce_pairs(pair_by_azimuth, "azimuth_diff", old_pairs,
                                 allow_self_pairing=True)


def similar_wf_pairs(data, threshold, old_pairs):
    """Pair the data using waveform similarity

    :param data: windowed data
    :type data: station
    :param threshold: threshold  value
    :type threshold: float
    :param old_pairs: previous pairs
    :type old_pairs: dict
    :returns: pairs
    :rtype: dict

    """

    def pair_by_similarity(pair_i, pair_j):
        data_i = data[pair_i]
        data_j = data[pair_j]
        minlen = min([len(data_i), len(data_j)])
        maxv = max([np.abs(data_i).max(), np.abs(data_j).max()])
        diff = 1 - np.abs(np.sum((data_i[:minlen]-data_j[:minlen])/maxv)/minlen)  # NOQA
        is_paired = diff > threshold
        return diff, is_paired

    return pairing._reduce_pairs(pair_by_similarity, "similarity", old_pairs)


def read_windows(event, conf, windows):
    print("{} Reading windows...".format(event))
    raw_windows = utils.load_yaml(conf.get_windows_filename(event))
    for win in raw_windows:
        comp = win[-1]
        winname = ".".join([win, event])
        for i, w in enumerate(raw_windows[win]):
            windows[comp][winname+":"+str(i)] = w


def write_pairs_for_event(event, conf):
    event_pairs_file = conf.root_work_dir / "pairs.{}.json"

    windows = {"X": {}, "Y": {}, "Z": {}}

    # print("{} Creating filenames...".format(event))
    # stations = conf.get_stations(event, with_comps=True)
    if isinstance(event, list):
        for e in event:
            read_windows(e, conf, windows)
    else:
        read_windows(event, conf, windows)

    stations = conf.get_stations(event,
                                 add_event_label=True,
                                 with_comps=True)

    print("{} Creating all pairs...".format(event))
    pairs = pairing.create_all_pairs(windows)
    if conf.dd.event_pairing:
        print("{} Creating event pairs...".format(event))
        pairs = event_pairs(
            allow_event_pairing=conf.dd.event_pairing,
            allow_cross_pairing=conf.dd.cross_event_pairing,
            allow_rec_pairing=True,
            old_pairs=pairs)
        if conf.dd.event_closeness:
            pairs = close_event_pairs(conf.get_event_locs(),
                                      conf.dd.event_closeness,
                                      pairs)

    if conf.dd.closeness:
        print("{} Creating close pairs...".format(event))
        pairs = close_pairs(stations, conf.dd.close_pairs, pairs)

    if conf.dd.azimuth:
        print("{} Creating azimuth pairs...".format(event))
        event_locs = conf.get_event_locs()
        pairs = azimuth_pairs(event_locs, stations,
                              conf.dd.azimuth_interval,
                              pairs)

    if conf.dd.similarity:
        print("{} Creating similar pairs...".format(event))
        data = {}
        if conf.simulation.seismogram_format == "ascii":
            for sta in conf.get_stations(event, with_comps=True):
                tr = utils.read_ascii_trace(
                    conf.get_trace_filename(event, sta))
                data["{}.{}:0".format(sta, event)] = tr.data
        elif conf.simulation.seismogram_format == "su":
            for comp in conf.simulation.comps:
                su_data = utils.SUReader(conf, event, "obs")
                for win in windows[comp[-1]]:
                    sta, n = win.split(":")
                    tr = su_data[sta]
                    left, right = [int(x*tr.stats.sampling_rate)
                                   for x in windows[comp[-1]][win]]
                    data[win] = tr.data[left:right]
        else:
            raise Exception("Not a valid seismogram format: {}".format(
                conf.simulation.seismogram_format))
        pairs = pairing.similar_pairs(data, conf.dd.similarity_threshold,
                                      pairs, allow_self_pairing=True)

    if isinstance(event, list):
        utils.write_json(conf.root_work_dir / "pairs.json", pairs)
    else:
        print("{} Writing...".format(event))
        utils.write_json(event_pairs_file.format(event), pairs)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Find DD Pairs")
    parser.add_argument("conf_file", help="config.yml file")
    args = parser.parse_args()
    conf = config.load_config(args.conf_file)

    pairs_file = conf.root_work_dir / "pairs.json"

    if conf.dd.use_pair_file:
        if mpi.is_main_rank:
            io.copy_file(conf.dd.use_pair_file, pairs_file)
        return

    events = conf.get_event_names()

    if conf.dd.event_pairing:
        write_pairs_for_event(events, conf)
    else:
        mpi.run_with_mpi(partial(write_pairs_for_event,
                                 conf=conf), events,
                         "Finding pairs...")

    def combine():
        print("Combining Pairs")
        pairs = {"X": [], "Y": [], "Z": []}
        event_pairs_file = conf.root_work_dir / "pairs.{}.json"
        for event in events:
            ind_pairs = utils.load_json(event_pairs_file.format(event))
            for comp in "XYZ":
                pairs[comp] = pairs[comp] + ind_pairs[comp]
            io.remove_file(event_pairs_file.format(event))
        print("Finding Weights")
        pairing.find_weights(pairs)
        utils.write_json(pairs_file, pairs)

    if not conf.dd.event_pairing:
        mpi.comm.barrier()
        mpi.run_on_main_rank(combine)
    else:
        pairs = utils.load_json(pairs_file)
        pairing.find_weights(pairs)
        utils.write_json(pairs_file, pairs)


if __name__ == '__main__':
    main()
