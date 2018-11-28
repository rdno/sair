# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import argparse

import numpy as np

from .config import load_config
from . import utils
from . import io_utils as io
from . import mpi

from collections import defaultdict
from functools import partial
import logging

import pyflex


class WindowGetter(object):
    def __init__(self, conf):
        self.conf = conf
        self.read_windows = {}
        self.paireds = None
        self.allow_singles_to_use_paired_windows = True

    def __getitem__(self, key):
        """Key is station name with comp and event_label"""
        return self._get_window(key)

    def set_pairs(self, pairs):
        self.paireds = defaultdict(set)
        for pair in pairs:
            sta_i, id_i = pair["window_id_i"].split(":")
            sta_j, id_j = pair["window_id_j"].split(":")
            self.paireds[sta_i].add(int(id_i))
            self.paireds[sta_j].add(int(id_j))

    def _get_window(self, key):
        win_type = None
        if isinstance(key, tuple):
            key, win_type = key
        net, name, comp, event = key.split(".")
        sta = ".".join([net, name, comp])
        if event not in self.read_windows:
            self.read_windows[event] = utils.load_json(
                self.conf.get_windows_filename(event))
        windows = self.read_windows[event][sta]
        if win_type is None or self.paireds is None:
            return windows
        elif win_type == "dd":
            paireds = self.paireds[key]
            return [w for i, w in enumerate(windows)
                    if i in paireds]
        elif win_type == "single":
            paireds = self.paireds[key]
            return [w for i, w in enumerate(windows)
                    if self.allow_singles_to_use_paired_windows or
                    i not in paireds]
        else:
            raise Exception("Unknown window type:", win_type)


def get_min_dist(conf, event_loc, sta_loc):
    dist = np.sqrt(np.sum((sta_loc-event_loc)**2))
    if conf.window.periodic_horiz_dist > 0:
        p = conf.window.periodic_horiz_dist
        per_sta_loc = np.array([sta_loc[0] % p, sta_loc[1]])
        per_event_loc = np.array([event_loc[0] % p, event_loc[1]])
        per_dist = np.sqrt(np.sum((per_sta_loc-per_event_loc)**2))
        if per_dist < dist:
            dist = per_dist
    return dist


def compute_windows_by_vel(event, conf):
    event_loc = conf.get_event_loc(event)
    max_vel = conf.window.max_vel
    min_vel = conf.window.min_vel
    maxt = conf.simulation.dt*(conf.simulation.nstep-10)
    windows = {}
    for sta, sta_loc in conf.get_stations(event).iteritems():
        dist = get_min_dist(conf, event_loc, sta_loc)
        a = max(dist/max_vel, 0.001)
        b = min(max(dist/min_vel, a+0.001), maxt)
        windows[sta] = [(a, b)]

    filename = conf.get_windows_filename(event)
    io.makedir_for(filename)
    utils.write_json(filename, windows)
    return {}


def compute_windows_pyflex(event, pyflex_conf, conf):
    obsds = utils.SUReader(conf, [event], "obs")
    synts = utils.SUReader(conf, [event], "syn")
    windows = {}

    event_loc = conf.get_event_loc(event)
    for station, loc in conf.get_stations(event,
                                          add_event_label=True).iteritems():
        dist = get_min_dist(conf, event_loc, loc)
        t = dist / conf.window.max_vel
        ti = int(t*obsds[station].stats.sampling_rate)
        pyflex_conf.noise_start_index = 0
        pyflex_conf.noise_end_index = ti
        pyflex_conf.signal_start_index = ti+1
        pyflex_conf.signal_end_index = conf.simulation.nstep
        sta_wins = pyflex.select_windows(obsds[station],
                                         synts[station],
                                         config=pyflex_conf,
                                         plot=False)
        sta = ".".join(station.split(".")[:3])
        windows[sta] = [(w.left*w.dt, w.right*w.dt) for w in sta_wins]

    filename = conf.get_windows_filename(event)
    io.makedir_for(filename)
    utils.write_json(filename, windows)
    return {}


def compute_windows(conf_file):
    conf = load_config(conf_file)
    if conf.window.use_pyflex:
        if conf.window.pyflex_conf_file is None:
            raise Exception("pyflex_conf_file should be set to compute windows.")  # NOQA
        logger = logging.getLogger("pyflex")
        logger.setLevel(logging.ERROR)

        pyflex_args = utils.load_yaml(conf.window.pyflex_conf_file)
        pyflex_conf = pyflex.Config(**pyflex_args)
        mpi.run_with_mpi(partial(compute_windows_pyflex,
                                 pyflex_conf=pyflex_conf,
                                 conf=conf),
                         conf.get_event_names())
    elif conf.window.max_vel == 0 or conf.window.min_vel == 0:
        raise Exception("Max & min velocity should be set for window selection.")  # NOQA
    else:
        mpi.run_with_mpi(partial(compute_windows_by_vel, conf=conf),
                         conf.get_event_names())


def main():
    parser = argparse.ArgumentParser(description='Calculate Adjoint')
    parser.add_argument('config_file', help="Config file")
    args = parser.parse_args()
    compute_windows(args.config_file)


if __name__ == "__main__":
    main()
