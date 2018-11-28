# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import obspy
import numpy as np

import json
import os
import psutil
import re
import signal
import sys
import yaml

from multiprocessing import Pool

from tqdm import tqdm as tqdm_orig
tqdm_orig.monitor_interval = 0  # NOQA


def tqdm(iterable=None, *args, **kwargs):
    kwargs.update({"file": sys.stdout})
    return tqdm_orig(iterable, *args, **kwargs)


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


def write_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_yaml(filename):
    with open(filename) as f:
        return yaml.load(f)


def write_yaml(filename, data):
    with open(filename, "w") as f:
        yaml.dump(data, f)


class AsciiReader(object):
    def __init__(self, filenames):
        super(AsciiReader, self).__init__()
        self.filenames = filenames
        self.read = {}

    def __getitem__(self, key):
        return self._readfile(key)

    def _readfile(self, key):
        filename = self.filenames[key]
        if filename in self.read:
            return self.read[filename]
        else:
            tr = read_ascii_trace(filename)
            self.read[filename] = tr
            return tr


def read_ascii_trace(filename):
    data = np.loadtxt(filename)
    dt = data[:, 0][1]-data[:, 0][0]
    stats = {"delta": dt, "channel": "BHZ"}
    return obspy.Trace(data[:, 1], stats)


class SUReader(object):
    def __init__(self, conf, events, data_type,
                 collected=False,
                 collected_it=None):
        super(SUReader, self).__init__()
        self.conf = conf
        self.events = events
        self.data_type = data_type
        self.collected = collected
        self.collected_it = collected_it
        self.read = {}

    def __getitem__(self, key):
        return self._readtrace(key)

    def _readtrace(self, key):
        net, name, comp, event = key.split(".")
        if event not in self.events:
            raise Exception("Event not found: {}".format(event))

        if key not in self.read:
            if self.collected:
                filename = self.conf.get_collected_su_filename(
                    event, self.collected_it, comp, self.data_type)
            else:
                filename = self.conf.get_su_filename(event, comp,
                                                     self.data_type)
            station_list = self.conf.get_station_list(event)
            traces = read_su(filename, comp,
                             self.conf.simulation.dt,
                             event, station_list)
            self.read.update(traces)

        return self.read[key]


def read_su(filename, comp, dt, event, station_list):
    """Reads SU data file and returns a dict where keys are the full names
    of stations.

    It also writes location and name information to the stats object."""
    st = obspy.read(filename, "SU", byteorder="<",
                    unpack_trace_headers=True)
    traces = {}
    for sta, tr in zip(station_list, st):
        tr.stats.delta = dt
        tr.stats.network = sta["net"]
        tr.stats.station = sta["name"]
        tr.stats.channel = comp
        tr.stats.srcx = tr.stats.su.trace_header.source_coordinate_x
        tr.stats.srcy = tr.stats.su.trace_header.source_coordinate_y
        tr.stats.stax = sta["x"]
        tr.stats.stay = sta["z"]
        staname = ".".join([sta["net"], sta["name"], comp, event])
        traces[staname] = tr
    return traces


def write_su_adjoint(conf, data, dt, events):
    sta_list = {}
    traces = {}
    for event in events:
        sta_list[event] = conf.get_station_list(event)
        for comp in conf.simulation.comps:
            traces[(event, comp)] = [None for s in sta_list[event]]

    for sta in data.keys():
        net, name, comp, event = sta.split(".")
        tr = obspy.Trace(data[sta].astype(np.float32), {"delta": dt})
        found = False
        for i, s in enumerate(sta_list[event]):
            if net == s["net"] and name == s["name"]:
                found = True
                break
        if not found:
            raise Exception("Stations could not be found: {}".format(sta))  # NOQA

        traces[(event, comp)][i] = tr

    for event in events:
        for comp in conf.simulation.comps:
            st = obspy.Stream(traces[(event, comp)])
            st.write(conf.get_adjoint_su_filename(event, comp),
                     format="SU", byteorder="<")


def get_pool(processes):
    """Returns interruptable Pool object

    From: https://stackoverflow.com/a/45259908
    """
    parent_id = os.getpid()

    def worker_init():
        def sig_int(signal_num, frame):
            parent = psutil.Process(parent_id)
            for child in parent.children():
                if child.pid != os.getpid():
                    child.kill()
            parent.kill()
            psutil.Process(os.getpid()).kill()
        signal.signal(signal.SIGINT, sig_int)

    return Pool(processes, worker_init)


# TODO: Use this in runner
def specfem_write_parameters(filename, parameters, output_file=None):
    """Write parameters to a specfem config file"""

    with open(filename) as f:
        pars = f.read()

    for varname, value in parameters.iteritems():
        pat = re.compile(
            "(^{varname}\s*=\s*)([^#$\s]+)".format(varname=varname),
            re.MULTILINE)
        pars = pat.sub("\g<1>{value}".format(value=value), pars)

    if output_file is None:
        output_file = filename

    with open(output_file, "w") as f:
        f.write(pars)


def read_su_file(filename):
    return obspy.read(filename, "SU", byteorder="<",
                      unpack_trace_headers=True)


def write_su_file(st, filename):
    st.write(filename, format="SU", byteorder="<")


def add_noise(st, noise_level):
    for tr in st:
        maxamp = np.max(np.abs(tr.data))
        noise_t = np.random.normal(0, 0.33, len(tr)).astype(np.float32)*maxamp*noise_level
        tr.data = tr.data + noise_t
        tr.stats.delta = 0.06
    return st
