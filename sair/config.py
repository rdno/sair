#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from . import io_utils as io
from . import utils
from namedlist import namedlist

import numpy as np


Adjoint = namedlist("Adjoint", ["freq_min", "freq_max",
                                "taper_type", "taper_percentage",
                                ("receiver_weights", None),
                                ("source_weights", None),
                                ("weight_single", 1.0),
                                ("weight_dd", 1.0),
                                ("use_receiver_weights_for_dd", True),
                                ("use_receiver_weights_for_single", True)])

Window = namedlist("Windows", [("min_vel", 0),
                               ("max_vel", 0),
                               ("periodic_horiz_dist", 0),
                               ("use_pyflex", False),
                               ("pyflex_conf_file", None)])

Simulation = namedlist("Simulation", ["kernels", "comps", "dt", "nstep",
                                      ("seismogram_format", "ascii"),
                                      ("target_noise_level", 0.0)])

DD = namedlist("DD", ["closeness", "close_pairs",
                      "azimuth", "azimuth_interval",
                      "similarity", "similarity_threshold",
                      ("event_pairing", False),
                      ("event_closeness", 0),
                      ("cross_event_pairing", False),
                      ("pair_wise_weighting", True),
                      ("use_pair_file", None)])

Inversion = namedlist("Inversion",
                      ["start_it", "stop_it",
                       "optimization",
                       ("loss_of_conjugacy", 0.1),
                       ("lbfgs_memory", 5)])

Linesearch = namedlist("Linesearch",
                       ["first", "step", "ntry",
                        "min_step", "min_improvement"])


PostProcessing = namedlist("PostProcessing",
                           ["precondition",
                            "preconditioners",
                            "smooth",
                            ("use_specfem_xsmooth", True),
                            ("sigmah", 0), ("sigmav", 1),
                            ("smooth_adaptive", False),
                            ("min_sigmah", 0), ("max_sigmah", 0),
                            ("min_sigmav", 0), ("max_sigmav", 0),
                            ("smooth_adaptive_data_folder", None),
                            ("smooth_adaptive_filename", "adaptive"),
                            ("precond_wtr", 0.01),
                            ("precond_folder", None),
                            ("mask_sources", False),
                            ("mask_stations", False),
                            ("mask_radius", 0),
                            ("clip_kernels", False),
                            ("clip_maxval", "auto"),
                            ("clip_minval", "auto"),
                            ("smooth_around_sources", False),
                            ("smooth_around_stations", False)])

Options = namedlist("Options",
                    ["n_parallel_src_runs",
                     "misfit_nproc",
                     "calc_events_individually"])

Prepare = namedlist("Prepare",
                    ["nproc",
                     ("mpi", False),
                     ("target_command", None),
                     ("init_command", None)])

KeepData = namedlist("KeepData",
                     [("traces", False),
                      ("adjoint_sources", False),
                      ("event_kernels", False)])


class Config(object):
    model_names = {
        "beta": "vs",
        "alpha": "vp",
        "rho": "rho",
        "rhop": "rho"
    }

    def __init__(self, name, init_model, target_model,
                 sources_file, station_folder, source_folder,
                 specfem_folder,
                 simulation, adjoint, window, dd, inversion,
                 linesearch, postprocessing,
                 options, prepare_model=None,
                 keep_data=None,
                 target_data_folder="",
                 filename=None,
                 root_work_dir=io.Path("work")):
        super(Config, self).__init__()
        self.filename = io.Path(filename)
        self.dirname = io.dirname_of(self.filename)
        self.name = name
        self.root_work_dir = io.Path(root_work_dir)
        self.specfem_folder = io.Path(specfem_folder, self.dirname)
        self.init_model = io.Path(init_model, self.dirname)
        self.target_model = io.Path(target_model, self.dirname)
        self.target_data_folder = io.Path(target_data_folder, self.dirname)
        if self.target_model is None and self.target_data_folder is None:
            raise Exception("Both target model and data cannot be None.")
        self.sources_file = io.Path(sources_file, self.dirname)
        self.station_folder = io.Path(station_folder, self.dirname)
        self.source_folder = io.Path(source_folder, self.dirname)
        self.simulation = Simulation(**simulation)

        self.adjoint = Adjoint(**adjoint)
        if self.adjoint.receiver_weights:
            self.adjoint.receiver_weights = io.Path(
                self.adjoint.receiver_weights, self.dirname)

        self.window = Window(**window)

        self.dd = DD(**dd)
        if self.dd.use_pair_file:
            self.dd.use_pair_file = io.Path(self.dd.use_pair_file,
                                            self.dirname)

        self.inversion = Inversion(**inversion)
        self.linesearch = Linesearch(**linesearch)

        self.postprocessing = PostProcessing(**postprocessing)
        if self.postprocessing.precond_folder:
            self.postprocessing.precond_folder = io.Path(
                self.postprocessing.precond, self.dirname)

        if self.postprocessing.smooth_adaptive_data_folder:
            self.postprocessing.smooth_adaptive_data_folder = io.Path(
                self.postprocessing.smooth_adaptive_data_folder,
                self.dirname)

        self.options = Options(**options)
        self.prepare_model = None
        if prepare_model:
            self.prepare_model = Prepare(**prepare_model)
        else:
            self.prepare_model = Prepare(0)

        if keep_data is None:
            self.keep_data = KeepData()
        else:
            self.keep_data = KeepData(**keep_data)

    def write_to_file(self, filename):
        data = {
            "name": self.name,
            "root_work_dir": self.root_work_dir,
            "specfem_folder": self.specfem_folder,
            "init_model": self.init_model,
            "target_model": self.target_model,
            "target_data_folder": self.target_data_folder,
            "sources_file": self.sources_file,
            "station_folder": self.station_folder,
            "source_folder": self.source_folder,
            "simulation": self.simulation._asdict(),
            "adjoint": self.adjoint._asdict(),
            "window": self.window._asdict(),
            "dd": self.dd._asdict(),
            "inversion": self.inversion._asdict(),
            "linesearch": self.linesearch._asdict(),
            "postprocessing": self.postprocessing._asdict(),
            "options": self.options._asdict(),
            "prepare_model": self.prepare_model._asdict(),
            "keep_data": self.keep_data._asdict()
        }
        utils.write_yaml(filename, data)

    def __str__(self):
        return "<Config {}>".format(self.name)

    def __repr__(self):
        return self.__str__()

    def get_workdir(self, event):
        return self.root_work_dir / "events" / event

    def get_obs_folder(self, event):
        return self.get_workdir(event) / "DATA_obs"

    def get_syn_folder(self, event):
        return self.get_workdir(event) / "DATA_syn"

    def get_sem_folder(self, event):
        return self.get_workdir(event) / "SEM"

    def get_event_kernel_folder(self, event, misfit_type, it):
        return self.get_workdir(event) / "kernels_{}_{}".format(
            misfit_type, it)

    def _get_data_folder(self, data_type, event):
        if data_type == "obs":
            return self.get_obs_folder(event)
        elif data_type == "syn":
            return self.get_syn_folder(event)
        else:
            raise Exception("Invalid Data Type: {}".format(data_type))

    def get_event_names(self):
        """Reads sources file and returns the names"""
        with open(self.sources_file) as f:
            return [line.rstrip() for line in f]

    def get_event_loc(self, event):
        def get_float_value(line):
            """Strip comments and get the value
            """
            return float(line.split("#")[0].split("=")[1])

        with open(self.source_folder / "SOURCE_{}".format(event)) as f:
            for line in f:
                if line.startswith("xs"):
                    x = get_float_value(line)
                elif line.startswith("zs"):
                    z = get_float_value(line)
            loc = np.array([x, z])
        return loc

    def get_event_locs(self):
        locs = {}
        for event in self.get_event_names():
            locs[event] = self.get_event_loc(event)
        return locs

    def _get_station_filename(self, event):
        return self.station_folder / "STATIONS_{}".format(event)

    def get_stations(self, events,
                     add_event_label=False, with_comps=True):

        # For a single event
        if not isinstance(events, list):
            events = [events]

        def get_name(event, *args):
            args = list(args)
            if add_event_label:
                args.append(event)
            return ".".join(args)

        stations = {}

        for event in events:
            with open(self._get_station_filename(event)) as f:
                for sta in f:
                    name, net, x, z, _, _ = sta.split()
                    if not with_comps:
                        fname = get_name(event, net, name)
                        stations[fname] = np.array([float(x), float(z)])
                    else:
                        for comp in self.simulation.comps:
                            fname = get_name(event, net, name, comp)
                            stations[fname] = np.array([float(x), float(z)])
        return stations

    def get_station_list(self, event):
        stations = []
        with open(self._get_station_filename(event)) as f:
            for sta in f:
                name, net, x, z, _, _ = sta.split()
                stations.append({
                    "net": net,
                    "name": name,
                    "event": event,
                    "x": float(x),
                    "z": float(z)
                })
        return stations

    def get_su_filename(self, event, comp, data_type="obs"):
        if data_type == "adjoint":
            return self.get_adjoint_su_filename(event, comp)
        c = comp[-1].lower()
        return self._get_data_folder(data_type, event) / "U{}_file_single.su".format(c)  # NOQA

    def get_adjoint_su_filename(self, event, comp):
        c = comp[-1].lower()
        return self.get_workdir(event) / "SEM" / "U{}_file_single.su.adj".format(c)  # NOQA

    def get_collected_folder(self, it):
        return self.root_work_dir / "DATA" / "iter_{}".format(it)

    def get_collected_trace_folder(self, event, it, data_type):
        return self.get_collected_folder(it) / event / data_type

    def get_collected_su_filename(self, event, it, comp, data_type):
        c = comp[-1].lower()
        filename = "U{}_file_single.su".format(c)
        if data_type == "adjoint":
            filename += ".adj"
        return self.get_collected_trace_folder(event, it, data_type) / filename

    def get_collected_event_kernel_folder(self, event, it):
        return self.get_collected_folder(it) / event / "kernels"

    def get_trace_filename(self, event, station, data_type="obs"):
        return self._get_data_folder(data_type, event) / "{}.semd".format(station)  # NOQA

    def get_windows_filename(self, event):
        return self.root_work_dir / "windows" / "{}.json".format(event)

    def get_pair_filename(self, winname):
        name = winname.split(":")[0]
        net, sta, comp, event = name.split(".")
        station = ".".join([net, sta, comp])
        return name, self.get_trace_filename(event, station)

    def get_info_from_filename(self, filename):
        import re
        pat = re.compile(r"""{root}/
        (?P<event>[\w\d]+)/
        DATA_(?P<data_type>[\w]+)/
        (?P<station>[\w\.]+)\.semd""".format(root=self.root_work_dir),
                         re.VERBOSE)
        return pat.match(filename).groupdict()

    def get_model_folder(self, misfit_type=None, iteration=None):
        model_folder = self.root_work_dir / "models"
        if misfit_type is None and iteration is None:
            return model_folder
        else:
            return model_folder / "DATA_{}_{}".format(misfit_type, iteration)

    def get_update_folder(self, misfit_type, iteration):
        return self.root_work_dir / "update_{}_{}".format(
            misfit_type, iteration)

    def get_log_folder(self):
        return self.root_work_dir / "logs"

    def get_log_files(self, name, it=1, try_no=None):
        try_suffix = "_{}".format(try_no) if try_no is not None else ""
        log_file = self.get_log_folder() / "{}_log_{}{}".format(
            name, it, try_suffix)
        err_file = self.get_log_folder() / "{}_err_{}{}".format(
            name, it, try_suffix)
        return log_file, err_file

    def get_data_misfits_file(self):
        return self.root_work_dir / "misfits"

    def get_model_misfits_file(self, data_type):
        return self.root_work_dir / "model_misfits_{}".format(data_type)

    def get_misfit_folder(self):
        return self.root_work_dir / "data_misfits"

    def get_linesearch_file(self, misfit_type, iteration):
        return self.root_work_dir / "linesearch_{}_{}".format(misfit_type,
                                                              iteration)

    def get_pairs(self):
        return utils.load_json(self.root_work_dir / "pairs.json")

    def get_model_name(self, kernel_name):
        return self.model_names[kernel_name]

    def get_src_weights(self):
        if self.adjoint.source_weights:
            return utils.load_json(self.adjoint.source_weights)
        else:
            return {e: 1.0 for e in self.get_event_names()}


def load_config(filename):
    data = utils.load_yaml(filename)
    conf = Config(filename=filename, **data)
    return conf


def get_runs(filename):
    """Reads a run file and returns an iterator of run parameters"""
    data = utils.load_yaml(filename)
    base_config_file = data["config_file"]
    for run in data["runs"]:
        name = run["name"]
        conf = load_config(base_config_file)
        conf.root_work_dir = io.Path(name)
        for varname, var in run["vars"].iteritems():
            if "." in varname:
                attrname, subattrname = varname.split(".")
                attr = getattr(conf, attrname)
                setattr(attr, subattrname, var)
            else:
                setattr(conf, varname, var)
        conf.write_to_file("tmp.yml")
        new_conf = load_config("tmp.yml")
        yield name, new_conf, run["misfit_type"]
