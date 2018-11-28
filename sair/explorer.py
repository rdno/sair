# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from .config import load_config
from . import io_utils as io
from . import linesearch
from . import plot
from . import specfem_io as sp
from . import utils
from . import weights

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from collections import OrderedDict

from namedlist import namedlist

RecipeItem = namedlist("Recipe", ["name", "model", "plot_func_name",
                                  "kwargs", ("param", None), ("title", None)])


def build_recipe(names, model, plot_func_name, kwargs,
                 param=None, append_to=None, prepend_to=None):
    recipe = []
    for name in names:
        recipe.append(RecipeItem(name, model, plot_func_name, kwargs, param))

    if append_to:
        append_to.extend(recipe)
        return append_to
    if prepend_to:
        recipe.extend(prepend_to)
        return recipe

    return recipe


def get_optimum_plot_conf(n_plots, ncols=None, nrows=None):
    """If you want you can, force column or row number.
    Otherwise, it will try to approximate a square.
    """
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(n_plots)))
    if nrows is None:
        nrows = int(np.ceil(n_plots / ncols))
    return ncols, nrows


class Explorer(object):
    """A helper class for data exploration """
    def __init__(self, folder):
        super(Explorer, self).__init__()
        self.folder = io.Path(folder)
        self.conf = load_config(self.folder / "config.yml")
        self.misfit_type = self._detect_misfit_type()
        self.n_iterations = self._detect_n_iterations()
        self._reconfigure_conf_params()
        self.clean_cache()

    def _detect_misfit_type(self):
        try:
            m = glob.glob(self.folder / "kernels_*_1")[0]
            return m.replace(self.folder / "kernels_", "").replace("_1", "")
        except IndexError:
            return None

    def _detect_n_iterations(self):
        try:
            return len(self.get_misfits()) - 1
        except:
            return 0

    def _reconfigure_conf_params(self):
        self.conf.root_work_dir = self.folder
        self.conf.sources_file = self.folder / "DATA" / "events"
        self.conf.source_folder = self.folder / "DATA" / "sources"
        self.conf.station_folder = self.folder / "DATA" / "stations"

        if self.misfit_type:
            self.conf.init_model = self.conf.get_model_folder(
                self.misfit_type, 0)
            if self.conf.target_model:
                self.conf.target_model = self.conf.get_model_folder(
                    self.misfit_type, "target")

        if self.conf.adjoint.receiver_weights:
            self.conf.adjoint.receiver_weights = self.folder / "DATA" / "receiver_weights"  # NOQA
        if self.conf.adjoint.source_weights:
            self.conf.adjoint.source_weights = self.folder / "DATA" / "source_weights"  # NOQA

    def clean_cache(self):
        self._xs = None
        self._zs = None
        self._models = {}
        self._kernels = {}
        self._updates = {}

    @property
    def xs(self):
        if self._xs is None:
            self._xs = self.get_model(0, "x")
        return self._xs

    @property
    def zs(self):
        if self._zs is None:
            self._zs = self.get_model(0, "z")
        return self._zs

    def has_pairs(self):
        return io.is_a_file(self.conf.root_work_dir / "pairs.json")

    def __str__(self):
        return "<Explorer ({name}) {misfit}-{n_iter}>".format(
            name=self.conf.name,
            misfit=self.misfit_type,
            n_iter=self.n_iterations)

    def __repr__(self):
        return self.__str__()

    # Getters
    def get_misfits(self, normalized=False):
        misfits = np.loadtxt(self.folder / "misfits")
        if normalized:
            misfits /= misfits[0]
        return misfits

    def get_model_misfits(self, param):
        misfits = np.loadtxt(self.folder / "model_misfits_{}".format(param))
        return misfits

    def get_linesearch(self, it):
        filename = self.folder / "linesearch_{}_{}".format(
            self.misfit_type, it)
        return linesearch.read_searchfile(filename)

    def get_step_lengths(self):
        """Get the step lengths decided with linesearch for each iteration"""
        alphas = []
        for it in range(1, self.n_iterations+1):
            filename = self.folder / "linesearch_{}_{}".format(
                self.misfit_type, it)
            alpha, _ = linesearch.get_best(self.conf, filename)
            alphas.append(alpha)
        return alphas

    def get_model_folder(self, model):
        if model == "init":
            model = 0
        elif model == "last":
            model = self.n_iterations
        return self.conf.get_model_folder(self.misfit_type, model)

    def get_kernel_folder(self, it):
        return self.folder / "kernels_{}_{}".format(self.misfit_type, it)

    def get_update_folder(self, it):
        return self.folder / "update_{}_{}".format(self.misfit_type, it)

    def get_model(self, it, model, fresh=False):
        key = (it, model)
        if fresh or key not in self._models:
            self._models[key] = sp.read_all_from_folder(
                self.get_model_folder(it), model)
        return self._models[key]

    def get_kernel(self, it, kernel, smooth=False, precond=False, fresh=False):
        kname = kernel
        if not kname.endswith("_kernel"):
            kname = kname + "_kernel"
        if smooth:
            kname = kname + "_smooth"
        if precond:
            kname = kname + "_precond"
        key = (it, kname)
        if fresh or key not in self._kernels:
            self._kernels[key] = sp.read_all_from_folder(
                self.get_kernel_folder(it), kname)
        return self._kernels[key]

    def get_update(self, it, update, fresh=False):
        uname = update
        if not uname.endswith("_update"):
            uname = uname + "_update"
        key = (it, uname)
        if fresh or key not in self._updates:
            self._updates[key] = sp.read_all_from_folder(
                self.get_update_folder(it), uname)
        return self._updates[key]

    def get_trace(self, data_type, trace, event, it):
        if self.conf.simulation.seismogram_format == "su":
            return utils.SUReader(self.conf, [event], data_type,
                                  collected=True,
                                  collected_it=it)[".".join([trace, event])]
        else:
            raise NotImplementedError()

    def get_windows(self, event, station):
        windows = utils.load_yaml(self.conf.get_windows_filename(event))
        return windows[station]

    def get_model_misfit_map_values(self, model, param, **kwargs):
        return plot.get_model_misfit_map_values(self.xs, self.zs,
                                                self.get_model(model, param),
                                                self.get_model("target", param),
                                                **kwargs)

    # Plotters
    def plot_model(self, model, param, *args, **kwargs):
        return plot.plot_model(self.xs, self.zs,
                               self.get_model(model, param),
                               conf=self.conf, *args, **kwargs)

    def plot_target(self, param, **kwargs):
        return self.plot_model("target", param, **kwargs)

    def plot_init(self, param, **kwargs):
        return self.plot_model("init", param, **kwargs)

    def plot_init_and_target(self, param,
                             plot_stations_on_init=False,
                             plot_pairs_on_init=False,
                             inch_per_ax=5,
                             **kwargs):
        is_global = kwargs.pop("is_global", False)
        if is_global:
            import cartopy.crs as ccrs
            proj = ccrs.PlateCarree(central_longitude=180)
            fig, (left_ax, right_ax) = plt.subplots(ncols=2, figsize=(2*inch_per_ax, inch_per_ax),
                                                    subplot_kw={'projection': proj})  # NOQA
            coastline_color = kwargs.get("coastline_color", "black")
            left_ax.coastlines(color=coastline_color)
            right_ax.coastlines(color=coastline_color)
        else:
            fig, (left_ax, right_ax) = plt.subplots(ncols=2, figsize=(10, 5))
        self.plot_target(param, title="Target Model",
                         fig=fig, ax=right_ax, **kwargs)
        kwargs["plot_stations"] = plot_stations_on_init
        kwargs["plot_pairs"] = plot_pairs_on_init
        self.plot_init(param, title="Initial Model",
                       fig=fig, ax=left_ax, **kwargs)
        return fig, (left_ax, right_ax)

    def plot_kernel(self, it, param, smooth=False, precond=False, **kwargs):
        plot.plot_kernel(self.xs, self.zs,
                         self.get_kernel(it, param, smooth, precond),
                         conf=self.conf, **kwargs)

    def plot_update(self, it, param, **kwargs):
        plot.plot_kernel(self.xs, self.zs,
                         self.get_update(it, param),
                         conf=self.conf,
                         **kwargs)

    def plot_misfits(self, normalized=False, show=True, fig=None, ax=None):
        misfits = self.get_misfits(normalized)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(range(self.n_iterations+1), misfits)
        ax.set_title("Data misfits")
        ax.set_xlabel("Iteration #")
        ax.set_ylabel("Misfit")

        if show:
            plt.show()

        return fig, ax

    def plot_model_misfits(self, model_param, show=True, fig=None, ax=None):
        misfits = self.get_model_misfits(model_param)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(range(self.n_iterations+1), misfits)
        ax.set_title("Model misfits ({})".format(model_param))
        ax.set_xlabel("Iteration #")
        ax.set_ylabel("Misfit (RMS)")

        if show:
            plt.show()

        return fig, ax

    def plot_linesearch(self, it, show=True):
        steps, misfits = self.get_linesearch(it)

        # sort steps for plotting
        p = steps.argsort()
        steps = steps[p]
        misfits = misfits[p]

        fig, ax = plt.subplots()
        ax.plot(steps, misfits, "o--")
        ax.set_title("Linesearch for Iteration {}".format(it))
        ax.set_xlabel("Step length")
        ax.set_ylabel("Data misfit")

        if show:
            plt.show()

        return fig, ax

    def plot_step_lengths(self,  show=True):
        steps = self.get_step_lengths()
        fig, ax = plt.subplots()
        ax.plot(range(1, self.n_iterations+1), steps)
        ax.set_title("Step Lengths")
        ax.set_xlabel("Iteration #")
        ax.set_ylabel("Step Length")

        if show:
            plt.show()

        return fig, ax

    def plot_rec_weights(self, event, dd_weights=False,
                         cc_for_non_paired=False, cc_for_all=True,
                         title="", is_global=False, output_file=None,
                         **kwargs):
        weights._plot_rec_weights(self.conf, event, dd_weights,
                                  cc_for_non_paired, cc_for_all,
                                  title, is_global, output_file,
                                  **kwargs)

    def plot_all_rec_weights(self, dd_weights=False,
                             cc_for_non_paired=False, cc_for_all=True,
                             title_format="{event}",
                             is_global=False, output_file=None,
                             show=True,
                             ncols=None, nrows=None,
                             inch_per_ax=5,
                             **kwargs):
        events = self.conf.get_event_names()
        n_plots = len(events)
        ncols, nrows = get_optimum_plot_conf(n_plots, ncols, nrows)
        if is_global:
            import cartopy.crs as ccrs
            proj = ccrs.PlateCarree(central_longitude=180)
            fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                                     figsize=(ncols*inch_per_ax,
                                              nrows*inch_per_ax),
                                     subplot_kw={'projection': proj})
        else:
            fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                                     figsize=(ncols*inch_per_ax,
                                              nrows*inch_per_ax))
        i = 0
        if ncols == 1 and nrows == 1:
            axes = [axes]

        for r in range(nrows):
            for c in range(ncols):
                if nrows == 1:  # there is only one row
                    ax = axes[c]
                else:
                    ax = axes[r][c]
                if is_global:
                    coastline_color = kwargs.get("coastline_color", "black")
                    ax.coastlines(color=coastline_color)
                if i >= n_plots:
                    ax.set_visible(False)
                else:
                    event = events[i]
                    title = title_format.format(event=event)
                    self.plot_rec_weights(event, dd_weights,
                                          cc_for_non_paired, cc_for_all,
                                          title=title, fig=fig, ax=ax,
                                          **kwargs)

                i += 1

        if output_file:
            fig.savefig(output_file)

        if show:
            plt.show()

        return fig, axes

    def plot_dd_weights(self, event, number_of_pairs=False,
                        title="", is_global=False, output_file=None,
                        **kwargs):
        if self.has_pairs():
            return weights._plot_dd_weights(self.conf, event, number_of_pairs,
                                            title, is_global, output_file,
                                            **kwargs)
        return None

    def plot_all_dd_weights(self, number_of_pairs=False,
                            title_format="{event}", is_global=False,
                            output_file=None, show=True,
                            ncols=None, nrows=None,
                            inch_per_ax=5,
                            **kwargs):
        if not self.has_pairs():
            return None

        events = self.conf.get_event_names()
        n_plots = len(events)
        ncols, nrows = get_optimum_plot_conf(n_plots, ncols, nrows)
        if is_global:
            import cartopy.crs as ccrs
            proj = ccrs.PlateCarree(central_longitude=180)
            fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                                     figsize=(ncols*inch_per_ax,
                                              nrows*inch_per_ax),
                                     subplot_kw={'projection': proj})
        else:
            fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                                     figsize=(ncols*inch_per_ax,
                                              nrows*inch_per_ax))
        i = 0
        if ncols == 1 and nrows == 1:
            axes = [axes]

        for r in range(nrows):
            for c in range(ncols):
                if nrows == 1:  # there is only one row
                    ax = axes[c]
                else:
                    ax = axes[r][c]
                if is_global:
                    coastline_color = kwargs.get("coastline_color", "black")
                    ax.coastlines(color=coastline_color)
                if i >= n_plots:
                    ax.set_visible(False)
                else:
                    event = events[i]
                    title = title_format.format(event=event)
                    self.plot_dd_weights(event, number_of_pairs,
                                         title=title, fig=fig, ax=ax,
                                         **kwargs)
                i += 1

        if output_file:
            fig.savefig(output_file)

        if show:
            plt.show()

        return fig, axes

    def plot_src_weights(self, title="", is_global=False, output_file=None,
                         **kwargs):
        return weights._plot_src_weights(self.conf,
                                         title, is_global, output_file,
                                         **kwargs)

    def plot_model_misfit_map(self, model, param, normalized=True,
                              limits=None, **kwargs):
        plot.plot_model_misfit_map(self.xs, self.zs,
                                   self.get_model(model, param),
                                   self.get_model("target", param),
                                   normalized=normalized,
                                   limits=limits,
                                   conf=self.conf, **kwargs)

    def plot_improvement(self, model, param, level=None, **kwargs):
        plot.plot_improvement(self.xs, self.zs,
                              self.get_model("target", param),
                              self.get_model("init", param),
                              self.get_model(model, param),
                              conf=self.conf, level=level, **kwargs)

    def _plot_windows(self, window, ax,
                      facecolor="green", alpha=0.3):
        bot, top = ax.get_ylim()
        left, right = window
        rect = mpatches.Rectangle((left, bot), right-left, top-bot,
                                  facecolor=facecolor,
                                  alpha=alpha)
        ax.add_patch(rect)

    def _plot_trace(self, data_type, trace, event, it, color, label,
                    plot_windows=False, fig=None, ax=None,
                    window_color="green", window_alpha=0.3,
                    **kwargs):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        tr = self.get_trace(data_type, trace, event, it)
        ax.plot(tr.times(), tr.data, color, label=label)
        ax.legend()
        if plot_windows:
            windows = self.get_windows(event, trace)
            for window in windows:
                self._plot_windows(window, ax, window_color, window_alpha)
        return fig, ax

    def plot_obsd(self, trace, event, it, plot_windows=False,
                  color="b", label="obsd",
                  **kwargs):
        return self._plot_trace("obs", trace, event, it, color, label,
                                plot_windows, **kwargs)

    def plot_synt(self, trace, event, it, plot_windows=False,
                  color="r--", label="synt",
                  **kwargs):
        return self._plot_trace("syn", trace, event, it, color, label,
                                plot_windows, **kwargs)

    def plot_obsd_and_synt(self, trace, event, it, plot_windows=False,
                           obsd_color="b", obsd_label="obsd",
                           synt_color="r--", synt_label="synt",
                           fig=None, ax=None,
                           **kwargs):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        self.plot_obsd(trace, event, it, plot_windows,
                       obsd_color, obsd_label, fig=fig, ax=ax, **kwargs)
        self.plot_synt(trace, event, it, False,
                       synt_color, synt_label, fig=fig, ax=ax, **kwargs)
        return fig, ax

    def plot_adjoint(self, trace, event, it, plot_windows=False,
                     adjoint_color="g", adjoint_label="Adjoint Source",
                     **kwargs):
        return self._plot_trace("adjoint", trace, event, it,
                                adjoint_color, adjoint_label,
                                plot_windows,
                                **kwargs)

    def plot_adjoint_with_data(self, trace, event, it, plot_windows=False,
                               **kwargs):
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
        self.plot_obsd_and_synt(trace, event, it, plot_windows,
                                fig=fig, ax=ax0,
                                **kwargs)
        self.plot_adjoint(trace, event, it, plot_windows,
                          fig=fig, ax=ax1,
                          **kwargs)
        return fig, (ax0, ax1)


class ComparativeExploration(object):
    """An helper class for comparing different inversions"""
    def __init__(self):
        super(ComparativeExploration, self).__init__()
        self.explorers = OrderedDict()

    @classmethod
    def from_multiplerunfile(cls, filename):
        params = utils.load_yaml(filename)
        dirname = io.dirname_of(filename)
        ce = ComparativeExploration()
        for run in params["runs"]:
            ce.add(run["name"], io.Path(run["name"], dirname))
        return ce

    def __getitem__(self, key):
        return self.explorers[key]

    def add(self, name, folder):
        self.explorers[name] = Explorer(folder)
        return self.explorers[name]

    def get_subset(self, *names):
        """Get subset of the current ComparativeExploration"""
        sub_explorers = OrderedDict()
        for name in names:
            sub_explorers[name] = self.explorers[name]
        ce = ComparativeExploration()
        ce.explorers = sub_explorers
        return ce

    # Getters
    def get_misfits(self, normalized=False):
        misfits = OrderedDict()
        for name, e in self.explorers.iteritems():
            misfits[name] = e.get_misfits(normalized)
        return misfits

    def get_model_misfits(self, model_param):
        misfits = OrderedDict()
        for name, e in self.explorers.iteritems():
            misfits[name] = e.get_model_misfits(model_param)
        return misfits

    def get_linesearchs(self, it):
        linesearchs = OrderedDict()
        for name, e in self.explorers.iteritems():
            linesearchs[name] = e.get_linesearch(it)
        return linesearchs

    def get_step_lengths(self):
        step_lengths = OrderedDict()
        for name, e in self.explorers.iteritems():
            step_lengths[name] = e.get_step_lengths()
        return step_lengths

    @property
    def first(self):
        # returns first explorer
        return self.explorers.values()[0]

    # Plotters
    def plot_misfits(self, normalized=True, show=True, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        for name, misfits in self.get_misfits(normalized).iteritems():
            ax.plot(range(len(misfits)), misfits, label=name)
        ax.legend()
        ax.set_title("Data misfits")
        ax.set_xlabel("Iteration #")
        ax.set_ylabel("Misfit")

        if show:
            plt.show()
        return fig, ax

    def plot_model_misfits(self, model_param, show=True, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        max_n_iter = 0
        for name, misfits in self.get_model_misfits(model_param).iteritems():
            ax.plot(range(len(misfits)), misfits, label=name)
            if len(misfits) - 1 > max_n_iter:
                max_n_iter = len(misfits) - 1
        ax.set_xlim([0, max_n_iter])
        ax.set_xticks(range(max_n_iter+1))
        ax.legend()
        ax.set_title("Model misfits ({})".format(model_param))
        ax.set_xlabel("Iteration #")
        ax.set_ylabel("Misfit (RMS)")
        if show:
            plt.show()
        return fig, ax

    def plot_linesearchs(self, it, show=True):
        fig, ax = plt.subplots()
        for name, search in self.get_linesearchs(it).iteritems():
            steps, misfits = search
            # sort steps for plotting
            p = steps.argsort()
            steps = steps[p]
            misfits = misfits[p]
            ax.plot(steps, misfits, "o--", label=name)
        ax.legend()
        ax.set_title("Linesearch for Iteration {}".format(it))
        ax.set_xlabel("Step length")
        ax.set_ylabel("Data misfit")

        if show:
            plt.show()

        return fig, ax

    def plot_step_lengths(self, show=True):
        fig, ax = plt.subplots()
        for name, steps in self.get_step_lengths().iteritems():
            ax.plot(range(1, self.explorers[name].n_iterations+1), steps,
                    label=name)
        ax.legend()
        ax.set_title("Step Lengths")
        ax.set_xlabel("Iteration #")
        ax.set_ylabel("Step Length")

        if show:
            plt.show()

        return fig, ax

    def plot_init_and_target(self, param, **kwargs):
        return self.first.plot_init_and_target(param, **kwargs)

    def _plot_bin(self, plot_func_name, model=None,
                  param=None, title_format="{name}",
                  output_file=None, show=True,
                  recipe=None, is_global=False,
                  nrows=None, ncols=None,
                  inch_per_ax=5,
                  plot_target_model=False,
                  plot_target_model_title="Target Model",
                  plot_target_param="vs",
                  plot_target_model_kwargs={},
                  **kwargs):
        if recipe is None:
            # recipe format:
            # explorer_name, modelname, plot_func, title, kwargs
            recipe = []
            if plot_target_model:
                recipe.append(RecipeItem(self.explorers.keys()[0],
                                         "target", "plot_model",
                                         plot_target_model_kwargs,
                                         plot_target_param,
                                         plot_target_model_title))
            recipe = build_recipe(self.explorers.keys(),
                                  model, plot_func_name, kwargs,
                                  param,
                                  append_to=recipe)
        n_plots = len(recipe)
        ncols, nrows = get_optimum_plot_conf(n_plots, ncols, nrows)
        if is_global:
            import cartopy.crs as ccrs
            proj = ccrs.PlateCarree(central_longitude=180)
            fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                                     figsize=(ncols*inch_per_ax,
                                              nrows*inch_per_ax),
                                     subplot_kw={'projection': proj})
        else:
            fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                                     figsize=(ncols*inch_per_ax,
                                              nrows*inch_per_ax))

        # This should not happen because this class for
        # comparisons. But, we can support it anyway.
        if ncols == 1 and nrows == 1:
            axes = [axes]

        i = 0
        for r in range(nrows):
            for c in range(ncols):
                if nrows == 1:  # there is only one row
                    ax = axes[c]
                else:
                    ax = axes[r][c]

                if is_global:
                    coastline_color = kwargs.get("coastline_color", "black")
                    ax.coastlines(color=coastline_color)
                if i >= n_plots:
                    ax.set_visible(False)
                else:
                    cur_item = recipe[i]
                    plot_func = getattr(self.explorers[cur_item.name],
                                        cur_item.plot_func_name)
                    if cur_item.title:
                        title = cur_item.title
                    else:
                        title = title_format.format(name=cur_item.name,
                                                    model=cur_item.model,
                                                    **cur_item.kwargs)
                    if cur_item.model:
                        plot_func(cur_item.model,
                                  cur_item.param,
                                  fig=fig, ax=ax,
                                  title=title,
                                  **cur_item.kwargs)
                    else:
                        plot_func(fig=fig, ax=ax, title=title,
                                  **kwargs)
                i += 1

        if output_file:
            fig.savefig(output_file)

        if show:
            plt.show()

        return fig, axes

    def plot_models(self, model, param, title_format="{name}",
                    output_file=None, show=True, **kwargs):

        return self._plot_bin("plot_model", model,
                              param, title_format,
                              output_file=output_file, show=show,
                              **kwargs)

    def plot_kernels(self, it, param, smooth=False, precond=False,
                     title_format="{name}",
                     same_max_value=False, max_value=None,
                     output_file=None, show=True, **kwargs):

        if same_max_value:
            max_value = -1
            for e in self.explorers.values():
                m = np.max(np.abs(e.get_kernel(it, param)))
                if m > max_value:
                    max_value = m
                    print("max_value:", max_value)

        return self._plot_bin("plot_kernel", it, param, title_format,
                              output_file=output_file, show=show,
                              smooth=smooth, precond=precond,
                              max_value=max_value,
                              **kwargs)

    def plot_updates(self, it, param, title_format="{name}",
                     same_max_value=False, max_value=None,
                     output_file=None, show=True, **kwargs):

        if same_max_value:
            max_value = -1
            for e in self.explorers.values():
                m = np.max(np.abs(e.get_update(it, param)))
                if m > max_value:
                    max_value = m

        return self._plot_bin("plot_update", it, param, title_format,
                              output_file=output_file, show=show,
                              max_value=max_value,
                              **kwargs)

    def plot_model_misfit_maps(self, it, param, title_format="{name}",
                               output_file=None, show=True, **kwargs):

        return self._plot_bin("plot_model_misfit_map",
                              it, param, title_format,
                              output_file=output_file, show=show,
                              **kwargs)

    def plot_model_improvements(self, it, param, title_format="{name}",
                                output_file=None, show=True, **kwargs):

        return self._plot_bin("plot_improvement",
                              it, param, title_format,
                              output_file=output_file, show=show,
                              **kwargs)

    def plot_rec_weights(self, event, dd_weights=False,
                         cc_for_non_paired=False, cc_for_all=True,
                         title_format="{name}", is_global=False,
                         output_file=None,
                         **kwargs):
        return self._plot_bin("plot_rec_weights",
                              event=event, dd_weights=dd_weights,
                              cc_for_non_paired=cc_for_non_paired,
                              cc_for_all=cc_for_all,
                              title_format=title_format,
                              is_global=is_global,
                              output_file=output_file,
                              **kwargs)

    def plot_dd_weights(self, event, number_of_pairs=False,
                        title_format="{name}", is_global=False,
                        output_file=None,
                        **kwargs):
        return self._plot_bin("plot_dd_weights",
                              event=event,
                              number_of_pairs=number_of_pairs,
                              title_format=title_format,
                              is_global=is_global,
                              output_file=output_file,
                              **kwargs)

    def plot_src_weights(self, title_format="{name}", is_global=False,
                         output_file=None,
                         **kwargs):
        return self._plot_bin("plot_src_weights",
                              title_format=title_format,
                              is_global=is_global,
                              output_file=output_file,
                              **kwargs)


def explore():
    import argparse
    from IPython import embed
    parser = argparse.ArgumentParser(
        description="Explore inversion results")
    parser.add_argument("-m", "--multiple-runs-file",
                        default=None,
                        help="Multiple run file")
    parser.add_argument("-f", "--folders", nargs="*",
                        default=[],
                        help="folders that contain the results")

    args = parser.parse_args()

    if args.multiple_runs_file:
        ce = ComparativeExploration.from_multiplerunfile(
            args.multiple_runs_file)
    else:
        ce = ComparativeExploration()

    for folder in args.folders:
        ce.add(folder, folder)

    if len(ce.explorers) == 0:
        print("No input is given.")
    else:
        if len(ce.explorers) == 1:
            e = ce.explorers.values()[0]
        embed()


if __name__ == "__main__":
    explore()
