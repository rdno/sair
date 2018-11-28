# -*- coding: utf-8 -*-

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs


def grid(x, y, z, resX=500, resY=500):
    """
    Converts 3 column data to grid
    """
    from scipy.interpolate import griddata

    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), z, (X, Y), method='linear')

    return X, Y, Z


def plot_points(ax, xs, ys, size=40, marker="o",
                color="white", edgecolor="black", **kwargs):
    ax.scatter(xs, ys, size, marker=marker,
               color=color, edgecolor=edgecolor, **kwargs)


def plot_sources(ax, conf, size=100,
                 marker="*",
                 color="red",
                 edgecolor="black", **kwargs):
    xs, zs = zip(*conf.get_event_locs().values())
    plot_points(ax, xs, zs, size, marker, color, edgecolor, **kwargs)


def plot_stations(ax, conf, size=50, marker="v",
                  color="white", edgecolor="black", **kwargs):
    stations = conf.get_stations(conf.get_event_names(),
                                 add_event_label=True)
    xs, zs = zip(*stations.values())
    plot_points(ax, xs, zs, size, marker=marker,
                color=color, edgecolor=edgecolor, **kwargs)


def plot_pairs(ax, conf, colors="k", linewidths=0.5, zorder=1):
    data = conf.get_pairs()
    stations = conf.get_stations(conf.get_event_names(),
                                 add_event_label=True)
    pairs = []
    for comp in conf.simulation.comps:
        for pair in data[comp[-1]]:
            pairs.append((pair["window_id_i"].split(":")[0],
                          pair["window_id_j"].split(":")[0]))

    lines = []
    for a, b in pairs:
        st_a = stations[a]
        st_b = stations[b]
        lines.append(([st_a[0], st_a[1]], [st_b[0], st_b[1]]))

    lc = LineCollection(lines, colors=colors,
                        linewidths=linewidths, zorder=zorder)
    ax.add_collection(lc)


def get_raylines(event_loc, receiver_locs):
    sx, sy = event_loc
    return [(event_loc, rec_loc) for rec_loc in receiver_locs]


def plot_raypaths(ax, conf, colors="k", linewidths=0.1, zorder=1):
    rays = []
    for event, loc in conf.get_event_locs().iteritems():
        rays.extend(get_raylines(
            loc, conf.get_stations(event, with_comps=False).values()))

    lc = LineCollection(rays,
                        colors=colors, linewidths=linewidths, zorder=zorder)
    ax.add_collection(lc)


def read_colormap_file(filename, name="custom", reverse=False):
    colormap_data = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                            "colormaps",
                                            "{}.dat".format(filename)))
    if reverse:
        colormap_data = colormap_data[::-1]
    cmap = LinearSegmentedColormap.from_list(name,
                                             colormap_data)
    return cmap


def plot_bin(x, y, z, conf=None,
             plot_stations_on=False, plot_pairs_on=False,
             plot_raypaths_on=False,
             vmax=None, vmin=None,
             title="", colorlabel="",
             auto_boundary=True,
             output_file="",
             fig=None, ax=None, cmap=None,
             events=None, is_global=False, show=True,
             colorbar=True, tight_layout=True,
             coastline_color="black",
             source_options={},
             station_options={},
             pair_options={},
             raypath_options={},
             tight_layout_options=None):

    if vmax is None:
        vmax = np.max(z)
        vmin = np.min(z)

    if cmap is None:
        cmap = read_colormap_file("red_yellow_cyan_blue", reverse=True)

    X, Y, Z = grid(x, y, z)

    fig_is_given = False
    if fig is None and ax is None:
        if is_global:
            projection = ccrs.PlateCarree(
                central_longitude=180)
            ax = plt.axes(projection=projection)
            ax.coastlines(color=coastline_color)
            fig = plt.gcf()
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig_is_given = True

    im = ax.imshow(Z, vmax=vmax, vmin=vmin,
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   cmap=cmap,
                   origin='lower')

    if colorbar:
        if hasattr(ax, "projection"):
            cbar = plt.colorbar(im, fraction=0.024, pad=0.02, ax=ax)
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, ax=ax, cax=cax)
        cbar.set_label(colorlabel)

    if conf is not None:
        if plot_raypaths_on:
            plot_raypaths(ax, conf, **raypath_options)
        if plot_pairs_on:
            try:
                plot_pairs(ax, conf, **pair_options)
            except IOError:
                pass
        if plot_stations_on:
            plot_stations(ax, conf, **station_options)
            plot_sources(ax, conf, **source_options)

    if auto_boundary:
        ax.set_xlim([min(x), max(x)])
        ax.set_ylim([min(y), max(y)])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    if tight_layout:
        if tight_layout_options is None:
            tight_layout_options = {}
        plt.tight_layout(**tight_layout_options)

    if title:
        ax.set_title(title)
    if output_file:
        fig.savefig(output_file)
    elif show and not fig_is_given:
        plt.show()
    return fig, ax
