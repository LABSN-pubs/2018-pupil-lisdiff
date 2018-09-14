#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'convenience_functions.py'
===============================================================================

This script defines convenience functions for plotting pupillometry data.
"""
# @author: Dan McCloy (drmccloy@uw.edu)
# Created on Mon Sep 28 09:37:40 2015
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt


def hatch_between(ax, n, x, y1, y2=0, bgcolor=None, hatch=True, **kwargs):
    # ax should be a matplotlib.axes.AxesSubplot object
    kw = (dict(color=bgcolor, edgecolor='none') if bgcolor is not None else
          dict(visible=False))
    mask = ax.fill_between(x, y1, y2, **kw)
    if hatch:
        path = mask.get_paths()[0]
        tran = mask.get_transform()
        xb, yb = ax.get_xbound(), ax.get_ybound()
        xx = np.linspace(xb[0], 2 * xb[1], 2 * n)
        yy = np.linspace(yb[0], 2 * yb[1], 2 * n)
        for x, y in zip(xx[1:], yy[1:]):
            lines = ax.plot((xx[0], x), (y, yy[0]), **kwargs)
            lines[0].set_clip_path(path, tran)
        # if we don't reset axis lims, they change with each call of this
        # function, and the hatch marks on different subplots end up at
        # different angles
        ax.set_xlim(*xb)
        ax.set_ylim(*yb)


def plot_pupil_responses(data, t, color=None, ax=None, alpha=0.4, zorder=None,
                         returns=[]):
    '''data should be shape (observations, conditions, time). Plots 1 line per
       condition, with confidence bands computed across observations.'''
    mean = data.mean(axis=0)
    std = data.std(axis=0) / np.sqrt(len(data) - 1)
    kwargs = dict() if color is None else dict(color=color)
    if zorder is not None:
        kwargs.update(zorder=zorder)
    if ax is None:
        fig, ax = plt.subplots()
    ribbon = ax.fill_between(t, mean - std, mean + std, alpha=alpha,
                             linewidth=0, **kwargs)
    line, = ax.plot(t, mean, **kwargs)
    this_color = line.get_color()
    return_dict = dict(line=line, ribbon=ribbon, color=this_color)
    # only return values for dict keys that actually exist
    return [return_dict[k] for k in (return_dict.keys() & set(returns))]


def plot_trial_timecourse(ax, x, y, w, h, boxcolors, linecolors,
                          linewidth=None, zorder=4):
    '''
    NB: z-order defaults: Artist: 1, Patch(Collection): 2, Line(Collection): 3
    '''
    x = np.array(x)
    y = np.array(y)
    if len(boxcolors) < len(x):
        boxcolors = np.resize(boxcolors, len(x))
    # lines
    xnodes = np.array([x[0] + w/2, x[3] + w, x[4], x[5] + 2*w])
    mnodes = np.repeat(y.max() + h/2, 4) + h/8  # maintain line
    snodes = np.repeat([y.max() + h/2, y.min() + h/2], 2) - h/8  # switch line
    kw = dict(linewidth=linewidth, zorder=zorder)
    ax.plot(xnodes, mnodes, color=linecolors[0], solid_capstyle='butt', **kw)
    ax.plot(xnodes, snodes, color=linecolors[1], linestyle='--', **kw)
    # boxes
    for x0, x1, y0, y1, c in zip(x, x+w, y, y+h, boxcolors):
        ax.fill_between((x0, x1), y0, y1, color=c, edgecolor='none',
                        zorder=(zorder + 1))
    # text
    kw = dict(xytext=(5, 0), textcoords='offset points', ha='left',
              va='center', fontsize=9, zorder=(zorder + 1))
    ax.annotate('maintain', (xnodes[-1], mnodes[-1]), color=linecolors[0],
                **kw)
    ax.annotate('switch', (xnodes[-1], snodes[-1]), color=linecolors[1], **kw)
    # cue
    new_kw = dict(xy=(x[1], y[1]+h), xytext=(0, 1.5), color=boxcolors[0],
                  fontstyle='italic', ha='center', va='bottom')
    kw.update(new_kw)
    ax.annotate('cue', **kw)
    return ax
