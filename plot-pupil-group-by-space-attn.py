#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'fig-pupil-capd-attn-by-space-group'
===============================================================================

This script plots pupil size & significance tests.
"""
# @author: Dan McCloy (drmccloy@uw.edu)
# Created on Fri Sep 25 11:15:34 2015
# License: BSD (3-clause)

import yaml
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from convenience_functions import (hatch_between, plot_pupil_responses,
                                   plot_trial_timecourse)
from expyfun.io import read_hdf5

# flags
plot_stderr = True
plot_signif = True
show_pval = False
savefig = True
use_deconv = True

# file I/O
styledir = 'styles'
paramdir = 'params'
outdir = op.join('figures', 'manuscript')
data_file = op.join('processed-data', 'capd-pupil-data.npz')
param_file = op.join(paramdir, 'params.hdf5')
yaml_param_file = op.join(paramdir, 'params.yaml')
deconv = 'deconv' if use_deconv else 'zscore'

# load data
dat = np.load(data_file)
subj_ord, fs, kernel = dat['subjects'], dat['fs'], dat['kernel']
data, time = dat['fits'], dat['t_fit']
data_z = dat['zscores']

'''
# exclude subj 907 (dprimes < 1 in all conds) and their age-matched ldiff subj
new_subj_ord = np.array([x for x in subj_ord if x not in ('9', '907')])
new_data_indices = np.in1d(subj_ord, new_subj_ord)
data = data[new_data_indices]
data_z = data_z[new_data_indices]
subj_ord = new_subj_ord
'''

# yaml params
with open(yaml_param_file, 'r') as yp:
    yaml_params = yaml.load(yp)
    subjects = yaml_params['subjects']  # already in npz file
    listening_difficulty = np.array(yaml_params['lis_diff'])
    spatial_mapping = yaml_params['spatial_mapping']
    talker_mapping = yaml_params['talker_mapping']
    t_min = yaml_params['t_min']
    t_max = yaml_params['t_max']
    t_peak = yaml_params['t_peak']

# hdf5 params
params = read_hdf5(param_file)
''' RELEVANT KEYS:
cond_mat:         array (384, 4). columns are [attn, spatial, idents, gap]
attns:            maintain vs switch
spatials:         -30x30, 30x-30, -30x-30, 30x30
idents:           MM, FF, MF, FM (male/female talkers)
stim_times:       array[0.0, 0.4, 1.2, 1.6, 2.6, 3.]
'''
stim_times = params['stim_times'][0]  # gap included already
cond_mat = params['cond_mat']
# figure params
with open(op.join(styledir, 'figure-params.yaml'), 'r') as fp:
    fig_params = yaml.load(fp)
    signifcol = fig_params['signif_color']
    cue = fig_params['cue_color']
    msk = fig_params['msk_color']
    hatchlwd = fig_params['hatch_linewidth']
    prp = fig_params['prp']  # purple
    pch = fig_params['pch']  # peach
    cyn = fig_params['cyn']  # cyan
    blu = fig_params['blu']  # blue

# plot style. cannot change cycler via plt.style.use(...) after axes created,
# so we define colors as cycler objects to assign directly to axes.
style_files = ('font-libertine.yaml', 'garnish.yaml')
plt.style.use([op.join(styledir, sf) for sf in style_files])
darks = cycler(color=[blu, prp])
lights = cycler(color=[cyn, pch])

# make DataFrame
df = pd.DataFrame()
df['attn'] = np.array(params['attns'])[cond_mat[:, 0]]
df['spat'] = np.array(params['spatials'])[cond_mat[:, 1]]
df['iden'] = np.array(params['idents'])[cond_mat[:, 2]]
df['lr'] = df['spat'].map(spatial_mapping)
df['mf'] = df['iden'].map(talker_mapping)

# align subject data with listening-difficulty data
lisdiff = {s: l for s, l in zip(subjects, listening_difficulty)}
lisdiff_bool = np.array([lisdiff[s] for s in subj_ord], dtype=bool)
# make boolean group subset vectors
groups = {'ldiff': lisdiff_bool, 'control': np.logical_not(lisdiff_bool)}

# groups = {'ldiff': lisdiff_bool,
#           'control': np.logical_not(lisdiff_bool),
#           'all': np.ones_like(lisdiff_bool)}

# prep data contrasts
non_spatial = np.in1d(df['lr'], ['LL', 'RR'])
spatial = np.in1d(df['mf'], ['MM', 'FF'])
both = np.logical_and(np.logical_not(spatial), np.logical_not(non_spatial))
all_conds = np.ones_like(spatial)
spatial_conds = {'non-spatial': non_spatial, 'spatial': spatial, 'mixed': both,
                 'all-trials': all_conds}
attn_conds = {'switch': df['attn'] == 'switch',
              'maintain': df['attn'] == 'maintain'}
renamer = dict(ldiff='listening difficulty')

# time vector
if use_deconv:
    t_max = t_max - t_peak
    ylabel = '“effort” (a.u.)'
else:
    time = t_min + np.arange(data_z.shape[-1]) / float(fs)
    data = data_z
    ylabel = 'pupil size (z-score)'

# axis limits
xlim = (t_min, t_max)
ymax = np.max(data.mean(axis=(0, 1)) + 2 * data.std(axis=(0, 1)) /
              np.sqrt(data.shape[0] - 1))
ylim = (-0.6 * ymax, ymax)

# init figure
fig, axs = plt.subplots(2, 4, figsize=(7.25, 3.5), sharey=True)
fig.subplots_adjust(left=0.08, right=0.92, bottom=0.15, top=0.88, wspace=0.2,
                    hspace=0.5)

# loop over (maintain, switch)
for attn_ix, (a_cond_name, a_cond) in enumerate(attn_conds.items()):
    # loop over (spatial, non-spatial, mixed)
    for space_ix, (s_cond_name, s_cond) in enumerate(spatial_conds.items()):
        ax = axs[attn_ix, space_ix]
        these_colors = []
        ax.set_prop_cycle((darks, lights)[attn_ix])
        # loop over (lisdiff, control, all)
        for group_ix, (group_name, group) in enumerate(groups.items()):
            # subset the data (data shape is (listeners, trials, timepts))
            trial_ix = np.logical_and(a_cond, s_cond)
            this_data = data[group][:, trial_ix].mean(axis=1)
            # plot pupil curves
            z = [8, 5][group_ix]
            this_color, = plot_pupil_responses(this_data, time, ax=ax,
                                               zorder=z, returns=['color'])
            these_colors.append(this_color)
            # add curve labels to plot
            if not space_ix:
                ypos = [0.95, 0.75][group_ix] * ymax
                ax.annotate(renamer.get(group_name, group_name), (-0.2, ypos),
                            xytext=(0, 0), textcoords='offset points',
                            color=this_color, ha='left', va='center',
                            fontsize=9)

        # trial timecourse
        stim_dur = 0.3  # really 0.4, but leave a gap to visually distinguish
        y = [ymax * -0.3] * 6 + [ymax * -0.45] * 4
        x = np.r_[stim_times, stim_times[2:]]
        w = stim_dur
        h = ymax * 0.08
        boxcolors = (cue,) * 2 + (msk,) * 8
        linecolors = ('w', msk) if a_cond_name == 'switch' else (msk, 'w')
        plot_trial_timecourse(ax, x, y, w, h, boxcolors, linecolors)

        # stats
        if plot_signif:
            # load stats
            fname = (f'capd_stats_{deconv}_ldiff-vs-control_{s_cond_name}_'
                     f'{a_cond_name}.yaml')
            with open(op.join('stats', fname), 'r') as f:
                stats = yaml.load(f)
                thresh = stats['thresh']
                clusters = stats['clusters']
                tvals = stats['tvals']
                pvals = np.array(stats['pvals'])
            signif = np.where(np.array([p < 0.05 for p in pvals]))[0]
            signif_clusters = [clusters[s] for s in signif]
            signif_cluster_pvals = pvals[signif]
            # plot stats
            for clu, pv in zip(signif_clusters, signif_cluster_pvals):
                clu = np.array(clu)
                cluster_ymin = ylim[0] * np.ones_like(time[clu])
                maint = data[group][:, np.logical_and(s_cond,
                                                      attn_conds['maintain'])]
                switch = data[group][:, np.logical_and(s_cond,
                                                       attn_conds['switch'])]
                cluster_ymax = np.array([maint.mean(axis=(0, 1)),
                                         switch.mean(axis=(0, 1))])
                cluster_ymax = np.max(cluster_ymax[:, clu], axis=0)
                # do the stats plotting
                hatch_between(ax, 9, time[clu], cluster_ymin,
                              cluster_ymax, linewidth=hatchlwd,
                              color=signifcol, zorder=0)
                if show_pval:
                    pval_x = time[int(np.mean(clu[[0, -1]]))]
                    pval_y = -0.1 * ylim[1]
                    pval_ord = np.trunc(np.log10(pv)).astype(int)
                    pval_txt = f'$p < 10^{{{pval_ord}}}$'
                    ax.text(pval_x, pval_y, pval_txt, ha='center',
                            va='baseline', fontdict=dict(size=10))
                # vertical lines
                if len(signif):
                    for ix in (0, -1):
                        ax.plot((time[clu][ix], time[clu][ix]),
                                (cluster_ymin[ix], cluster_ymax[ix]),
                                linestyle=':', color=signifcol,
                                linewidth=hatchlwd)

        # subplot labels / titles
        if not attn_ix:
            scn = {'all-trials': 'all trials'}.get(s_cond_name, s_cond_name)
            ax.set_title(scn, pad=14)
        letters = ['ABCD', 'EFGH']
        label = f'{letters[attn_ix][space_ix]})'
        label_y = 1.05 if attn_ix else 1.15
        ax.text(0.1, label_y, label, fontweight='bold', ha='right',
                va='bottom', transform=ax.transAxes)

        # fixup ticks
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.xaxis.set_ticks(np.arange(np.ceil(xlim[1])))
        # remove yaxis / ticks / ticklabels near bottom
        ax.set_yticks([0., 0.05, 0.1])
        ax.spines['left'].set_bounds(0, 0.1)
        ax.set_ylim(*ylim)  # have to do this twice, why?

        # annotations
        ytickrange = [-0.1 * ymax, 1.01 * ymax]
        ycenter = 1 - np.diff(ytickrange) / np.diff(ylim) / 2.
        if not space_ix:
            ax.set_ylabel(ylabel, y=ycenter)
            ax.spines['left'].set_visible(True)
        elif space_ix == (axs.shape[-1] - 1):
            ax.set_ylabel(a_cond_name.upper(),
                          rotation=-90, labelpad=28)
            ax.yaxis.set_label_position('right')
            ax.yaxis.set_tick_params(length=0)
        else:
            ax.yaxis.set_visible(False)
        if attn_ix:
            ax.set_xlabel('time (s)')

if savefig:
    fname = f'capd-pupil-{deconv}-group-by-space-attn.pdf'
    fig.savefig(op.join(outdir, fname))
else:
    plt.ion()
    plt.show()
