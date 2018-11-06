#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'plot-rt'
===============================================================================

This script plots behavioral data for the CAPD attention switching
pupillometry experiment.
"""
# @author: Dan McCloy  (drmccloy@uw.edu)
# Created on Fri Aug 10 13:43:29 PDT 2018
# License: BSD (3-clause)

import yaml
import os.path as op
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', None)
savefig = True

# file I/O
indir = 'posthocs'
outdir = op.join('figures', 'manuscript')
styledir = 'styles'

# figure params
with open(op.join(styledir, 'figure-params.yaml'), 'r') as fp:
    fig_params = yaml.load(fp)
    signifcol = fig_params['signif_color']
    prp = fig_params['prp']  # purple
    pch = fig_params['pch']  # peach
    cyn = fig_params['cyn']  # cyan
    blu = fig_params['blu']  # blue
    mgn = fig_params['mgn']  # magenta
    tll = fig_params['tll']  # teal
    orn = fig_params['orn']  # orange
    ylo = fig_params['ylo']  # yellow
    red = fig_params['red']  # red
    grn = fig_params['grn']  # green
    gry = fig_params['gry']  # gray
    dgy = fig_params['dgy']  # dark gray

colors = {'spatial': orn, 'non-spatial': tll, 'mixed': mgn, 'ctrl': prp,
          'ldiff': blu, 'maintain': cyn, 'switch': blu, 'main effect': gry,
          'control': prp, 'listening\ndifficulty': blu, 'non-\nspatial': tll}

# load data
model_summary = pd.read_csv(op.join(indir, 'rt-model-summary.csv'),
                            dtype=dict(slot=object))  # don't convert to float
main_effects = pd.read_csv(op.join(indir, 'main-effect-rt-estimates.csv'))
twoway = pd.read_csv(op.join(indir, 'twoway-rt-estimates.csv'))
threeway = pd.read_csv(op.join(indir, 'threeway-rt-estimates.csv'))
fourway = pd.read_csv(op.join(indir, 'fourway-rt-estimates.csv'))

# parse model summary
model_summary['terms'] = (model_summary[['ldiff', 'attn', 'space', 'slot']]
                          .applymap(lambda x: isinstance(x, str))
                          .apply('sum', axis=1))
model_summary['effect'] = model_summary['terms'].map({0: 'intercept',
                                                      1: 'main effect',
                                                      2: 'twoway',
                                                      3: 'threeway',
                                                      4: 'fourway'})

# merge dataframes
main_effects['effect'] = 'main effect'
twoway['effect'] = 'twoway'
twoway.rename(columns=dict(group='group1'), inplace=True)
threeway['effect'] = 'threeway'
fourway['effect'] = 'fourway'
fourway['group3'] = fourway['group3'].map(str).astype(object)
estimates = pd.concat([main_effects, twoway, threeway, fourway],
                      ignore_index=True)
col_order = ['effect', 'grouping', 'group1', 'group2', 'group3', 'variable',
             'level', 'emmean', 'lower.CL', 'upper.CL', 'SE', 'df']
estimates = estimates[col_order]

# factor order
categories = ['control', 'ctrl', 'ldiff', 'listening\ndifficulty', 'mixed',
              'non-\nspatial', 'non-spatial', 'spatial', 'maintain', 'switch',
              '1', '2', '3', '4']
dt = CategoricalDtype(categories=categories, ordered=True)
estimates['level'] = estimates['level'].astype(dt)
estimates['group1'] = estimates['group1'].astype(dt)
estimates['group2'] = estimates['group2'].astype(dt)
estimates['group3'] = estimates['group3'].astype(dt)
# relative error
estimates['lerr'] = np.abs(estimates['emmean'] - estimates['lower.CL'])
estimates['uerr'] = np.abs(estimates['emmean'] - estimates['upper.CL'])

# style setup
style_files = ('font-libertine.yaml', 'garnish.yaml')
plt.style.use([op.join(styledir, sf) for sf in style_files])
# undo setting from garnish.yaml
plt.style.use({'axes.spines.left': True, 'xtick.labelsize': 8,
               'ytick.labelsize': 8})
# renamers
x_axis_labeller = {'attn': 'Attention condition', 'space': 'Spatial condition',
                   'ldiff': 'Participant group', 'slot': 'Timing slot',
                   'maintain': 'maint.',
                   'listening\ndifficulty': 'lis.\ndiffic.'}
legend_labeller = {'listening\ndifficulty': 'listening difficulty',
                   'control': 'age-matched control',
                   'non-\nspatial': 'non-spatial'}
var_renamer = {'non-\nspatial': 'non-spatial'}
# kwargs
signif_kwargs = dict(xytext=(0, 2), textcoords='offset points', ha='center',
                     va='center', fontsize=9)

# axis lims & dodge
x_dodge = (-0.03, 0, 0.03)
x_gap = 1.2
# ylims = np.array([np.floor(10 * estimates['lower.CL'].min()),
#                   np.ceil(10 * estimates['upper.CL'].max())]) / 10
# yticks = np.linspace(*ylims, int(10 * np.diff(ylims) + 1))
ylims = {0: (0.35, 0.6), 1: (0.35, 0.65)}
yticks = {0: (0.4, 0.5, 0.6), 1: (0.4, 0.5, 0.6)}
height_ratios = [np.diff(yl)[0] for yl in list(ylims.values())]

# init figure
fig, axs = plt.subplots(2, 4, figsize=(6.5, 4), sharey='row',
                        gridspec_kw=dict(height_ratios=height_ratios))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, hspace=1.1,
                    wspace=0.3)

# plot
for ix, variable in enumerate(('attn', 'ldiff', 'space', 'slot')):
    df = estimates.loc[(estimates['effect'] == 'main effect') &
                       (estimates['variable'] == variable)
                       ].sort_values(by='level')
    # plot main effect
    x = np.arange(df.shape[0])
    y = df['emmean']
    ax = axs[0, ix]
    ax.errorbar(x, y, yerr=df[['lerr', 'uerr']].values.T,
                color=colors['main effect'], label=None)
    ax.set_ylim(ylims[0])
    ax.set_yticks(yticks[0])
    ax.set_xticks(x)
    ax.set_xticklabels(df['level'])
    # ax.set_xticklabels([x_axis_labeller.get(xtl, xtl) for xtl in
    #                     df['level']])
    ax.set_xlabel(x_axis_labeller[df['variable'].iloc[0]])
    # signif
    locator = ((model_summary['effect'] == 'main effect') &
               (model_summary[df['variable'].iloc[0]
                              ].map(lambda x: isinstance(x, str))))
    signif = model_summary.loc[locator]
    for ss, s in enumerate(signif['signif']):
        xx = x[(ss+1)] if variable == 'slot' else x[ss:(ss+2)].mean()
        yy = df['upper.CL'].iloc[(ss+1)] if variable == 'slot' else y.mean()
        if isinstance(s, str):
            ax.annotate(s, xy=(xx, yy), color=colors['main effect'],
                        **signif_kwargs)

    # plot interaction
    if variable == 'slot':
        df = estimates.loc[(estimates['effect'] == 'fourway') &
                           (estimates['group1'] == 'switch') &
                           (estimates['group2'] == 'non-spatial')]
        for il, line in enumerate(df['level'].unique()):
            this_line = df.loc[df['level'] == line].sort_values(by='group3')
            x = np.arange(this_line.shape[0]) + x_dodge[il]
            y = this_line['emmean']
            label = this_line['level'].iloc[0]
            ax = axs[1, ix]
            ax.errorbar(x, y, yerr=this_line[['lerr', 'uerr']].values.T,
                        color=colors[line],
                        label=legend_labeller.get(label, label), ls='--')
            # signif
            if not il:
                locator = ((model_summary['effect'] == 'fourway') &
                           (model_summary['space'] == 'non-spatial'))
                signif = model_summary.loc[locator]
                for ss, s in enumerate(signif['signif']):
                    xx = x[(ss+1)]
                    yy = this_line['upper.CL'].iloc[(ss+1)]
                    if isinstance(s, str):
                        ax.annotate(s, xy=(xx, yy), color=colors[line],
                                    **signif_kwargs)
        # garnish
        ax.set_xticks(x)
        ax.set_xticklabels(this_line['group3'])
        ax.set_xlabel(x_axis_labeller['slot'])
    else:
        df = estimates.loc[(estimates['effect'] == 'twoway') &
                           (estimates['grouping'] == 'slot') &
                           (estimates['variable'] == variable)]
        for il, line in enumerate(df['level'].unique()):
            this_line = df.loc[df['level'] == line].sort_values(by='group1')
            x = np.arange(this_line.shape[0]) + x_dodge[il]
            y = this_line['emmean']
            label = this_line['level'].iloc[0]
            ax = axs[1, ix]
            ax.errorbar(x, y, yerr=this_line[['lerr', 'uerr']].values.T,
                        color=colors[line],
                        label=legend_labeller.get(label, label))
            # signif
            mod_sum_var = var_renamer.get(line, line)
            locator = ((model_summary['effect'] == 'twoway') &
                       (model_summary[variable] == mod_sum_var) &
                       (model_summary[this_line['grouping'].iloc[0]
                                      ].map(lambda x: isinstance(x, str))))
            signif = model_summary.loc[locator]
            for ss, s in enumerate(signif['signif']):
                xx = x[(ss+1)]
                yy = this_line['upper.CL'].iloc[(ss+1)]
                if isinstance(s, str):
                    ax.annotate(s, xy=(xx, yy), color=colors[line],
                                **signif_kwargs)
        # garnish
        ax.set_ylim(ylims[1])
        ax.set_yticks(yticks[1])
        ax.set_xticks(x)
        ax.set_xticklabels(this_line['group1'])
        ax.set_xlabel(x_axis_labeller[df['grouping'].iloc[0]])
    if not ix:
        x = fig.subplotpars.left / 3
        y = (fig.subplotpars.top + fig.subplotpars.bottom) / 2
        fig.text(x, y, 'Reaction time (s)', ha='center', va='center',
                 rotation='vertical', fontsize=14)
    # legend. the h[0] hack is to suppress the errorbars in the legend
    hh, ll = ax.get_legend_handles_labels()
    legend = ax.legend(handles=[h[0] for h in hh], labels=ll,
                       loc='lower right', bbox_to_anchor=(1.1, 0.95),
                       fontsize=8, frameon=False, markerfirst=False,
                       labelspacing=0.25, handlelength=1)
    if variable == 'slot':
        legend.set_title('4-WAY INTERACTION:\nnon-spatial switch trials,',
                         prop=dict(size=8))
# subplot tags
for ax in axs.ravel():
    tag = list('ABCDEFGH')[np.where(axs.ravel() == ax)[0][0]]
    ax.annotate(f'({tag})', xy=(0, 0), xytext=(8, 2), xycoords='axes fraction',
                textcoords='offset points', fontsize=9, fontweight='bold',
                va='bottom', ha='center', color=dgy)
# row labels
row_label_kwargs = dict(xy=(1, 0.5), xytext=(8, 0),
                        xycoords='axes fraction', textcoords='offset points',
                        rotation=270, ha='left', va='center', fontsize=10)
axs[0, -1].annotate('MAIN EFFECTS', **row_label_kwargs)
axs[1, -1].annotate('INTERACTIONS', **row_label_kwargs)

fig.align_xlabels()

# finish
if savefig:
    fig.savefig(op.join(outdir, 'rt-figure.pdf'))
else:
    plt.ion()
    plt.show()
