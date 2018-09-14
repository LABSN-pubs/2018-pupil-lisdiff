#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'plot-behavior'
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
from expyfun.analyze import dprime
import matplotlib.pyplot as plt


def probit(x):
    from scipy.stats import norm
    return norm.ppf(x)


pd.set_option('display.max_columns', 50)
pd.set_option('display.width', None)
savefig = True
do_probit = True

# file I/O
indir = 'posthocs'
datadir = 'processed-data'
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
          'control': prp, 'listening difficulty': blu}

model_summary = pd.read_csv(op.join(indir, 'model-summary.csv'))
main_effects = pd.read_csv(op.join(indir, 'main-effects-estimates.csv'))
interactions = pd.read_csv(op.join(indir, 'interaction-estimates.csv'))
slot_df = pd.read_csv(op.join(datadir, 'behavioral-data-by-slot.tsv'),
                      sep='\t')
# maineff_posthocs = pd.read_csv('main-effects-posthocs.csv')
# interact_posthocs = pd.read_csv('interaction-posthocs.csv')

# parse model summary
model_summary['terms'] = (model_summary[['ldiff', 'attn', 'space']]
                          .applymap(lambda x: isinstance(x, str))
                          .apply('sum', axis=1))
model_summary['effect'] = model_summary['terms'].map({0: 'intercepts',
                                                      1: 'main effect',
                                                      2: 'interaction'})

# merge dataframes
main_effects['effect'] = 'main effect'
interactions['effect'] = 'interaction'
estimates = pd.concat([main_effects, interactions])
col_order = ['effect', 'grouping', 'group', 'variable', 'level', 'token',
             'prob', 'asymp.LCL', 'asymp.UCL', 'SE', 'df']
estimates = estimates[col_order]
# factor order
categories = ['control', 'ctrl', 'ldiff', 'listening\ndifficulty', 'mixed',
              'non-\nspatial', 'non-spatial', 'spatial', 'maintain', 'switch']
dt = CategoricalDtype(categories=categories, ordered=True)
estimates['level'] = estimates['level'].astype(dt)
estimates['group'] = estimates['group'].astype(dt)
# relative error
estimates['probit'] = probit(estimates['prob'])
if do_probit:
    estimates['lerr'] = np.abs(estimates['probit'] -
                               probit(estimates['asymp.LCL']))
    estimates['uerr'] = np.abs(estimates['probit'] -
                               probit(estimates['asymp.UCL']))
else:
    estimates['lerr'] = np.abs(estimates['prob'] - estimates['asymp.LCL'])
    estimates['uerr'] = np.abs(estimates['prob'] - estimates['asymp.UCL'])

# style setup
style_files = ('font-libertine.yaml', 'font-latex.yaml', 'garnish.yaml')
plt.style.use([op.join(styledir, sf) for sf in style_files])
# undo setting from garnish.yaml
plt.style.use({'axes.spines.left': True, 'xtick.labelsize': 8,
               'ytick.labelsize': 8})
# renamers
x_axis_labeller = {'attn': 'Attention condition', 'space': 'Spatial condition',
                   'ldiff': 'Participant group', 'maintain': 'maint.',
                   'listening\ndifficulty': 'lis.\ndiffic.'}
legend_labeller = {'ldiff': 'listening difficulty',
                   'ctrl': 'age-matched control'}
# kwargs
signif_kwargs = dict(xytext=(0, 4), textcoords='offset points', ha='center',
                     va='bottom', fontsize=9)

# axis lims & dodge
x_dodge = (-0.04, 0, 0.04)
x_gap = 1.2
x_pad = 0.1
zero = 2e-2 if do_probit else 0.
fzero = 1.5e-2 if do_probit else 0.
ylims = dict(target=(0.75, 0.95), foil=(fzero, 0.12), neither=(zero, 0.1))
yticks = dict(target=(0.75, 0.8, 0.85, 0.9, 0.95), foil=(zero, 0.05, 0.1),
              neither=(zero, 0.05, 0.1), dprime=range(5))
yls = probit(list(ylims.values())) if do_probit else list(ylims.values())
ylims.update(dict(dprime=(0, 4)))  # add this after the probit in yls
width_ratios = np.array([2, 2, 3]) * 2 + x_gap + 1.5 * x_pad
height_ratios = [np.diff(yl)[0] for yl in yls]
height_ratios += [np.mean(height_ratios)]  # height of dprime row

# init figure
fig, axs = plt.subplots(4, 3, figsize=(6.5, 4.5), sharey='row', sharex='col',
                        gridspec_kw=dict(width_ratios=width_ratios,
                                         height_ratios=height_ratios))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95, hspace=0.3,
                    wspace=0.2)

# plot
for ax_row, token in zip(axs, ('target', 'foil', 'neither', 'dprime')):
    for ax, variable in zip(ax_row, ('attn', 'ldiff', 'space')):
        if token == 'dprime':
            column_dict = dict(attn='attn', ldiff='group', space='s_cond')
            var = column_dict[variable]
            # main effect
            cols = ['subj', var] + list('hmfc')
            hmfc = slot_df[cols].groupby(cols[:-4]).aggregate('sum')
            hmfc['dprime'] = hmfc.apply(dprime, axis=1)
            hmfc_main = hmfc.reset_index().sort_values([var, 'subj'])
            # interaction
            ivar = estimates.loc[(estimates['variable'] == variable) &
                                 (estimates['effect'] == 'interaction'),
                                 'grouping'].iat[0]
            inter_var = column_dict[ivar]
            cols = ['subj', var, inter_var] + list('hmfc')
            hmfc = slot_df[cols].groupby(cols[:-4]).aggregate('sum')
            hmfc['dprime'] = hmfc.apply(dprime, axis=1)
            hmfc_int = hmfc.reset_index().sort_values([var, inter_var, 'subj'])
            # combine
            gb_main = hmfc_main[[var, 'dprime']].groupby(var)
            gb_int = hmfc_int[[var, inter_var, 'dprime']
                              ].groupby([var, inter_var])
            means = pd.concat([gb_main.aggregate('mean'),
                               gb_int.aggregate('mean')])
            stds = pd.concat([gb_main.aggregate('std'),
                              gb_int.aggregate('std')])
            # plot
            n = len(gb_main)
            m = len(gb_int) // n  # integer divide OK; guaranteed no remainder
            w = 0.25
            x = (np.repeat(np.arange(n * 2), np.repeat((1, m), n)) +
                 np.array([0] * n + [x_gap - 1] * n * m))
            offsets = np.concatenate((np.zeros(n),
                                      np.tile(np.arange(m)*w - (m-1)*w/2, n)))
            x += offsets
            keys = (np.repeat(['main effect'], n),
                    np.tile(hmfc_int[inter_var].unique().astype(str), n))
            this_colors = [colors[k] for k in np.concatenate(keys)]
            ax.bar(x, means['dprime'], yerr=stds['dprime'], width=w,
                   color=this_colors, error_kw=dict(size=0.1, capsize=1.5,
                   alpha=0.3))
        else:
            df = estimates.loc[(estimates['token'] == token) &
                               (estimates['variable'] == variable)]
            xticks = []
            xticklabels = []
            # plot main effect
            this_df = df.loc[df['effect'] == 'main effect'
                             ].sort_values(by='level')
            x = np.arange(this_df.shape[0])
            y = 'probit' if do_probit else 'prob'
            ax.errorbar(x=x, y=this_df[y],
                        yerr=this_df[['lerr', 'uerr']].values.T,
                        color=colors['main effect'], label=None)
            xticks.extend(x)
            xticklabels.extend(this_df['level'].unique())
            # signif
            locator = ((model_summary['truth'] == token) &
                       (model_summary['effect'] == 'main effect') &
                       (model_summary[this_df['variable'].iloc[0]
                                      ].map(lambda x: isinstance(x, str))))
            signif = model_summary.loc[locator]
            for ss, s in enumerate(signif['signif']):
                if isinstance(s, str):
                    ax.annotate(s, xy=(x[ss:(ss+2)].mean(), this_df[y].mean()),
                                color=colors['main effect'], **signif_kwargs)
            # plot interaction
            this_df = df.loc[df['effect'] == 'interaction'
                             ].sort_values(by='group')
            for ig, group in enumerate(this_df['group'].unique()):
                this_line = this_df.loc[this_df['group'] == group
                                        ].sort_values(by='level')
                xx = (x.max() + x_gap + np.arange(this_line.shape[0]) +
                      x_dodge[ig])
                label = this_line['group'].iloc[0]
                ax.errorbar(x=xx, y=this_line[y],
                            yerr=this_line[['lerr', 'uerr']].values.T,
                            color=colors[group],
                            label=legend_labeller.get(label, label))
            xticks.extend(xx - x_dodge[ig])
            xticklabels.extend(this_line['level'].unique())
            # signif
            locator = ((model_summary['truth'] == token) &
                       (model_summary['effect'] == 'interaction') &
                       (model_summary[this_line['variable'].iloc[0]
                                      ].map(lambda x: isinstance(x, str))) &
                       (model_summary[this_line['grouping'].iloc[0]
                                      ].map(lambda x: isinstance(x, str))))
            signif = model_summary.loc[locator]
            for ss, s in enumerate(signif['signif']):
                this_x = xx[ss:(ss+2)] if this_line.shape[0] > 2 else xx
                if isinstance(s, str):
                    ax.annotate(s, xy=(this_x.mean(), this_line[y].mean()),
                                color=colors[group], **signif_kwargs)
            # garnish
            ax.set_xticks(xticks)
            ax.set_xticklabels([x_axis_labeller.get(xtl, xtl) for xtl in
                                xticklabels])
        if do_probit:
            if token == 'dprime':
                ax.set_ylim(ylims[token])
            else:
                ax.set_ylim(probit(ylims[token]))
                ax.set_yticks(probit(yticks[token]))
                ax.set_yticklabels([f'{y:.2f}' for y in yticks[token]])
        else:
            ax.set_ylim(ylims[token])
            ax.set_yticks(yticks[token])
        # x labels
        if token == 'dprime':
            ax.set_xlabel(x_axis_labeller[variable])
        else:
            # remove x tickmarks
            ax.tick_params(axis='x', bottom=False)
        # make room for subplot tags
        xlim = np.array(ax.get_xlim()) - np.array((x_pad, -0.5*x_pad))
        ax.set_xlim(xlim)
        # subplot tags
        tag = list('ABCDEFGHIJKL')[np.where(axs.ravel() == ax)[0][0]]
        ax.annotate(tag, xy=(0, 0), xytext=(6, 1), xycoords='axes fraction',
                    textcoords='offset points', fontsize=10, va='bottom',
                    ha='center', color=dgy)
        # legend
        if token == 'target':
            # the h[0] hack is to suppress the errorbars in the legend
            hh, ll = ax.get_legend_handles_labels()
            legend = ax.legend(handles=[h[0] for h in hh], labels=ll,
                               loc='upper right', bbox_to_anchor=(1, 1.3),
                               fontsize=8, frameon=False, markerfirst=False,
                               labelspacing=0.3)
    # row labels
    row_labels = dict(dprime='d-prime')
    ax.annotate(row_labels.get(token, token).upper(), xy=(1, 0.5),
                xytext=(8, 0), xycoords='axes fraction',
                textcoords='offset points', rotation=270, ha='left',
                va='center', fontsize=10)
    # y axis label
    if token == 'dprime':
        ax_row[0].set_ylabel('d′', labelpad=10)
    elif token == 'foil':
        ps = ', probit-scaled' if do_probit else ''
        label = r'$P\thinspace(\mathdefault{{button\ press}})${}'.format(ps)
        # label = r'$\Pr(\mathdefault{{button\ press}})${}'.format(ps)
        # label = 'ℙ(button press){}'.format(ps)
        ax_row[0].set_ylabel(label, labelpad=10)

fig.align_labels()

if savefig:
    p = '-probit' if do_probit else ''
    fname = f'accuracy-figure{p}.pdf'
    fig.savefig(op.join(outdir, fname))
else:
    plt.ion()
    plt.show()
