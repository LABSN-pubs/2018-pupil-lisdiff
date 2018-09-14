#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'plot-posthocs'
===============================================================================

This script plots behavioral data for the CAPD attention switching
pupillometry experiment.
"""
# @author: Dan McCloy  (drmccloy@uw.edu)
# Created on Wed Jul 4 16:55:53 PDT 2018
# License: BSD (3-clause)

import yaml
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, pearsonr
from expyfun.analyze import dprime
from expyfun.io import read_hdf5

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', None)


def corrfunc(x, y, **kws):
    if 'color' in kws.keys():
        xy = (0.1, 0.9) if (kws['color'][0] > 0.5) else (0.6, 0.9)
    r, p = pearsonr(x, y)
    if p < 0.05:
        ax = plt.gca()
        ax.annotate(f'r = {r:.2}\np = {p:.2}', xy=xy, xycoords=ax.transAxes,
                    **kws)


# file I/O
figdir = 'figures'
paramdir = 'params'
datadir = 'processed-data'
param_file = op.join(paramdir, 'params.hdf5')
yaml_param_file = op.join(paramdir, 'params.yaml')

# experiment params
with open(yaml_param_file, 'r') as ep:
    params = yaml.load(ep)
    age_matching = params['age_matching']
    subjects = params['subjects']  # already in npz file
    listening_difficulty = np.array(params['lis_diff'])
    spatial_mapping = params['spatial_mapping']
    talker_mapping = params['talker_mapping']

# load behavioral data
slot_df = pd.read_csv(op.join(datadir, 'behavioral-data-by-slot.tsv'),
                      sep='\t')

# drop NaNs (slots with no RT)
slot_nonan = slot_df.dropna(subset=['rt']).copy()
slot_hitonly = slot_nonan[slot_nonan['h'].astype(bool)].copy()

# subj-level dprime summary
cols = ['subj'] + list('hmfc')
hmfc = slot_df[cols].groupby(cols[:-4]).aggregate('sum')
hmfc['dprime'] = hmfc.apply(dprime, axis=1)
hmfc.reset_index(inplace=True)
print(hmfc['dprime'].describe())

# subj-level RT summary
cols = ['subj', 'rt']
rt_summary = slot_hitonly[cols].groupby(cols[:-1]).aggregate('median')
print(rt_summary.describe(), end='\n\n')

# # # # # # # # # # # # # #
# post-hoc: foils by slot #
# # # # # # # # # # # # # #
ph = slot_df.groupby(['lisdiff', 'slot'])['ff'].aggregate('sum').unstack(0)
# ph.plot()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# post-hoc: correlate switch-minus-maintain AUC against slot 4 AUC  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# hdf5 params
params = read_hdf5(param_file)
cond_mat = params['cond_mat']
df = pd.DataFrame()
df['attn'] = np.array(params['attns'])[cond_mat[:, 0]]
df['spat'] = np.array(params['spatials'])[cond_mat[:, 1]]
df['iden'] = np.array(params['idents'])[cond_mat[:, 2]]
df['lr'] = df['spat'].map(spatial_mapping)
df['mf'] = df['iden'].map(talker_mapping)
# load pupil data
pupil_data = np.load(op.join(datadir, 'capd-pupil-data.npz'))
subj_ord = pupil_data['subjects']
times = pupil_data['t_fit']
data = pupil_data['fits']
t_zero_ix = np.where(np.isclose(times, 0))[0][0]
delta_t = times[-1] - times[t_zero_ix]
assert np.array_equal(subj_ord, np.array(subjects))
# make boolean group subset vectors
lisdiff = {s: l for s, l in zip(subjects, listening_difficulty)}
lisdiff_bool = np.array([lisdiff[s] for s in subj_ord], dtype=bool)
groups = {'ldiff': lisdiff_bool, 'control': np.logical_not(lisdiff_bool)}
# prep data contrasts
non_spatial = np.in1d(df['lr'], ['LL', 'RR'])
spatial = np.in1d(df['mf'], ['MM', 'FF'])
both = np.logical_and(np.logical_not(spatial), np.logical_not(non_spatial))
all_conds = np.ones_like(spatial)
spatial_conds = {'non-spatial': non_spatial, 'spatial': spatial, 'mixed': both,
                 'all-trials': all_conds}
attn_conds = {'maintain': df['attn'] == 'maintain',
              'switch': df['attn'] == 'switch'}
age_matched_order = np.array([(k, v) for k, v in age_matching.items()])
# select just the trials we want
ns_sw = np.logical_and(spatial_conds['non-spatial'], attn_conds['switch'])
ns_mt = np.logical_and(spatial_conds['non-spatial'], attn_conds['maintain'])
# collapse across trials
nonsp_switch = data[:, ns_sw].mean(axis=1)
nonsp_maint = data[:, ns_mt].mean(axis=1)
nonsp_smm = nonsp_switch - nonsp_maint  # switch-minus-maintain
# area under the curve (omitting baseline)
nonsp_switch_auc = nonsp_switch[:, t_zero_ix:].sum(axis=-1) / delta_t
nonsp_maint_auc = nonsp_maint[:, t_zero_ix:].sum(axis=-1) / delta_t
nonsp_smm_auc = nonsp_smm[:, t_zero_ix:].sum(axis=-1) / delta_t
# extract slot 4 reaction time
slot4_rts = (slot_hitonly.loc[slot_hitonly['slot'] == 4].groupby(['subj'])
             .aggregate(dict(rt=np.mean)))
slot4_rts.index.name = None
# correlate
cdf = pd.DataFrame(dict(ldiff=listening_difficulty.astype(bool),
                        switch_AUC=nonsp_switch_auc, maint_AUC=nonsp_maint_auc,
                        sw_minus_mt_AUC=nonsp_smm_auc),
                   index=map(int, subjects))
cdf = pd.concat([cdf, slot4_rts], axis=1)
cdf.rename(columns=dict(rt='slot4_sw_nonsp_RT'), inplace=True)
# plot
vars = ['switch_AUC', 'maint_AUC', 'sw_minus_mt_AUC', 'slot4_sw_nonsp_RT']
grid = sns.pairplot(cdf, hue='ldiff', y_vars=vars[:-1], x_vars=vars[-1])
grid.map(corrfunc)
grid.savefig(op.join(figdir, 'slot4-nonspatial-RT-vs-pupil-AUC.pdf'))

# grid = sns.pairplot(cdf[cdf.ldiff], y_vars=vars[:-1], x_vars=vars[-1])
# grid.map(corrfunc)
# grid = sns.pairplot(cdf[np.logical_not(cdf.ldiff)], y_vars=vars[:-1],
#                     x_vars=vars[-1])
# grid.map(corrfunc)

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# post-hoc: correlations with BinDiff data and SSQ  #
# # # # # # # # # # # # # # # # # # # # # # # # # # #
bindiff = read_hdf5(op.join('subject-info', 'from-ross', 'thresholds.hdf5'))
bd_subjs = bindiff['sub_nums']
assert np.all([len(x) == len(bd_subjs) for x in bindiff['thresh'].values()])
bd = pd.DataFrame(bindiff['thresh'], index=bd_subjs)
bd = cdf.merge(bd, left_index=True, right_index=True, how='left')
# SSQ
cols = ['sub_num_dan', 'sub_id_global', 'peak_amplitude_maintain',
        'peak_amplitude_switch', 'peak_latency_maintain',
        'peak_latency_switch']
ssq_cols = [f'ssq{x:02}' for x in range(1, 51)]
ssq_keys = np.repeat(['speech', 'space', 'quality'], (14, 17, 19))
ssq = pd.read_csv(op.join('subject-info', 'from-ross',
                          'export-with-dan-data.csv'),
                  usecols=cols + ssq_cols)
# subset to only our subjects, and reorder columns
ssq = ssq.loc[np.isfinite(ssq['sub_num_dan']), cols + ssq_cols]
ssq = ssq.rename(columns=dict(sub_num_dan='subj', sub_id_global='subj_code'))
ssq['subj'] = ssq['subj'].astype(int)
# make aggregate SSQ measures
speech_cols = [c for c, k in zip(ssq_cols, ssq_keys) if k == 'speech']
space_cols = [c for c, k in zip(ssq_cols, ssq_keys) if k == 'space']
qual_cols = [c for c, k in zip(ssq_cols, ssq_keys) if k == 'quality']
ssq['SSQ_speech'] = ssq[speech_cols].apply('mean', axis=1)
ssq['SSQ_space'] = ssq[space_cols].apply('mean', axis=1)
ssq['SSQ_quality'] = ssq[qual_cols].apply('mean', axis=1)
ssq.drop(ssq_cols + ['subj_code'], axis='columns', inplace=True)
# merge in bindiff
ssq.set_index('subj', inplace=True)
ssq = ssq.merge(bd, left_index=True, right_index=True, how='left')
# plot
var_order = (['switch_AUC', 'maint_AUC', 'sw_minus_mt_AUC',
              'peak_amplitude_maintain', 'peak_amplitude_switch',
              'peak_latency_maintain', 'peak_latency_switch',
              'slot4_sw_nonsp_RT', 'SSQ_speech', 'SSQ_space', 'SSQ_quality'] +
             list(bindiff['thresh'].keys()))
grid = sns.pairplot(ssq.dropna(), vars=var_order, kind='scatter', hue='ldiff',
                    diag_kind='kde', height=2, aspect=1,
                    plot_kws=dict(s=25))
grid.map_lower(corrfunc, fontsize=8)
grid.map_upper(corrfunc, fontsize=8)
grid.fig.subplots_adjust(bottom=0.08, top=0.98, left=0.04, wspace=0.3,
                         hspace=0.3)
grid.fig.align_labels()
grid.savefig(op.join(figdir, 'bindiff-vs-SSQ-vs-pupil.pdf'))
# paired t-tests
with open(op.join('posthocs', 'covariate-t-tests.csv'), 'w') as ff:
    ff.write('# paired samples t-tests of covariates by listener group\n')
    ff.write('# (listening difficulty subjects vs. age-matched controls)\n')
    ff.write('covariate,t,p\n')
    for var in var_order:
        pairs = np.array([(ssq.loc[int(l_subj), var],
                           ssq.loc[int(c_subj), var])
                          for l_subj, c_subj in age_matching.items()]).T
        tval, pval = ttest_rel(*pairs, nan_policy='omit')
        ff.write(f'{var},{np.round(tval, 3)},{np.round(pval, 4)}\n')
