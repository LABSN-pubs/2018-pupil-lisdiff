# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'analyze-behavior'
===============================================================================

This script analyzes behavioral responses.
"""
# @author: Dan McCloy (drmccloy@uw.edu)
# Created on Wed Sep 20 15:19:11 PDT 2017
# License: BSD (3-clause)

import yaml
import numpy as np
import os.path as op
import pandas as pd
from expyfun.io import read_hdf5
from expyfun.analyze import dprime
from scipy.stats import ttest_rel, ttest_ind

# file I/O
data_dir = 'data'
param_file = 'params.hdf5'
yaml_param_file = 'params.yaml'

# yaml params
with open(yaml_param_file, 'r') as yp:
    yaml_params = yaml.load(yp)
    subjects = yaml_params['subjects']
    listening_difficulty = np.array(yaml_params['lis_diff'])
    spatial_mapping = yaml_params['spatial_mapping']
    talker_mapping = yaml_params['talker_mapping']
assert len(subjects) == len(listening_difficulty)

# hdf5 params
params = read_hdf5(param_file)
cond_mat = params['cond_mat']

# make DataFrame
trial_df = pd.DataFrame()
trial_df['attn'] = np.array(params['attns'])[cond_mat[:, 0]]
trial_df['spat'] = np.array(params['spatials'])[cond_mat[:, 1]]
trial_df['iden'] = np.array(params['idents'])[cond_mat[:, 2]]
trial_df['lr'] = trial_df['spat'].map(spatial_mapping)
trial_df['mf'] = trial_df['iden'].map(talker_mapping)

# prep data contrasts
non_spatial = np.in1d(trial_df['lr'], ['LL', 'RR'])
spatial = np.in1d(trial_df['mf'], ['MM', 'FF'])
both = np.logical_and(np.logical_not(spatial), np.logical_not(non_spatial))
all_conds = np.ones_like(spatial)
trial_df['s_cond'] = ''
trial_df.loc[spatial, 's_cond'] = 'spatial'
trial_df.loc[non_spatial, 's_cond'] = 'non-spatial'
trial_df.loc[both, 's_cond'] = 'mixed'
spatial_conds = {'non-spatial': non_spatial, 'spatial': spatial, 'mixed': both,
                 'all-trials': all_conds}
attn_conds = {'maintain': trial_df['attn'] == 'maintain',
              'switch': trial_df['attn'] == 'switch'}
renamer = dict(ldiff='listening difficulty')

# make boolean population subset vectors
groups = {'ldiff': listening_difficulty.astype(bool),
          'control': np.logical_not(listening_difficulty)}
ldiff_subjs = np.array(subjects)[groups['ldiff']].tolist()

# relevant subsets
maint_ix = trial_df.groupby('attn').indices['maintain']
switch_ix = trial_df.groupby('attn').indices['switch']

# load behavioral data
raw_df = pd.DataFrame()
hmfco = []
rt = []
for subj in subjects:
    this_df = trial_df.copy()
    this_df['subj'] = subj
    this_df['trial'] = this_df.index
    this_df['group'] = 'ldiff' if subj in ldiff_subjs else 'control'
    # load HMFCO and RT
    data = read_hdf5(op.join(data_dir, f'{subj}_perf.hdf5'))
    hmfco = data['hmfco']
    # reassign "other" presses as false alarms
    hmfc = hmfco[..., :4].copy()
    hmfc[..., 2] += hmfco[..., 4]
    # merge in HMFC columns
    this_df = pd.concat((this_df, pd.DataFrame(hmfc, columns=list('HMFC'))
                         ), axis=1)
    '''
    # reaction time
    rt_list = data['rt']
    # massage RTs into a friendly shape
    max_n_rt = max([len(x) for x in rt_list])
    rt_array = np.full(hmfco.shape[:-1] + (max_n_rt,), -1.)
    for trial, rts in enumerate(rt_list):
        for press, rt in enumerate(rts):
            rt_array[trial, press] = rt
    rt_ma = np.ma.MaskedArray(rt_array, mask=(rt_array < 0))
    # merge in RT columns
    rt_df = pd.DataFrame(rt_ma, columns=('rt1', 'rt2'))
    this_df = pd.concat((this_df, rt_df), axis=1)
    '''
    # merge subjects together
    raw_df = pd.concat((raw_df, this_df), axis=0, ignore_index=True)

dprimes = dict()
print('d-prime comparisions: maintain vs. switch')
print('=========================================')
# loop over (lisdiff, control, all)
for group_name, group in groups.items():
    dprimes[group_name] = dict()
    # loop over (spatial, non-spatial, both)
    for s_cond_name, s_cond in spatial_conds.items():
        dprimes[group_name][s_cond_name] = dict()
        # loop over (maintain, switch)
        for a_cond_name, a_cond in attn_conds.items():
            dprimes[group_name][s_cond_name][a_cond_name] = list()
            # loop over subjects
            for subj, member in zip(subjects, group):
                if member:
                    indexer = np.logical_and(raw_df['subj'] == subj,
                                             raw_df['attn'] == a_cond_name)
                    if s_cond_name != 'all-trials':
                        s_indexer = (raw_df['s_cond'] == s_cond_name)
                        indexer = np.logical_and(indexer, s_indexer)
                    this_df = raw_df.loc[indexer]
                    this_dprime = dprime(this_df[list('HMFC')].sum(axis=0))
                    dprimes[group_name
                            ][s_cond_name][a_cond_name].append(this_dprime)
# t-test
for group_name in groups:
    for s_cond_name in spatial_conds:
        t, p = ttest_rel(dprimes[group_name][s_cond_name]['maintain'],
                         dprimes[group_name][s_cond_name]['switch'])
        contrast = '{} {}, maint vs. switch'.format(group_name, s_cond_name)
        print('{:48} p={}'.format(contrast, np.round(p, 3)))

print()
print('d-prime comparisions: lisdiff vs. control')
print('=========================================')

# loop over (spatial, non-spatial, both)
for s_cond_name in spatial_conds:
    # loop over (maintain, switch)
    for a_cond_name in attn_conds:
        # t-test
        t, p = ttest_ind(dprimes['ldiff'][s_cond_name][a_cond_name],
                         dprimes['control'][s_cond_name][a_cond_name])
        contrast = '{} {}, ldiff vs. control'.format(s_cond_name, a_cond_name)
        print('{:48} p={}'.format(contrast, np.round(p, 3)))

dprime_df = pd.DataFrame()
for group_name in groups:
    group_df = pd.DataFrame()
    for s_cond_name in spatial_conds:
        this_df = pd.DataFrame(dprimes[group_name][s_cond_name]).T
        this_df.index.name = 'attn'
        this_df['space'] = s_cond_name
        this_df.set_index('space', append=True, inplace=True)
        group_df = pd.concat((group_df, this_df))
    group_df['group'] = group_name
    group_df.set_index('group', append=True, inplace=True)
    dprime_df = pd.concat((dprime_df, group_df))
dprime_df = dprime_df.reorder_levels(('group', 'space', 'attn'))
