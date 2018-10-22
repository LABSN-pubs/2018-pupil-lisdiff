# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'pupil-stats'
===============================================================================

This script runs non-parametric stats on pupillometry time series.
"""
# @author: Dan McCloy (drmccloy@uw.edu)
# Created on Tue Sep 19 16:37:00 PDT 2017
# License: BSD (3-clause)

import yaml
import numpy as np
import os.path as op
from os import makedirs
from pandas import DataFrame
from scipy.stats import distributions
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       spatio_temporal_cluster_test)
from expyfun.io import read_hdf5


def run_permutation_test(data, fname, twosamp=False):
    # setup
    if twosamp:
        threshold = distributions.f.ppf(q=0.95, dfn=len(data) - 1,
                                        dfd=len(data) * (data.shape[1] - 1))
        n_permutations = 2 ** 14
    else:
        threshold = -1 * distributions.t.ppf(q=0.05 / 2, df=len(data) - 1)
        n_permutations = 'all'
    kw = dict(threshold=threshold, n_jobs=8, buffer_size=None,
              n_permutations=n_permutations)
    permute_fun = (spatio_temporal_cluster_test if twosamp else
                   spatio_temporal_cluster_1samp_test)
    # run permutation test
    tvals, clusters, pvals, H0 = permute_fun(data, **kw)
    # we only need x[0] in clusters because this is 1-D data;
    # x[1] in these clusters is just a list of all zeros (no
    # spatial connectivity). All the hacky conversions to
    # float, int, and list are because yaml doesn't understand
    # numpy dtypes.
    clusters = [[int(y) for y in x[0]] for x in clusters]
    stats = dict(thresh=float(threshold), clusters=clusters,
                 n_clusters=len(clusters),
                 tvals=tvals.tolist(), pvals=pvals.tolist())
    with open(fname, 'w') as f:
        yaml.dump(stats, stream=f)


# file I/O
paramdir = 'params'
datadir = 'processed-data'
statdir = 'stats'
param_file = op.join(paramdir, 'params.hdf5')
yaml_param_file = op.join(paramdir, 'params.yaml')
data_file = op.join(datadir, 'capd-pupil-data.npz')
makedirs(statdir, exist_ok=True)

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
    age_matching = yaml_params['age_matching']
    use_deconv = yaml_params['use_deconv']
deconv = 'deconv' if use_deconv else 'zscore'

# hdf5 params
params = read_hdf5(param_file)
cond_mat = params['cond_mat']
df = DataFrame()
df['attn'] = np.array(params['attns'])[cond_mat[:, 0]]
df['spat'] = np.array(params['spatials'])[cond_mat[:, 1]]
df['iden'] = np.array(params['idents'])[cond_mat[:, 2]]
df['lr'] = df['spat'].map(spatial_mapping)
df['mf'] = df['iden'].map(talker_mapping)

# load data
data = np.load(data_file)
data_deconv, t_fit, subj_ord = data['fits'], data['t_fit'], data['subjects']
data_z, fs, kernel = data['zscores'], data['fs'], data['kernel']

'''
# exclude subj 907 (dprimes < 1 in all conds) and their age-matched ldiff subj
new_subj_ord = np.array([x for x in subj_ord if x not in ('9', '907')])
new_data_indices = np.in1d(subj_ord, new_subj_ord)
data_deconv = data_deconv[new_data_indices]
data_z = data_z[new_data_indices]
subj_ord = new_subj_ord
'''

# align subject data with listening-difficulty data, and
# make boolean group subset vectors
lisdiff = {s: l for s, l in zip(subjects, listening_difficulty)}
lisdiff_bool = np.array([lisdiff[s] for s in subj_ord], dtype=bool)
groups = {'ldiff': lisdiff_bool, 'control': np.logical_not(lisdiff_bool)}
# groups = {'ldiff': lisdiff_bool,
#           'control': np.logical_not(lisdiff_bool),
#           'all-subjs': np.ones_like(lisdiff_bool)}

# make sure loaded data is in the right order to align matched controls
assert np.array_equal(subj_ord, np.array(subjects))

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

# prep data
t_z = t_min + np.arange(data_z.shape[-1]) / float(fs)
t = t_fit if use_deconv else t_z
data = data_deconv if use_deconv else data_z

'''
# loop over (maintain, switch), comparing ldiff vs control
for a_cond_name, a_cond in attn_conds.items():
    label = '-vs-'.join(groups.keys())
    fname = f'capd_stats_{deconv}_{label}_{a_cond_name}.yaml'
    fpath = op.join(work_dir, 'stats', fname)
    # Add third  axis (fake spatial axis)
    l_data = data[np.ix_(groups['ldiff'], a_cond)].mean(axis=1)
    c_data = data[np.ix_(groups['control'], a_cond)].mean(axis=1)
    """
    group_data = np.array((l_data[:, :, np.newaxis], c_data[:, :, np.newaxis]))
    run_permutation_test(group_data, fpath, twosamp=True)
    """
    diff_data = (l_data - c_data)[:, :, np.newaxis]
    run_permutation_test(diff_data, fpath, twosamp=False)
# loop over (ldiff, control), comparing maintain vs switch
for group_name, group in groups.items():
    label = '-vs-'.join(attn_conds.keys())
    fname = f'capd_stats_{deconv}_{label}_{group_name}.yaml'
    fpath = op.join(work_dir, 'stats', fname)
    # within-subject difference between conditions. Add third  axis (fake
    # spatial axis) in order to use spatio_temporal_cluster_1samp_test
    m_ix = df.groupby('attn').indices['maintain']
    s_ix = df.groupby('attn').indices['switch']
    contr_diff = (data[groups[group_name]][:, m_ix].mean(axis=1) -
                  data[groups[group_name]][:, s_ix].mean(axis=1)
                  )[:, :, np.newaxis]
    run_permutation_test(contr_diff, fpath)
'''

# compare ldiff/control within spatial, non-spatial, mixed, and all trials
for a_cond_name, a_cond in attn_conds.items():
    for s_cond_name, s_cond in spatial_conds.items():
        fname = (f'capd_stats_{deconv}_ldiff-vs-control_'
                 f'{s_cond_name}_{a_cond_name}.yaml')
        fpath = op.join(statdir, fname)
        # across-subject-pairs difference. Add third axis (fake
        # spatial axis) in order to use spatio_temporal_cluster_1samp_test
        _cond = np.logical_and(a_cond, s_cond)
        contr_diff = (data[groups['ldiff']][:, _cond].mean(axis=1) -
                      data[groups['control']][:, _cond].mean(axis=1)
                      )[:, :, np.newaxis]
        run_permutation_test(contr_diff, fpath)

# compare maintain/switch within spatial, non-spatial, mixed, and all trials
for group_name, group in groups.items():
    for s_cond_name, s_cond in spatial_conds.items():
        fname = (f'capd_stats_{deconv}_maintain-vs-switch_'
                 f'{s_cond_name}_{group_name}.yaml')
        fpath = op.join(statdir, fname)
        # within-subject difference between conditions. Add third axis (fake
        # spatial axis) in order to use spatio_temporal_cluster_1samp_test
        m_cond = np.logical_and(s_cond, attn_conds['maintain'])
        s_cond = np.logical_and(s_cond, attn_conds['switch'])
        contr_diff = (data[group][:, m_cond].mean(axis=1) -
                      data[group][:, s_cond].mean(axis=1))[:, :, np.newaxis]
        run_permutation_test(contr_diff, fpath)

'''
# compare difference waves (maintain minus switch) for each spatial cond
for s_cond_name, s_cond in spatial_conds.items():
    m_ix = np.logical_and(s_cond, attn_conds['maintain'])
    s_ix = np.logical_and(s_cond, attn_conds['switch'])
    s_minus_m = data[:, s_ix].mean(axis=1) - data[:, m_ix].mean(axis=1)
    fname = (f'capd_stats_{deconv}_switch-minus-maintain_'
             f'{s_cond_name}_ldiff-vs-control.yaml')
    fpath = op.join(work_dir, 'stats', fname)
    """
    # F-test
    group_data = np.array((s_minus_m[groups['ldiff']][:, :, np.newaxis],
                           s_minus_m[groups['control']][:, :, np.newaxis]))
    run_permutation_test(group_data, fpath, twosamp=True)
    """
    # paired-samples t-test
    diff_data = np.array(s_minus_m[groups['ldiff']] -
                         s_minus_m[groups['control']])[:, :, np.newaxis]
    run_permutation_test(diff_data, fpath, twosamp=False)
'''
