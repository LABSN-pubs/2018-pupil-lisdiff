# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'pupil-metrics'
===============================================================================

This script extracts peak latency, peak amplitude, and area under the curve.
"""
# @author: Dan McCloy (drmccloy@uw.edu)
# Created on Wed Sep 20 10:53:16 PDT 2017
# License: BSD (3-clause)

import yaml
import numpy as np
import os.path as op
from pandas import DataFrame
from expyfun.io import read_hdf5
from scipy.stats import ttest_rel

use_deconv = False

# file I/O
paramdir = 'params'
datadir = 'processed-data'
param_file = op.join(paramdir, 'params.hdf5')
yaml_param_file = op.join(paramdir, 'params.yaml')
data_file = op.join(datadir, 'capd-pupil-data.npz')
deconv = 'deconv' if use_deconv else 'zscore'

# yaml params
with open(yaml_param_file, 'r') as yp:
    yaml_params = yaml.load(yp)
    subjects = yaml_params['subjects']  # already in npz file
    listening_difficulty = np.array(yaml_params['lis_diff'])
    t_min = yaml_params['t_min']
    t_max = yaml_params['t_max']
    t_peak = yaml_params['t_peak']
lisdiff = {s: l for s, l in zip(subjects, listening_difficulty)}

# hdf5 params
params = read_hdf5(param_file)

# make DataFrame
trial_df = DataFrame()
trial_df['attn'] = np.array(params['attns'])[params['cond_mat'][:, 0]]

# load data
data = np.load(data_file)
data_deconv, t_fit, subj_ord = data['fits'], data['t_fit'], data['subjects']
data_zscore, fs, kernel = data['zscores'], data['fs'], data['kernel']

# prep data
t_zs = t_min + np.arange(data_zscore.shape[-1]) / float(fs)
t = t_fit if use_deconv else t_zs
t_zero_ix = np.where(t == 0)[0]
delta_t = t[-1] - t[t_zero_ix]
data = data_deconv if use_deconv else data_zscore

'''
# by trial
auc = data.sum(axis=-1) / delta_t
peak_amplitude = data.max(axis=-1)
peak_latency = t[np.argmax(data, axis=-1)]
'''

# by condition
data_maint = data[:, trial_df.groupby('attn').indices['maintain']].mean(axis=1)
data_switch = data[:, trial_df.groupby('attn').indices['switch']].mean(axis=1)
auc_maintain = data_maint.sum(axis=-1) / delta_t
auc_switch = data_switch.sum(axis=-1) / delta_t
peak_amplitude_maintain = data_maint.max(axis=-1)
peak_amplitude_switch = data_switch.max(axis=-1)
peak_latency_maintain = t[np.argmax(data_maint, axis=-1)]
peak_latency_switch = t[np.argmax(data_switch, axis=-1)]

cols = dict(auc_maintain=auc_maintain,
            auc_switch=auc_switch,
            peak_amplitude_maintain=peak_amplitude_maintain,
            peak_amplitude_switch=peak_amplitude_switch,
            peak_latency_maintain=peak_latency_maintain,
            peak_latency_switch=peak_latency_switch)
df = DataFrame(cols, index=subjects)
df['lisdiff'] = [bool(lisdiff[s]) for s in df.index]
df.to_csv(op.join(datadir, 'pupil-metrics-by-subj.csv'))

# t-tests
for column in df.columns[:-1]:
    t, p = ttest_rel(df.loc[df['lisdiff'], column],
                     df.loc[np.logical_not(df['lisdiff']), column])
    print('{:24} p={}'.format(column, np.round(p, 3)))
