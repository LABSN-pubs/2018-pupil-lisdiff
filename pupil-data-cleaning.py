#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'analyze-pupil-data'
===============================================================================

This script cleans and epochs pupillometry data for the talker/space switching
experiment.
"""
# @author: Eric Larson (larsoner@uw.edu)
# @author: Dan McCloy  (drmccloy@uw.edu)
# Created on Thu Sep 17 15:47:08 2015
# License: BSD (3-clause)


import time
import yaml
import os.path as op
from glob import glob

import numpy as np
from pyeparse import read_raw, Epochs
from pyeparse.utils import pupil_kernel
from expyfun import decimals_to_binary
from expyfun.io import read_hdf5, read_tab
from pupil_helper_functions import get_gaze_angle


def ttl_to_id(val):
    """Convert the val id to a task ID"""
    return (np.array(val) * np.array([1000, 100, 10, 1])).sum()


def get_pupil_data_files(subj, data_dir):
    edf_files = sorted(glob(op.join(data_dir, '{}_*'.format(subj), '*.edf')))
    return edf_files


# flags
downsample = False
n_jobs = 4  # for parallelizing epochs.resample and epochs.deconvolve

# file I/O
indir = 'data'
outdir = 'processed-data'
paramdir = 'params'
param_file = op.join(paramdir, 'params.hdf5')
yaml_param_file = op.join(paramdir, 'params.yaml')

# params
with open(yaml_param_file, 'r') as yp:
    yaml_params = yaml.load(yp)
    subjects = yaml_params['subjects']
    lis_diff = np.array(yaml_params['lis_diff'])
    t_min = yaml_params['t_min']
    t_max = yaml_params['t_max']
    t_peak = yaml_params['t_peak']
assert len(subjects) == len(lis_diff)

# SUBJ 504:
# WARNING: EDF FILE MAY BE CORRUPTED.
# Samples 3030370 found without start recording
# ... 3030370-3030498

deconv_time_pts = None
fs_in = 1000.0
fs_out = 25.  # based on characterize-freq-content.py; no appreciable energy
              # above 3 Hz in average z-score data or kernel  (analysis:ignore)
fs_out = fs_out if downsample else fs_in

# physical details of the eyetracker/screen setup
# (for calculating gaze deviation from fixation cross in degrees)
screenprops = dict(dist_cm=50., width_cm=53., height_cm=29.8125,
                   width_px=1920, height_px=1080)

# load trial info
params = read_hdf5(param_file)
''' KEYS:
block_trials:     dict; keys are codes for different experiment conditions
                  ['LF0', 'LF1', 'LM0', 'LM1', 'RF0', 'RF1', 'RM0', 'RM1']
                  values are lists of ints (trial numbers)
blocks:           list of lists; elements drawn from the exp. condition codes
cond_mat:         array (384, 4). columns are [attn, spatial, idents, gap]
cond_readme:      "attn, spatial, idents, gap"
attns:            maintain vs switch
spatials:         -30x30, 30x-30, -30x-30, 30x30
idents:           MM, FF, MF, FM (male/female talkers)
gap_durs:         0.6  (this was not varied in this experiment)
gap_after:        2  (gap after which letter?)
inter_trial_dur:  2.5
letter_mat:       array (384, 2, 6)  trial × talker × timing-slot.  Values are
                  integer indices into "letters"; -1 indicates silence (during
                  first 2 slots, cue is only spoken by one talker)
letters:          "OAUBDEGPTV"
n_blocks:         8
n_cue_let:        2
n_targ_let:       4
stim_dir:         "stimuli"
stim_names:       list (384,).  filenames for each trial.
stim_times:       array[0.0, 0.4, 1.2, 1.6, 2.6, 3.]
targ_pos:         array (384, 2, 4). Boolean of target locations (dtype float)
trial_durs:       array (384,).  all values are 5.9
'''
attns = params['attns']
spatials = params['spatials']
idents = params['idents']
stim_times = params['stim_times']
gap_durs = params['gap_durs']
n_blocks = params['n_blocks']
cond_mat = params['cond_mat']
n_trials = cond_mat.shape[0]

# construct event dict (mapping between trial parameters and integer IDs)
event_dict, rev_dict = dict(), dict()
for ai in range(len(attns)):
    for si in range(len(spatials)):
        for ii in range(len(idents)):
            if len(set(spatials[si].split('x'))) == 1 and \
                    len(set(idents[ii].split('-'))) == 1:
                continue  # shouldn't have run this condition
            for gi in range(len(gap_durs)):
                key = '%s_%s_%s_%s' % (attns[ai], spatials[si], idents[ii],
                                       gap_durs[gi])
                val = ttl_to_id([ai, si, ii, gi])
                event_dict[key] = val
                rev_dict[val] = key
# sanitize numpy scalars into ints before yaml dump
event_dict = {key: int(val) for key, val in event_dict.items()}
for d, fn in [(event_dict, 'event_dict.yaml'),
              (rev_dict, 'event_dict_rev.yaml')]:
    with open(fn, 'w') as f:
        yaml.dump(rev_dict, f, default_flow_style=False)

# init some containers
zscores = list()
fits = list()
gaze_angles = list()  # relative to fixation cross

# pre-calculate kernel
kernel = pupil_kernel(fs_out, t_max=t_peak, dur=2.0)

for subj in subjects:
    t0 = time.time()
    raws = list()
    events = list()
    stim_nums = list()
    print('Subject {}...'.format(subj))
    # find files for this subj
    fnames = get_pupil_data_files(subj, indir)
    n_files_expected = list(np.arange(1, 3) + len(params['block_trials']))
    assert len(fnames) in n_files_expected
    ix = 0 - len(params['block_trials'])
    fnames = fnames[ix:]  # first blocks are training & pupil response function

    # subject's expyfun log
    subj_tab = glob(op.join(indir, '{}_*.tab'.format(subj)))
    assert len(subj_tab) == 1
    subj_tab = subj_tab[0]
    with open(subj_tab, 'r') as fid:
        session = int(eval(fid.readline().strip()[2:])['session']) - 1
    subj_tab = read_tab(subj_tab)
    subj_tab = subj_tab[-n_trials:]
    stim_onset_times = [s['play'][0][1] for s in subj_tab]

    print('  Loading block', end=' ')
    for run_ix, fname in enumerate(fnames):
        print(str(run_ix + 1), end=' ')
        raw = read_raw(fname)
        assert raw.info['sfreq'] == fs_in
        raw.remove_blink_artifacts()
        raws.append(raw)
        # get the stimulus numbers presented in this block
        this_stim_nums = \
            params['block_trials'][params['blocks'][session][run_ix]]
        stim_nums.extend(this_stim_nums)
        this_cond_mat = cond_mat[this_stim_nums]
        # extract event codes from eyelink data
        event = raw.find_events('SYNCTIME', 1)
        ttls = [np.array([int(mm) for mm in m[1].decode().split(' ')[1:]])
                for m in raw.discrete['messages']
                if m[1].decode().startswith('TRIALID')]
        assert len(ttls) == len(event) == len(this_cond_mat)

        # convert event IDs. the 4-digit integer array in this_cond_mat gives
        # codes for [attn, spatial, idents, gap]. These are coded in binary in
        # the TTLs, with 2 digits for "spatial" and "idents" and 1 digit for
        # "attn" and "gap".  After checking that the expected trial code from
        # this_cond_mat matches the trial code recorded as TTL, we convert the
        # separate condition codes to a single integer code used in event_dict
        # and rev_dict (has to be a single integer to serve as a dict key); see
        # ttl_to_id function.  Those integer codes are then put into the event
        # list in place of the 1-triggers used to mark stimulus onsets.
        n_digits = [1, 2, 2, 1]
        for c, t in zip(this_cond_mat, ttls):
            assert np.array_equal(decimals_to_binary(c, n_digits), t)
        # this maps the sequence of 4 (single-digit) ints into a single int.
        # the sequence of experiment variables corresponding to each order of
        # magnitude is taken from params['cond_readme']; in this case it is
        # 1000s: attn, 100s: spatial, 10s: idents, 1s: gap
        event[:, 1] = [ttl_to_id(c) for c in this_cond_mat]
        for e in event[:, 1]:
            assert e in rev_dict
        events.append(event)

    print('\n  Epoching...')
    epochs = Epochs(raws, events, event_dict, t_min, t_max)
    if downsample:
        print('  Downsampling...')
        epochs.resample(fs_out, n_jobs=n_jobs)
    # compute gaze angles (in degrees)
    angles = get_gaze_angle(epochs, screenprops)
    # zscore pupil sizes
    zscore = epochs.pupil_zscores()
    # init some containers
    kernel_fits = list()
    kernel_zscores = list()

    print('  Deconvolving...')
    deconv_kwargs = dict(kernel=kernel, n_jobs=n_jobs, acc=1e-3)
    if deconv_time_pts is not None:
        deconv_kwargs.update(dict(spacing=deconv_time_pts))
    fit, time_pts = epochs.deconvolve(**deconv_kwargs)
    if deconv_time_pts is None:
        deconv_time_pts = time_pts
    assert np.array_equal(deconv_time_pts, time_pts)
    # order sequentially by stimulus ID
    order = np.argsort(stim_nums)
    angles = angles[order]
    zscore = zscore[order]
    fit = fit[order]
    # add this subject's data to global container
    fits.append(fit)
    zscores.append(zscore)
    gaze_angles.append(angles)
    print('  Done: {} sec.'.format(str(round(time.time() - t0, 1))))

# convert to arrays
fits_array = np.array(fits)
zscores_array = np.array(zscores)
gaze_angles = np.array(gaze_angles)

# params to output for all kernels
out_dict = dict(fs=fs_out, subjects=subjects, t_fit=deconv_time_pts,
                kernel=kernel, fits=fits_array, zscores=zscores_array,
                angles=gaze_angles, times=epochs.times)

np.savez_compressed(op.join(outdir, 'capd-pupil-data.npz'), **out_dict)
