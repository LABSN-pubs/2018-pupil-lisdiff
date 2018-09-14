#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'pupil_helper_functions'
===============================================================================

These are helper functions for the analysis of pupillometry data, specifically
for the vocoder/switch-gap and reverb/gender experiments.
"""
# Created on Wed Mar  1 17:36:26 2017
# @author: Dan McCloy  (drmccloy@uw.edu)
# License: BSD (3-clause)

from __future__ import print_function
from glob import glob
from os import path as op
import numpy as np


def parse_run_indices(run_inds, n_blocks):
    # run_inds is a list of length 10 describing which stimuli were played in
    # each block, and in what order, and whether the stim was 10-channel or
    # 20-channel vocoded.
    assert len(run_inds) == n_blocks
    # each list element is an object array of length 1, whose sole element is
    # a 2 x 32 object array of integers. Here we convert run_inds to a list of
    # 32 x 2 arrays of ints...
    run_inds = [np.array([ri[0][0], ri[0][1]], int).T for ri in run_inds]
    run_inds = np.array(run_inds)
    # ...pull out the stim indices (converting to 0-indexing in the process)
    stim_indices = run_inds[:, :, 0] - 1
    # ...and the code for number of vocoder channels (1=10, 2=20, but we
    # don't actually convert the 1 and 2 codes here, because they're used
    # to generate the pseudo-binary event ID)
    bands = run_inds[:, :, 1]
    return stim_indices, bands


def get_onset_times(subj, data_dir):
    from scipy.io import loadmat
    subj_mat_file = glob(op.join(data_dir, 'subj{}_*.mat'.format(subj)))
    assert len(subj_mat_file) == 1
    subj_mat = loadmat(subj_mat_file[0])
    time_vecs = subj_mat['timeVecs'][-10:]  # a 10 x 1 object array.
    # Within each cell of time_vecs is a 32 trials x 4 array; the second
    # column of that array (index 1) is "sound onset".
    stim_onset_times = [t[0][:, 1] for t in time_vecs]
    return stim_onset_times


def get_pupil_data_file_list(subj, data_dir):
    subj_data_dir = glob(op.join(data_dir, 'subj{}_*_el'.format(subj)))
    assert len(subj_data_dir) == 1
    subj_data_dir = subj_data_dir[0]
    fnames = sorted(glob(op.join(subj_data_dir, 'subj{}_*.edf'.format(subj))))
    assert len(fnames) in [13, 14, 15]
    fnames = fnames[-10:]  # 10 actual test blocks; others are training
    return fnames


def extract_event_codes(raw, this_stim_nums, stim_onset_times, run_ix):
    from expyfun.analyze import restore_values
    this_stim_count = len(this_stim_nums)
    # TRIALID 3 -> a real trial (0, 1, and 2 are types of training trials)
    # event_id=1  -> the start-stimulus trigger
    ev = raw.find_events('TRIALID 3', event_id=1)
    n_missing = this_stim_count - len(ev)
    stim_order_according_to_eyelink = [int(m[1].split(',')[3]) - 1
                                       for m in raw.discrete['messages']
                                       if 'TRIALID 3' in m[1]]
    if n_missing:
        missing_ixs = list()
        eyelink_stim_ix = 0
        # find which indices from design matrix are missing from eyelink data
        for ix, trial_num in enumerate(this_stim_nums):
            if stim_order_according_to_eyelink[eyelink_stim_ix] == trial_num:
                eyelink_stim_ix += 1
            else:
                missing_ixs.append(ix)
        assert len(missing_ixs) == n_missing
        not_missing = np.setdiff1d(np.arange(this_stim_count), missing_ixs)
        restored_sample_nums = restore_values(correct=stim_onset_times[run_ix],
                                              other=ev[:, 0], idx=missing_ixs)
        # re-make event array with missing values restored
        ev = np.array((restored_sample_nums[0],
                       np.ones(restored_sample_nums[0].size)), int).T
        stim_order = np.empty_like(this_stim_nums)
        stim_order[missing_ixs] = this_stim_nums[missing_ixs]
        stim_order[not_missing] = stim_order_according_to_eyelink
        stim_order_according_to_eyelink = stim_order
        print('Recovered {} trial(s)'.format(n_missing), end='\n    ')
    assert np.array_equal(stim_order_according_to_eyelink, this_stim_nums)
    assert len(ev) == this_stim_count
    return ev


def reorder_epoched_data(data, stim_mat, bands, stim_indices, n_times):
    # reorder zscored and deconvolved pupil signals to match first dimension
    # of stim_mat (stim# in serial order). second dim is number of vocoder
    # bands; third dim is time samples
    band_idx = (bands - 1).ravel()
    stim_idx = (stim_indices).ravel()
    ordered = np.full((len(stim_mat), 2, n_times), np.inf)
    ordered[stim_idx, band_idx, :] = data
    assert np.all(np.isfinite(ordered))
    return ordered


def restructure_dims(data, stim_mat, bands, n_times):
        # new dims: (trial, gap, attn, bands, time)
        n_attn = np.unique(stim_mat[:, 0]).size
        n_gaps = np.unique(stim_mat[:, 1]).size
        n_band = np.unique(bands).size
        trials_per_cond = len(stim_mat) / n_attn / n_gaps
        reshaped_data = np.empty((trials_per_cond, n_gaps, n_attn, n_band,
                                  n_times))
        for ai in range(n_attn):
            for gi in range(n_gaps):
                ix = np.logical_and(stim_mat[:, 0] == ai + 1,
                                    stim_mat[:, 1] == gi + 1)
                assert sum(ix) == trials_per_cond
                reshaped_data[:, gi, ai, :, :] = data[ix, :, :]
        return reshaped_data


def do_continuous_deconv(data, kernel, times):
    from scipy.signal import deconvolve
    # zero padding
    kernel_nsamp = np.round(kernel.shape[-1]).astype(int)
    zeropad = np.zeros(data.shape[:-1] + (kernel_nsamp,))
    zeropadded = np.c_[zeropad, data, zeropad]
    print('  Continuous deconvolution...')
    len_deconv = zeropadded.shape[-1] - kernel_nsamp + 1
    times = times[:len_deconv - 2 * kernel_nsamp]  # no zeropad
    deconvolved = np.full(zeropadded.shape[:-1] + (len_deconv,), np.inf)
    # do deconvolution
    for _trial in range(zeropadded.shape[0]):
        for _gap in range(zeropadded.shape[1]):
            for _attn in range(zeropadded.shape[2]):
                for _band in range(zeropadded.shape[3]):
                    signal = zeropadded[_trial, _gap, _attn, _band, :]
                    (deconvolved[_trial, _gap, _attn, _band, :],
                     _) = deconvolve(signal, kernel)
    assert np.all(np.isfinite(deconvolved))
    # remove zero padding
    deconvolved = deconvolved[:, :, :, :, kernel_nsamp:-kernel_nsamp]
    return deconvolved, times


def get_gaze_angle(epochs, screenprops):
    xx = epochs.get_data('xpos')  # in pixels
    yy = epochs.get_data('ypos')  # in pixels
    x_px = xx - screenprops['width_px'] // 2   # rel. to h. pos. of fix. cross
    y_px = yy - screenprops['height_px'] // 2  # rel. to v. pos. of fix. cross
    x_cm = x_px * screenprops['width_cm'] / screenprops['width_px']
    y_cm = y_px * screenprops['height_cm'] / screenprops['height_px']
    dist_from_center = np.sqrt(x_cm ** 2 + y_cm ** 2)
    deviation_rad = np.arctan2(dist_from_center, screenprops['dist_cm'])
    deviation_deg = deviation_rad * 180 / np.pi
    return deviation_deg
