# -*- coding: utf-8 -*-
"""
===============================================================================
Script ''
===============================================================================

This script cleans and analyzes behavioral data for the CAPD attention
switching pupillometry experiment.
"""
# @author: Dan McCloy  (drmccloy@uw.edu)
# Created on Tue Jun 26 12:42:15 PDT 2018
# License: BSD (3-clause)

import yaml
import os.path as op
import numpy as np
import pandas as pd
from glob import glob
from expyfun.io import read_tab, read_hdf5
from expyfun import binary_to_decimals
from expyfun.analyze import press_times_to_hmfc, dprime

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', None)


def parse_trial_id(trial_id):
    int_id = eval(trial_id[0][0])
    attn = binary_to_decimals(int_id[:1], n_bits=1)[0]
    spatial = binary_to_decimals(int_id[1:3], n_bits=2)[0]
    idents = binary_to_decimals(int_id[3:5], n_bits=2)[0]
    gap = binary_to_decimals(int_id[5:], n_bits=1)[0]
    return [attn, spatial, idents, gap]


def parse_presses(presses):
    return [p[1] for p in presses]


def spread_slots(series, name):
    n_slots = len(series.iloc[0])
    names = [name] * n_slots
    slots = list(range(1, n_slots + 1))
    new_index = pd.MultiIndex.from_arrays((names, slots),
                                          names=(None, 'slot'))
    spread_df = pd.DataFrame(np.stack(series.values, axis=0),
                             columns=new_index)
    spread_df.index = series.index
    return spread_df.stack().reset_index()


def is_first_press(df, trial, resps):
    slot = np.where(resps)[0][0] + 1
    locator = np.logical_and((df['trial'] == trial), (df['slot'] == slot))
    assert locator.sum() == 1
    locator = np.where(locator)[0][0]
    if np.isnan(df.at[locator, 'rt']):
        return True
    return False


# file I/O
datadir = 'data'
paramdir = 'params'
param_file = op.join(paramdir, 'params.hdf5')
yaml_param_file = op.join(paramdir, 'params.yaml')

# yaml params
with open(yaml_param_file, 'r') as yp:
    yaml_params = yaml.load(yp)
    subjects = yaml_params['subjects']
    listening_difficulty_bool = np.array(yaml_params['lis_diff'], dtype=bool)
    spatial_mapping = yaml_params['spatial_mapping']
    talker_mapping = yaml_params['talker_mapping']
    rt_min = yaml_params['rt_min']
    rt_max = yaml_params['rt_max']
assert len(subjects) == len(listening_difficulty_bool)
subjs_with_lisdiff = np.array(subjects)[listening_difficulty_bool].tolist()

# hdf5 params
params = read_hdf5(param_file)
cond_mat = params['cond_mat']
''' RELEVANT KEYS:
cond_mat:         array (384, 4). columns are [attn, spatial, idents, gap]
attns:            maintain vs switch
spatials:         -30x30, 30x-30, -30x-30, 30x30
idents:           MM, FF, MF, FM (male/female talkers)
stim_times:       array[0.0, 0.4, 1.2, 1.6, 2.6, 3.]
letters:          'OAUBDEGPTV'
letter_mat:       array (384, 2, 6)
targ_pos:         array (384, 2, 4)
block_trials:     keys: LF0, LF1, LM0, LM1, RF0, RF1, RM0, RM1
                  gives order of trial #s within each named block
blocks:           8×8 list giving block order based on session number
'''

n_trials = len(cond_mat)
stim_times = params['stim_times'][0][2:]  # first 2 are cue
letters = np.array(list(params['letters']) + [''])
letter_mat = letters[params['letter_mat']]
maint_bool = np.all(letter_mat[:, 0, :2] == 'A', axis=-1)  # AA cue = maint
switch_bool = np.logical_not(maint_bool)
deviant_bool = params['targ_pos'].astype(bool)
''' !!!!!!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!!!!!!!
    !  targ_pos[trial_num, 0, :] is target stream       !
    !  incorporating switch behavior (on switch trials) !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! '''
# make slot-based representation of targ/foil/neither
# this will work because there is never both target & foil in same slot
# start by marking all slots as '-' (neither targ nor foil)
slot_codes = np.full((deviant_bool.shape[0], deviant_bool.shape[-1]), '-')
targ_ix = np.where(deviant_bool[:, 0])
foil_ix = np.where(deviant_bool[:, 1])
slot_codes[targ_ix] = 't'
slot_codes[foil_ix] = 'f'
assert np.array_equal(np.sum(deviant_bool, axis=(1, 2)),
                      np.sum(slot_codes != '-', axis=1))

# init DF
trial_df = pd.DataFrame()
slots_df = pd.DataFrame()

# loop over subjects
for subj in subjects:
    logfile = glob(op.join(datadir, f'{subj}_*.tab'))
    assert len(logfile) == 1
    with open(logfile[0], 'r') as f:
        # first line includes dict with experiment metadata
        metadata = eval(f.readline().strip()[2:])
        assert metadata['participant'] == subj
        session = int(metadata['session']) - 1
    all_trials = read_tab(logfile[0])
    trials = all_trials[-n_trials:]  # omit training
    this_trial_df = pd.DataFrame(trials,
                                 columns=('trial_id', 'play', 'keypress'))
    this_trial_df.rename(columns=dict(play='trial_onset', keypress='presses'),
                         inplace=True)
    this_trial_df['trial_id'] = this_trial_df['trial_id'].map(parse_trial_id)
    this_trial_df['trial_onset'] = this_trial_df['trial_onset'].map(lambda x:
                                                                    x[0][1])
    this_trial_df['presses'] = this_trial_df['presses'].map(parse_presses)
    # merge in info from cond_mat (put in correct order first!)
    block_order = params['blocks'][session]
    trial_order = np.concatenate([params['block_trials'][block]
                                  for block in block_order])
    this_cond_mat = cond_mat[trial_order]
    assert np.array_equal(np.array(this_trial_df['trial_id'].values.tolist()),
                          this_cond_mat)
    this_trial_df['attn'] = np.array(params['attns'])[this_cond_mat[:, 0]]
    this_trial_df['angles'] = np.array(params['spatials'])[this_cond_mat[:, 1]]
    this_trial_df['genders'] = np.array(params['idents'])[this_cond_mat[:, 2]]
    this_trial_df['lr'] = this_trial_df['angles'].map(spatial_mapping)
    this_trial_df['mf'] = this_trial_df['genders'].map(talker_mapping)
    # make data contrast vectors
    non_spatial = np.in1d(this_trial_df['lr'], ['LL', 'RR'])
    spatial = np.in1d(this_trial_df['mf'], ['MM', 'FF'])
    both = np.logical_and(np.logical_not(spatial), np.logical_not(non_spatial))
    this_trial_df['s_cond'] = ''
    this_trial_df.loc[non_spatial, 's_cond'] = 'non-spatial'
    this_trial_df.loc[spatial, 's_cond'] = 'spatial'
    this_trial_df.loc[both, 's_cond'] = 'mixed'
    assert np.all(this_trial_df['s_cond'] != '')
    # add sequential trial number and subject number
    this_trial_df['seq_trial'] = np.arange(this_trial_df.shape[0])
    this_trial_df['trial'] = trial_order
    this_trial_df['subj'] = subj
    this_trial_df['lisdiff'] = subj in subjs_with_lisdiff
    group_ix = int(subj in subjs_with_lisdiff)
    this_trial_df['group'] = ('control', 'listening difficulty')[group_ix]
    # make sure letter_mat cues match this_trial_df attn codes
    assert np.all((this_trial_df['attn'] == 'maintain') ==
                  maint_bool[trial_order])
    # add timing slots
    this_trial_df['slot_codes'] = slot_codes[trial_order].tolist()
    this_trial_df['slot_times'] = \
        (this_trial_df['trial_onset'].values[:, None] + stim_times).tolist()
    # add HMFC at trial level
    for letter in 'hmfc':
        this_trial_df[letter] = -1
    for row in this_trial_df.itertuples(index=False):
        targets = np.array(row.slot_times)[np.array(row.slot_codes) == 't']
        foils = np.array(row.slot_times)[np.array(row.slot_codes) == 'f']
        hmfco = np.array(press_times_to_hmfc(row.presses, targets, foils,
                                             tmin=rt_min, tmax=rt_max))
        hmfc = hmfco[:4].copy()
        hmfc[2] += hmfco[4]  # "other" presses treated as false alm.
        row_locator = (this_trial_df['seq_trial'] == row.seq_trial)
        this_trial_df.loc[row_locator, list('hmfc')] = hmfc
    assert np.all(this_trial_df[list('hmfc')].values >= 0)
    # make longform (1 row per slot)
    code_series = this_trial_df[['trial', 'slot_codes']].set_index('trial')
    time_series = this_trial_df[['trial', 'slot_times']].set_index('trial')
    slot_codes_df = spread_slots(code_series['slot_codes'], 'slot_code')
    slot_times_df = spread_slots(time_series['slot_times'], 'slot_onset')
    slots = pd.merge(slot_codes_df, slot_times_df, on=['trial', 'slot'])
    this_slots_df = this_trial_df.merge(slots, on='trial')
    # compute RTs
    this_slots_df['rt'] = np.nan
    this_slots_df['extra_rts'] = [[] for _ in range(len(this_slots_df))]
    cols = ['trial', 'presses']
    for trial, presses in this_trial_df[cols].itertuples(index=False):
        for press in presses:
            trial_locator = (this_slots_df['trial'] == trial)
            rts = press - this_slots_df.loc[trial_locator, 'slot_onset'].values
            codes = this_slots_df.loc[trial_locator, 'slot_code'].values
            inrange = np.logical_and(rt_min <= rts, rts <= rt_max)
            hits = np.logical_and(inrange, codes == 't')
            foil_hits = np.logical_and(inrange, codes == 'f')
            false_alarms = np.logical_and(inrange, codes == '-')
            '''
            late_hits = np.logical_and(rt_min <= rts, codes == 't')
            late_foil_hits = np.logical_and(rt_min <= rts, codes == 'f')
            '''
            # first assume press was response to targ
            if np.any(hits) and is_first_press(this_slots_df, trial, hits):
                ix = np.where(hits)[0][0]
            # if not, assume response to distractor
            elif np.any(foil_hits) and is_first_press(this_slots_df, trial,
                                                      foil_hits):
                ix = np.where(foil_hits)[0][0]
            # if not, in-range press was a false alarm
            elif np.any(false_alarms):
                ix = np.where(false_alarms)[0][0]
            # if maybe was duplicate resp to targ but otherwise unattributable
            elif np.any(hits):
                ix = np.where(hits)[0][0]
            # if maybe was duplicate resp to foil but otherwise unattributable
            elif np.any(foil_hits):
                ix = np.where(foil_hits)[0][0]
                '''
            elif np.any(late_hits):  # if not, assume resp. to targ, but late
                col = 'long_rt'
                ix = np.where(late_hits)[0][-1]
            elif np.any(late_foil_hits):  # if not, assume late resp. to foil
                col = 'long_rt'
                ix = np.where(late_foil_hits)[0][-1]
                '''
            elif np.any(rts > 0):  # if not, assign wherever yields shortest RT
                ix = np.where(rts > 0)[0][-1]
            else:
                print(f'unattributed press: subj {subj}, trial {trial}, '
                      f'RTs: {np.round(rts, 3)}')
                continue
            rt = rts[ix]
            slot_locator = (this_slots_df['slot'] == (ix + 1))
            locator = np.logical_and(trial_locator, slot_locator)
            assert locator.sum() == 1
            locator = np.where(locator)[0][0]
            if np.isnan(this_slots_df.at[locator, 'rt']):
                this_slots_df.at[locator, 'rt'] = rt
            else:
                this_slots_df.at[locator, 'extra_rts'].append(rt)
                # print(f'duplicate press: subj {subj}, '
                #       f'trial {trial}, RTs: {np.round(rts, 3)}')

    # check press attribution
    total_rts = (this_slots_df['rt'].map(np.isfinite) +
                 this_slots_df['extra_rts'].map(len)).sum()
    total_presses = this_trial_df['presses'].map(len).sum()
    if total_rts != total_presses:
        print(f'press/RT mismatch: {total_presses} presses; {total_rts} RTs')

    # add to master DFs
    trial_df = pd.concat((trial_df, this_trial_df), axis=0, ignore_index=True)
    slots_df = pd.concat((slots_df, this_slots_df), axis=0, ignore_index=True)

# sanity check
assert slots_df.shape[0] == 4 * trial_df.shape[0]

# reorder/drop columns
common_cols = ['subj', 'lisdiff', 'group', 'trial', 'trial_id', 'lr', 'mf',
               'attn', 's_cond', 'presses']
trial_df = trial_df[common_cols + ['trial_onset', 'slot_codes'] + list('hmfc')]
slots_df = slots_df[common_cols + ['slot', 'slot_code', 'slot_onset',
                                   'rt', 'extra_rts']].copy()  # 'long_rt'

# compute HMFC by slot
# h = hit target   |  f = total F.A.  |  ff = resp foil      fo = resp other
# m = miss target  |  c = total C.R.  |  cf = no resp foil   co = no resp other
# note that non-responses to non-targ-non-foil slots are ignored (i.e., they
# don't count as correct rejections)
pressed = np.isfinite(slots_df['rt'])
not_pressed = np.isnan(slots_df['rt'])
slots_df['h'] = np.logical_and(slots_df['slot_code'] == 't', pressed)
slots_df['m'] = np.logical_and(slots_df['slot_code'] == 't', not_pressed)
# 3 false alarm types: foil responses (ff), non-targ-non-foil responses (fo),
# and duplicate presses (fd)
slots_df['ff'] = np.logical_and(slots_df['slot_code'] == 'f', pressed)  # foil
slots_df['fo'] = np.logical_and(slots_df['slot_code'] == '-', pressed)  # err
slots_df['fd'] = slots_df['extra_rts'].map(len)  # duplicate resps.
slots_df['f'] = np.logical_or(slots_df['ff'], slots_df['fo']) + slots_df['fd']
# NB: to be consistent with the linear modeling approach, slots with neither
# target nor foil, in which no button press occurred, are treated as
# correct rejections (below, c = cf | co). This choice relates to the question
# of what counts as an "event" in a signal detection context; including "co"
# implies that each timing slot is an event (regardless of target/foil
# presence); the alternative (c = cf) implies that only occurrences of the
# target letter (whether in attended or masker stream) should count as events.
slots_df['cf'] = np.logical_and(slots_df['slot_code'] == 'f', not_pressed)
slots_df['co'] = np.logical_and(slots_df['slot_code'] == '-', not_pressed)
slots_df['c'] = np.logical_or(slots_df['cf'], slots_df['co'])
# slots_df['c'] = slots_df['cf']
slots_df[list('hmc')] = slots_df[list('hmc')].applymap(int)

# dprime sanity check
cols = ['subj'] + list('hmfc')
trial_based_hmfc = trial_df[cols].groupby('subj').aggregate(np.sum)
trial_based_dprime = trial_based_hmfc.apply(dprime, axis=1)
slot_based_hmfc = slots_df[cols].groupby('subj').aggregate(np.sum)
slot_based_dprime = slot_based_hmfc.apply(dprime, axis=1)
foo = pd.concat(dict(trial=trial_based_dprime, slot=slot_based_dprime), axis=1)
print(f"largest dprime diff: "
      f"{np.round(np.max(np.abs(foo['slot'] - foo['trial'])), 3)}")
# save
slots_df.to_csv('behavioral-data-by-slot.tsv', sep='\t', index=False)
trial_df.to_csv('behavioral-data-by-trial.tsv', sep='\t', index=False)
