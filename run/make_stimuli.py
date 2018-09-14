# -*- coding: utf-8 -*-

import numpy as np
import os
from os import path as op
from shutil import rmtree
import warnings

from expyfun.stimuli import resample, rms, convolve_hrtf
from expyfun.io import read_wav, write_wav, write_hdf5

# We only actually dope the 3rd pos @ 0.25 instead of 0.5 (don't need RTs)
# as badly, don't want to confuse subjects

warnings.simplefilter('error')

fix_rgb = [0.8] * 3

work_dir = op.dirname(__file__)

letters = 'OAUBDEGPTV'
letter_dur = 0.4
cue_targ_gap = letter_dur
fs = 24414
letter_ns = int(letter_dur * fs)
talkers = ('male', 'female')
gap_durs = (0.6,)  # gap duration (sec)
spatials = ('-30x30', '30x-30', '-30x-30', '30x30')
idents = ('male-male', 'female-female', 'male-female', 'female-male')
run_matrix = np.array([[True, True, False, False],  # MM, in each spatial
                       [True, True, False, False],  # FF, in each spatial
                       [True, True, True, True],    # MF, in each spatial
                       [True, True, True, True]], bool)
attns = ('maintain', 'switch')
n_tpc = 16  # desired trials per condition
dope_after_fake = 0.5  # prop of trials that should have targ or mask in slot 3
dope_after_actual = 0.25  # we don't actually do any doping
n_blocks = 8

n_cue_let = 2
n_targ_let = 4
gap_after = 2
inter_trial_dur = 2.5
assert n_tpc % n_blocks == 0
assert len(talkers) == 2  # several things assume this
n_dpc = int(n_tpc * dope_after_fake)
n_npc = int(n_tpc * (1. - dope_after_fake))
assert n_dpc + n_npc == n_tpc


##############################################################################
# Load letter WAV files

all_spatials = [s.split('x') for s in spatials]
for s in all_spatials[1:]:
    all_spatials[0] += s
all_spatials = all_spatials[0]
all_spatials = list(np.unique([float(s) for s in all_spatials]))

letter_dir = op.join(work_dir, 'letters')
wavs = np.zeros((len(talkers), len(letters), len(all_spatials), 2, letter_ns))
for li, letter in enumerate(letters):
    for ti, talker in enumerate(talkers):
        data, fs_in = read_wav(op.join(letter_dir, talker, '%s.wav' % letter),
                               verbose=False)
        data = resample(data[0], fs, fs_in)
        for si, angle in enumerate(all_spatials):
            dd = convolve_hrtf(data, fs, angle)
            dd *= 0.01 / np.mean(rms(data))
            idx = min(dd.shape[1], letter_ns)
            wavs[ti, li, si, :, :idx] = dd[:, :idx]


##############################################################################
# Randomization

n_trials = n_tpc * len(attns) * run_matrix.sum() * len(gap_durs)
trial_dur = (letter_dur * (n_cue_let + n_targ_let) + cue_targ_gap +
             np.mean(gap_durs) + inter_trial_dur)
exp_dur = trial_dur * n_trials
print('Experiment duration: %s min (%s blocks)'
      % (round(exp_dur / 60., 1), round((exp_dur / 60. / n_blocks), 1)))

# figure out what positions work
assert n_targ_let == 4
pos = ['0000', '0001', '0010', '0100', '0101', '0110', '1000',
       '1001', '1010']
min_iti = 0.8

double_poss = [list() for _ in range(len(gap_durs))]
ts = list()
for gap in gap_durs:
    idx = np.arange(n_targ_let)
    ts.append(idx * letter_dur + (idx >= gap_after) * gap)
count = [0] * len(gap_durs)
for p1 in pos:
    x1 = np.array([int(p) for p in p1])
    idx_1 = np.where(x1 == 1)[0]
    c1a = p1[:gap_after].count('1')
    c1b = p1[gap_after:].count('1')
    c1 = c1a + c1b
    for p2 in pos:
        # total counts and counts during first and second part must differ
        c2a = p2[:gap_after].count('1')
        c2b = p2[gap_after:].count('1')
        c2 = c2a + c2b
        if (c1a != c2a and c1b != c2b):
            for gi, gap in enumerate(gap_durs):
                x2 = np.array([int(p) for p in p2])
                idx_2 = np.where(x2 == 1)[0]
                good = all(np.diff(ts[gi][idx_1]) >= min_iti)
                good = good and all(np.diff(ts[gi][idx_2]) >= min_iti)
                if good and len(idx_2) > 0:
                    good = all(np.abs(ts[gi][idx] - ts[gi][idx_2]).min()
                               >= min_iti for idx in idx_1)
                if good:
                    double_poss[gi].append([x1, x2])
                    count[gi] += 1
double_poss = [np.array(d, dtype=bool) for d in double_poss]
for d in double_poss:
    assert (d.sum(-1) <= 3).all()
    assert (d.sum(-1) >= 0).all()
    assert not (d.sum(1) > 1).any()
rng = np.random.RandomState(0)

# target/masker positions

# trial type: n_bands, n_attn
cond_readme = 'attn, spatial, idents, gap'
n_spec = 4
cond_mat = np.zeros((n_trials, n_spec), int)
ti = 0

# auditory
targ_pos = np.empty((n_trials, 2, n_targ_let))
targ_pos.fill(np.nan)
for ii in range(len(idents)):
    for si in np.where(run_matrix[ii])[0]:
        for ai in range(len(attns)):
            for gi in range(len(gap_durs)):
                # dope third position
                didx = np.any(double_poss[gi][:, :, gap_after], axis=1)
                nidx = ~didx
                didx = np.where(didx)[0]
                nidx = np.where(nidx)[0]

                # doped trials
                ldp = len(didx)
                aud_idx = np.arange(int(np.ceil(n_dpc / float(ldp))) * ldp)
                aud_idx = didx[rng.permutation(aud_idx % ldp)]
                for ci in range(n_dpc):
                    cond_mat[ti] = [ai, si, ii, gi]
                    targ_pos[ti] = double_poss[gi][aud_idx[ci]]
                    ti += 1

                # non-doped trials
                ldp = len(nidx)
                aud_idx = np.arange(int(np.ceil(n_npc / float(ldp))) * ldp)
                aud_idx = nidx[rng.permutation(aud_idx % ldp)]
                for ci in range(n_npc):
                    cond_mat[ti] = [ai, si, ii, gi]
                    targ_pos[ti] = double_poss[gi][aud_idx[ci]]
                    ti += 1
                assert ti % n_blocks == 0
assert not (targ_pos == np.nan).any()
assert ti == n_trials
assert np.mean(targ_pos[:, 0, gap_after]) == dope_after_actual

counts = []
for ii, row in enumerate(run_matrix):
    for si, b in enumerate(row):
        n_type = np.sum(np.logical_and(cond_mat[:, 1] == si,
                                       cond_mat[:, 2] == ii))
        if b:
            counts.append(n_type)
        else:
            assert n_type == 0
assert len(counts) == run_matrix.sum()
assert len(np.unique(counts)) == 1

trial_durs = (letter_dur * (n_cue_let + n_targ_let) + cue_targ_gap +
              inter_trial_dur + np.array(gap_durs)[cond_mat[:, 3]])
assert np.allclose(trial_durs.sum(), exp_dur)

##############################################################################
# Blocking

block_types = [
    ['LM', 'LF'],
    ['RM', 'RF'],
    ['LM', 'RM'],
    ['LF', 'RF'],
]

block_ords = [
    [0, 1, 2, 3],
    [1, 0, 2, 3],
    [0, 1, 3, 2],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 2, 1, 0],
    [3, 2, 0, 1]
]

blocks = [[b for bii in bi for b in block_types[bii]] for bi in block_ords]
assert all(set(b[:4]) == set(b[4:]) for b in blocks)
blocks = [[bb + ('0' if bi < 4 else '1') for bi, bb in enumerate(b)]
          for b in blocks]
assert all(set(blocks[0]) == set(b) for b in blocks)
assert all(len(b) == n_blocks for b in blocks)

block_trials = dict()
for bt in set(blocks[0]):
    block_trials[bt] = []
sorted_keys = sorted(block_trials.keys())

used = np.zeros(n_trials, bool)
for ai in range(len(attns)):
    for gi in range(len(gap_durs)):
        for bt in sorted_keys:
            mask = (cond_mat[:, 0] == ai) & (cond_mat[:, 3] == gi) & ~used
            assert bt in block_trials
            assert bt[0] in 'LR'
            assert bt[1] in 'MF'
            # add L/R
            mask_l = np.in1d(cond_mat[:, 1],
                             np.where([s[0] == '-' for s in spatials])[0])
            mask &= mask_l if bt[0] == 'L' else ~mask_l
            # add M/F
            mask_m = np.in1d(cond_mat[:, 2],
                             np.where([s.startswith('male') for s in idents]))
            mask &= mask_m if bt[1] == 'M' else ~mask_m
            # Pick the indices to add
            assert mask.sum() == n_tpc // 2 * (6 if bt[2] == '0' else 3)
            idx = rng.permutation(np.where(mask)[0])[:n_tpc // 2 * 3]
            block_trials[bt] += list(idx)
            used[idx] = True


assert np.array_equal(np.unique([len(b) for b in block_trials.values()]),
                      [n_trials // n_blocks])
for key in block_trials:
    block_trials[key] = rng.permutation(block_trials[key])
assert np.array_equal(np.sort(np.concatenate(block_trials.values())),
                      np.arange(n_trials))

##############################################################################
# Create stimuli

stim_dir_end = 'stimuli'
stim_dir = op.join(work_dir, stim_dir_end)
if op.isdir(stim_dir):
    rmtree(stim_dir)
os.mkdir(stim_dir)

mask_idx = np.arange(len(letters) - 4) + 3  # OAU, target + cues

# letter timing
cue_t = np.arange(n_cue_let) * letter_dur
for gi in range(len(gap_durs)):
    ts[gi] = np.concatenate((cue_t, letter_dur * n_cue_let
                             + cue_targ_gap + ts[gi]))
max_t = max([t[-1] for t in ts]) + letter_dur
stim = np.empty((2, int(max_t * fs) + 1))
ns = [(np.array(t) * fs).astype(int) for t in ts]
stim_names = []
letter_mat = np.empty((n_trials, 2, n_targ_let + 2), int)  # per talker
for ti in range(n_trials):
    stim.fill(0)
    ai, si, ii, gi = cond_mat[ti, :]

    # spatial and talker indices
    sidx = np.array([all_spatials.index(float(s))
                     for s in spatials[si].split('x')])
    tidx = np.array([talkers.index(s) for s in idents[ii].split('-')])

    # letters
    good = False
    while(not good):
        lidxs = rng.choice(mask_idx, (2, n_targ_let))
        good = (np.diff(lidxs, axis=0).all() and   # no same across talker
                np.diff(lidxs, axis=1).all())  # no repeats
    lidxs = [[0 if targ_pos[ti, jj, li] else l for li, l in enumerate(ll)]
             for jj, ll in enumerate(lidxs)]
    lidxs = np.concatenate(([[1, cond_mat[ti, 0] + 1], [-1, -1]], lidxs),
                           axis=1)

    if ai == 1:  # switch second halves
        lidxs[:, n_cue_let + gap_after:] = lidxs[::-1, n_cue_let + gap_after:]
    letter_mat[ti] = lidxs

    # construct
    for jj in range(2):  # talkers
        for li in range(n_cue_let + n_targ_let):
            if lidxs[jj, li] >= 0:
                x = wavs[tidx[jj], lidxs[jj, li], sidx[jj]]
                stim[:, ns[gi][li]:ns[gi][li] + letter_ns] += x

    if ai == 1:  # switch second halves back
        lidxs[:, n_cue_let + gap_after:] = lidxs[::-1, n_cue_let + gap_after:]

    t, m = np.array([l for l in letters])[lidxs[:, 2:]]
    fname = ('stim_%s_%s_%s_%s_%s_t-%s_m-%s.wav'
             % (ti + 1, attns[ai], spatials[si], idents[ii],
                int(1e3 * gap_durs[gi]), ''.join(t), ''.join(m)))
    write_wav(op.join(stim_dir, fname), stim, fs, verbose=False)
    print('  Writing %s (%s/%s)' % (fname, ti + 1, n_trials))
    stim_names.append(fname)

##############################################################################
# Write out result

params = dict(gap_durs=gap_durs, spatials=spatials, idents=idents, attns=attns,
              inter_trial_dur=inter_trial_dur, targ_pos=targ_pos,
              cond_mat=cond_mat, cond_readme=cond_readme,
              blocks=blocks, block_trials=block_trials,
              stim_names=stim_names, stim_dir=stim_dir_end,
              n_targ_let=n_targ_let, n_cue_let=n_cue_let,
              trial_durs=trial_durs, stim_times=ts, n_blocks=n_blocks,
              letter_mat=letter_mat, letters=letters, gap_after=gap_after)
write_hdf5(op.join(work_dir, 'params.hdf5'), params, overwrite=True)
