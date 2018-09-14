# -*- coding: utf-8 -*-

import numpy as np
from os import path as op
from expyfun import (ExperimentController, EyelinkController, visual,
                     get_keyboard_input, assert_version, decimals_to_binary)
from expyfun.io import read_hdf5, read_wav
from expyfun.codeblocks import find_pupil_dynamic_range
from expyfun.analyze import press_times_to_hmfc

# TODO:
# - Button 1?
# - Center feedback more?

assert_version('7673bf0')
p = read_hdf5(op.join(op.dirname(__file__), 'params.hdf5'))
stim_dir = op.join(op.dirname(__file__), p['stim_dir'])

tmin, tmax = 0.1, 0.8  # acceptable delays for button presses


pre_instr = (
    'This set of trials is like the previous set of trials, except that')

dir_instr = (
    '\n\nOn this set of trials, the initial target speaker will always be a '
    '%s voice coming from the %s side.'
)

button_instr = (
    dir_instr +
    '\n\nYour job is to press a button as quickly as possible after the '
    'talker you\'re supposed to attend to says the letter "O". Please '
    'try to keep your eyes centered on the fixation dot at the center of '
    'the screen.')

full_instr = (
    ' the first two letters could be either "AU" or "AA".\n\n'
    'On "AA" trials, you should maintain attention to the talker that said '
    '"AA" for all four subsequent letters.\n\nOn "AU" trials, you should '
    'switch attention from the talker that said "AU" to the other talker '
    'after the first two letters following the "AU" cue.' + button_instr)

tr_instructions = [
    'In this first set of trials, there will be two people speaking letters '
    'simultaneously.\n\n'
    'You should listen *only* to the talker who starts out saying the letters '
    '"AA" at the beginning of the trial.' + button_instr,

    pre_instr + ' the first two letters spoken will "AU" instead of "AA".\n\n'
    'Instead of listening to the same talker for the four letters after the '
    'cue, you should start out by attending to the talker that said "AU" for '
    'the first two letters following "AU", then switch your attention to '
    'listen to the other talker for the last two letters.' + button_instr,

    pre_instr + full_instr,

    pre_instr + ' the talkers will now sound like they are located in the '
    'same location in space.\n\nLike last block,' + full_instr,

    pre_instr + ' the talkers will both be male speakers, located in '
    'different places.\n\nLike last block,' + full_instr,

    pre_instr + ' the talkers could be both male, or one male one female; '
    'and they could also be located in the same or different positions in '
    'space.\n\nLike last block,' + full_instr,
]

demo_instr = (
    'You will now be shown the flow of an example trial.\n\n'
    'The letters from the target you should listen to will turn white, and '
    'the target "O" letters (that should elicit a button press during a '
    'trial) will turn green.\n\n'
    'Press a button to see the demonstration.'
)
n_tr_trials = [5, 5, 10, 5, 5, 16]
tr_thresholds = [4, 4, 7, 0, 0, 8]

assert len(tr_instructions) == len(n_tr_trials) == len(tr_thresholds)


def _get_strs(type_):
    """Helper to convert block type to strings"""
    assert len(type_) == 3
    assert type_[0] in 'LR'
    assert type_[1] in 'MF'
    dstr = 'left' if type_[0] == 'L' else 'right'
    tstr = 'male' if type_[1] == 'M' else 'female'
    return tstr, dstr


with ExperimentController('capd', stim_db=65, noise_db=45,
                          check_rms=None) as ec:
    # Pupil parameters ######################################################
    el = EyelinkController(ec)
    bgcolor, fcolor = find_pupil_dynamic_range(ec, el)[:2]
    ec.write_data_line('bgcolor', bgcolor)
    ec.write_data_line('fcolor', fcolor)
    print('Using colors: %s, %s\n' % (bgcolor, fcolor))
    ec.set_background_color(bgcolor)
    fix = visual.FixationDot(ec, colors=[fcolor, 'k'])

    session = int(ec.session) if ec.session != '' else 0
    assert 0 <= session <= len(p['blocks'])
    n_blocks = p['n_blocks']

    # Trial running #########################################################

    def run_trial(ti, feedback=False):
        """Run a trial, optionally with feedback"""
        samples = read_wav(op.join(stim_dir, p['stim_names'][ti]))[0]
        ec.load_buffer(samples)
        id_ = decimals_to_binary(p['cond_mat'][ti], [1, 2, 2, 1])
        ec.identify_trial(ec_id=id_, ttl_id=id_, el_id=id_)
        ec.listen_presses()
        t0 = ec.start_stimulus(flip=False)
        wait_dur = p['trial_durs'][ti]
        wait_dur -= p['inter_trial_dur'] - 0.5 if feedback else 0
        presses = ec.wait_for_presses(wait_dur, relative_to=t0)
        ec.stop()
        ec.trial_ok()
        correct = True
        if feedback:
            press_times = [pp[1] for pp in presses]
            p['targ_pos'][ti][0]
            t = p['stim_times'][p['cond_mat'][ti][3]][2:]
            targ = t[p['targ_pos'][ti][0].astype(bool)]
            foil = t[p['targ_pos'][ti][1].astype(bool)]
            hmfc = press_times_to_hmfc(press_times, targ, foil, tmin, tmax)
            correct = (hmfc[1] == hmfc[2] == hmfc[4] == 0)
            if correct:
                msg = 'Correct!\n\n'
            else:
                msg = 'Incorrect:\n'
                if hmfc[1] > 0:
                    pl = 's' if hmfc[1] != 1 else ''
                    msg += '\n    - Missed %s target "O"%s\n' % (hmfc[1], pl)
                x = hmfc[2] + hmfc[4]
                if x > 0:
                    pl = 's' if x != 1 else ''
                    msg += '\n    - Pressed to %s non-target "O"%s' % (x, pl)
            ec.screen_prompt(msg + continue_msg, color=fcolor)
            fix.draw()
            ec.flip()
        return correct

    letters = [[None] * 8 for _ in range(2)]

    # Visual demo ###########################################################

    def demo_trial(ti, title, tstr):
        ec.screen_prompt(demo_instr, color=fcolor)
        samples = read_wav(op.join(stim_dir, p['stim_names'][ti]))[0]
        ec.load_buffer(samples)
        these_letters = [[p['letters'][ii] for ii in row]
                         for row in p['letter_mat'][ti]]

        # init and draw letters
        let_times = p['stim_times'][p['cond_mat'][ti][3]]
        xs = (let_times - let_times[0]) / (let_times[-1] - let_times[0])
        xs = xs / 2.
        ys = [0.1, -0.1]
        let_colors = [(0.25,) * 3, 'w', (0., 1., 0.)]
        sw_trial = (p['cond_mat'][ti, 0] == sw_id)
        assert tstr in ('male', 'female')
        strs = ['Male:', 'Female:'] if tstr == 'male' else ['Female:', 'Male:']
        cues = [visual.Text(ec, s, [-0.1, y], anchor_x='right', font_size=48,
                            color=let_colors[1-ii])
                for ii, (s, y) in enumerate(zip(strs, ys))]
        title = visual.Text(ec, title, [-0.1, 0.4], font_size=36, color='w',
                            anchor_x='left')
        for ii, y in enumerate(ys):
            for jj, x in enumerate(xs):
                if ii == 0 or jj >= p['n_cue_let']:
                    if jj >= p['n_cue_let']:
                        let = these_letters[ii][jj]
                    else:
                        let = 'A' if jj == 0 or not sw_trial else 'U'
                    letters[ii][jj] = visual.Text(ec, let, [x, y],
                                                  font_size=48,
                                                  anchor_x='left',
                                                  color=let_colors[0])
                    letters[ii][jj].draw()
        title.draw()
        [c.draw() for c in cues]
        ec.flip()
        ec.wait_secs(2.0)  # show stim for a bit
        for tidx, t in enumerate(let_times):
            color = let_colors[1]
            ii = 0
            if tidx >= p['n_cue_let'] and \
                    p['targ_pos'][ti][0][tidx-p['n_cue_let']]:
                color = let_colors[2]
            if sw_trial and tidx >= p['n_cue_let'] + p['gap_after']:
                ii = 1
            letters[ii][tidx].set_color(color)

            for jj in range(len(let_times)):
                for ii in range(2):
                    if ii == 0 or jj >= p['n_cue_let']:
                        letters[ii][jj].draw()
            title.draw()
            [c.draw() for c in cues]

            if tidx == 0:
                t0 = ec.start_stimulus(flip=True, start_of_trial=False)
            else:
                ec.flip(t0 + t)
        ec.flip(t0 + p['trial_durs'][ti])
        ec.stop()
        ec.screen_prompt('Press a button to continue.', color=fcolor)

    # Do training ###########################################################

    calibrated = False
    mf_id = p['idents'].index('male-female')
    fm_id = p['idents'].index('female-male')
    sw_id = p['attns'].index('switch')
    sp_ids = [p['spatials'].index('-30x30'), p['spatials'].index('30x-30')]
    ec.set_visible(False)
    bi = get_keyboard_input('Enter training block number (0): ', 0, int)
    ec.set_visible(True)
    ec.flip()
    switch_m = (p['cond_mat'][:, 0] == sw_id)
    spatial_m = np.logical_or(p['cond_mat'][:, 1] == sp_ids[0],
                              p['cond_mat'][:, 1] == sp_ids[1])
    mf_m = np.logical_or(p['cond_mat'][:, 2] == mf_id,
                         p['cond_mat'][:, 2] == fm_id)
    left_m = np.in1d(p['cond_mat'][:, 1],
                     np.where([s[0] == '-' for s in p['spatials']])[0])

    continue_msg = '\n\nPress a button to continue.'
    first_try = True
    n_tr = 0
    titles = ('Maintain attention:'.upper(), 'Switch attention:'.upper())
    while 0 <= bi < 6:  # use -1 to skip
        block_type = p['blocks'][session][n_tr % n_blocks][:2]
        trials = np.concatenate((p['block_trials'][block_type + '0'],
                                 p['block_trials'][block_type + '1']))
        tstr, dstr = _get_strs(block_type + '0')
        tr_trials = [
            np.where(mf_m & spatial_m & ~switch_m)[0],  # MF_S_M
            np.where(mf_m & spatial_m & switch_m)[0],   # MF_S_S
            np.where(mf_m & spatial_m)[0],              # MF_S_x
            np.where(mf_m & ~spatial_m)[0],             # MF_D_x
            np.where(~mf_m & spatial_m)[0],             # MM_S_x
            None,                                               # Any
        ]
        tr_trials[:-1] = [np.intersect1d(t, trials) for t in tr_trials[:-1]]
        assert len(tr_trials) == 6

        n_tr += 1
        ec.start_noise()
        ec.flip()
        ec.screen_prompt(('You will now do a training block of trials.\n\n'
                          + tr_instructions[bi] + continue_msg)
                         % (tstr, dstr), color=fcolor)
        if bi < 2 and first_try:
            idx = p['targ_pos'][tr_trials[bi]].any(axis=-1).all(axis=-1)
            demo_trial(tr_trials[bi][np.random.choice(np.where(idx)[0])],
                       titles[bi], tstr)
        first_try = False

        if not calibrated:  # use one calibration for all training runs
            ec.system_beep()
            el.calibrate()
            calibrated = True
        ec.write_data_line('training', bi)
        fix.draw()
        ec.flip()
        ec.wait_for_presses(5.0)
        count_mask = np.ones(n_tr_trials[bi], bool)
        if tr_trials[bi] is not None:
            these_trials = np.random.choice(tr_trials[bi], n_tr_trials[bi],
                                            False)
        else:
            assert bi == 5  # special case, last condition
            assert n_tr_trials[bi] == 16
            a = np.random.choice(tr_trials[0], 5, False)
            b = np.random.choice(tr_trials[1], 5, False)
            c = np.random.choice(tr_trials[3], 3, False)
            d = np.random.choice(tr_trials[4], 3, False)
            these_trials = np.concatenate((a, b, c, d))
            count_mask[len(a)+len(b):] = False
            order = np.random.permutation(np.arange(16))
            count_mask = count_mask[order]
            these_trials = these_trials[order]
        scores = np.array([run_trial(ti, True) for ti in these_trials])
        ec.stop_noise()

        m = np.sum(scores[count_mask])
        if m >= tr_thresholds[bi]:
            bi += 1
            first_try = True
            msg = 'Great job, you passed that block of training!\n\n'
        else:
            msg = ('You got %s trials correct, need %s to move on.'
                   % (m, tr_thresholds[bi]))
        ec.screen_prompt(msg + continue_msg, color=fcolor)

    # Testing ###############################################################

    ec.set_visible(False)
    bi = get_keyboard_input('Enter block number (0): ', 0, int)
    ec.set_visible(True)
    ec.flip()
    while 0 <= bi < n_blocks:
        block_type = p['blocks'][session][bi]
        trials = p['block_trials'][block_type]
        tstr, dstr = _get_strs(block_type)
        ec.flip()
        ec.screen_prompt(('You are ready to start experimental block %i/%i.'
                          '\n\nNo feedback will be provided.\n\n'
                          'Remember to try to press the button as quickly as '
                          'possible whenever the target speaker (only) says '
                          'the letter "O".' + dir_instr + '\n\n%s')
                         % (bi+1, n_blocks, tstr, dstr, continue_msg),
                         color=fcolor)
        # start of each block
        ec.start_noise()
        ec.write_data_line('block', block_type)
        ec.system_beep()
        el.calibrate()
        ec.flip()
        fix.draw()
        ec.flip()
        ec.wait_for_presses(5.0)  # wait to settle

        # run each trial
        for ti in trials:
            run_trial(ti)

        # end of each block
        el.stop()
        ec.stop_noise()
        bi += 1
        if bi < len(p['block_trials']):
            ec.screen_prompt('You are done with block %s/%s.\n\nFeel free to '
                             'take a break, then press the button when you '
                             'are ready to continue.' % (bi, n_blocks),
                             color=fcolor)
    ec.screen_prompt('You are done, thanks for your participation!',
                     max_wait=5., color=fcolor)
