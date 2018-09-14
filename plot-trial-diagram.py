#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script 'trial-vocoder.py'
===============================================================================

This script plots a trial diagram for the pupil vocoder switching task.
"""
# @author: Dan McCloy (drmccloy@uw.edu)
# Created on Wed Sep 23 16:57:41 2015
# License: BSD (3-clause)

import os.path as op
import numpy as np
import matplotlib.pyplot as plt

plt.ioff()

# file I/O
outdir = op.join('figures', 'manuscript')
styledir = 'styles'

# color defs
gapcol = 'C2'
male_col = 'C4'
female_col = 'C5'
cue_box_col = 'w'
maintcol = 'C0'
switchcol = 'C1'
lettercol = 'w'
garnish_col = 'k'
timeline_col = 'C4'
slot_col = 'C3'

# plot style
style_files = ('font-libertine.yaml', 'garnish.yaml', 'trial-colors.yaml')
plt.style.use([op.join(styledir, sf) for sf in style_files])

# set up figure
fig = plt.figure(figsize=(6.5, 1.5))
ax = plt.Axes(fig, [0.025, 0.15, 0.9, 0.85])
ax.axis('off')
fig.add_axes(ax)

# temporal landmarks
letter_dur = 0.4
cue_gap = 0.4
switch_gap = 0.6
end_gap = 0.2
times = (0, letter_dur,
         2 * letter_dur,
         2 * letter_dur + cue_gap,
         3 * letter_dur + cue_gap,
         4 * letter_dur + cue_gap,
         4 * letter_dur + cue_gap + switch_gap,
         5 * letter_dur + cue_gap + switch_gap,
         6 * letter_dur + cue_gap + switch_gap,
         6 * letter_dur + cue_gap + switch_gap + end_gap)

# maint / switch lines
ax.plot((times[2], times[-1]), (2.1, 2.1),
        color=maintcol, linewidth=3.5, solid_capstyle='butt')
ax.plot((times[2], times[5], times[6], times[-1]), (1.9, 1.9, 0, 0),
        color=switchcol, linewidth=3.5, linestyle='--')
ax.text(times[-1] + 0.1, 2.1, 'maintain', color=maintcol, ha='left',
        va='center')
ax.text(times[-1] + 0.1, 0, 'switch', color=switchcol, ha='left', va='center')

# boxes & letters
centers_x = ([times[1] - 0.03, times[1] + 0.01] +
             [np.mean(times[3:5]), np.mean(times[4:6]),
              np.mean(times[6:8]), np.mean(times[7:9])] * 2)
centers_y = [1.95] * 6 + [-0.05] * 4
box_x = ([(times[0], times[0], times[2], times[2])] +
         [(times[3], times[3], times[5], times[5]),
          (times[6], times[6], times[8], times[8])] * 2)
box_y = [(1.5, 2.5, 2.5, 1.5)] * 3 + [(-0.5, 0.5, 0.5, -0.5)] * 2
# cue AU target O foils DEGPV
box_l = ['AA', 'AU', 'E', 'O', 'P', 'O', 'P', 'V', 'D', 'E']

color = [maintcol, switchcol] + [lettercol] * 8
bcolor = [cue_box_col] + [male_col] * 2 + [female_col] * 2
ecolor = [garnish_col] + ['none'] * 4
wt = ['bold'] * 10
ha = ['right', 'left'] + ['center'] * 8
for x, y, b, e in zip(box_x, box_y, bcolor, ecolor):
    ax.fill(x, y, b, alpha=1, zorder=4, edgecolor=e, linewidth=0.5)
for x, y, s, c, h, w in zip(centers_x, centers_y, box_l, color, ha, wt):
    ax.text(x, y - 0.04, s, ha=h, va='center', color=c, weight=w, zorder=5)
ax.vlines([times[4], times[7]], ymin=-0.5, ymax=2.5, zorder=5, color=lettercol,
          linewidth=1.5)
ax.text(times[1], centers_y[0], '/', color=garnish_col, ha='center',
        va='center', zorder=5)

# switch gap
bot = -2.6
ht = 0.6
top = bot + ht
lwd = 0.4
rect = plt.Rectangle((times[5], bot), width=switch_gap, height=ht, zorder=4,
                     fill=False, linewidth=lwd, edgecolor=gapcol,
                     clip_on=False)
yy = np.tile([bot + ht, bot], (6, 1))
xx = [(x, x + 0.1) for x in np.linspace(times[5], times[6] - 0.1, 6)]
for x, y in zip(xx, yy):
    plt.plot(x, y, linewidth=lwd, color=gapcol, solid_capstyle='butt',
             zorder=4, clip_on=False)
ax.add_artist(rect)
ax.set_clip_on(False)
ax.text(np.mean(times[5:7]), -1.9, 'switch gap', ha='center', va='bottom',
        fontsize=10, color=gapcol)

# captions
ax.text(times[1], 3.6, 'Cue', color=garnish_col, fontsize=11, ha='center',
        va='center')
ax.text(np.mean(times[5:7]), 3.6, 'Concurrent target and distractor streams',
        color=garnish_col, fontsize=11, ha='center', va='center',
        weight='normal')
ax.text(times[3], 2.55, 'male', color=male_col, fontsize=9, ha='left',
        va='baseline', weight='bold')
ax.text(times[3], 0.55, 'female', color=female_col, fontsize=9, ha='left',
        va='baseline', weight='bold')

# timeline
arr_y = -2.3
arr_xmax = times[-1] + 0.2
tcklen = 0.25
ticktimes = list(times)
ticktimes.pop(1)
ticktimes.pop(-1)
ticktimes = [np.round(t, 1) for t in ticktimes]
# ticktimes = [0, 1, 1.5, 2.0, 2.5, 3.1, 3.6, 4.1]
ticklabels = [str(tt) for tt in ticktimes]
ax.vlines(ticktimes, arr_y - tcklen, arr_y + tcklen, linewidths=0.5, zorder=5,
          color=timeline_col)
for x, y, s in zip(ticktimes, [arr_y - 4 * tcklen] * len(ticktimes),
                   ticklabels):
    ax.text(x, y, s, ha='center', va='baseline', fontsize=9,
            color=timeline_col)
arr = ax.arrow(-0.2, arr_y, arr_xmax, 0, head_width=0.4, head_length=0.15,
               fc=timeline_col, ec=timeline_col, linewidth=0.5, zorder=5)
plt.annotate('time (s)', (arr_xmax, arr_y),  xytext=(3, 0), fontsize=9,
             textcoords='offset points', ha='left', va='center',
             color=timeline_col)


# finalize
plt.ylim(-2.8, 4.8)
plt.xlim(-0.1, times[-1] + 0.3)
fig.savefig(op.join(outdir, 'trial-diagram.pdf'))
