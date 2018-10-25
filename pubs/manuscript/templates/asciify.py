#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

infile = sys.argv[-2]
outfile = sys.argv[-1]

with open(infile, 'r') as f, open(outfile, 'w') as g:
    for line in f:
        for dp in ('_d′_', '_d_′', '*d′*', '*d*′', 'd′'):
            line = line.replace(dp, r'$d\thinspace^\prime$')
        for chi in ('_χ_', '*χ*', 'χ'):
            line = line.replace(chi, r'$\chi$')
        for beta in ('_β_', '*β*', 'β'):
            line = line.replace(beta, r'$\beta$')
        line = line.replace('≥', r'$\geq$ ')
        line = line.replace('±', r'\textpm{}')
        line = line.replace('−', r'\textminus{}')
        line = line.replace('×', r'\texttimes{}')
        line = line.replace('¼', r'\textonequarter{}')
        line = line.replace('½', r'\textonehalf{}')
        line = line.replace('²', r'\textsuperscript{2}')
        line = line.replace('°', r'\textdegree{}')
        line = line.replace('’', "'")
        g.write(line)
