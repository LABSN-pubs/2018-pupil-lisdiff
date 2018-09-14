#!/usr/bin/env python3

import os
import sys

if len(sys.argv) != 3:
    sys.exit(f'Usage: {sys.argv[0]} infile outfile')

infile = sys.argv[1]
outfile = sys.argv[2]

if not os.path.exists(infile):
    sys.exit(f'ERROR: input file {infile} not found')

# figure mapping
fm = {'trial-diagram': 'Figure1',
      'accuracy-figure-probit': 'Figure2',
      'rt-figure': 'Figure3',
      'capd-pupil-deconv-group-by-space-attn': 'Figure4',
      'capd-pupil-deconv-attn-by-space-group': 'Figure5'}


with open(infile, 'r') as f, open(outfile, 'w') as g:
    for line in f:
        if line.strip().startswith('\\includegraphics'):
            line = line.replace('\\includegraphics{figures/',
                                '\\includegraphics{')
            for k, v in fm.items():
                line = line.replace(k, v)
        g.write(line)
