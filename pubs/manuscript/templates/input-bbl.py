#!/usr/bin/env python3

import os
import sys

if len(sys.argv) != 4:
    sys.exit(f'Usage: {sys.argv[0]} texfile bblfile outfile')

texfile = sys.argv[1]
bblfile = sys.argv[2]
outfile = sys.argv[3]

for file in (texfile, bblfile):
    if not os.path.exists(file):
        sys.exit(f'ERROR: input file {file} not found')

with open(texfile, 'r') as f, open(outfile, 'w') as g:
    for line in f:
        if line.startswith('\\bibliography{'):
            line = f'\\input{{{bblfile}}}\n'
            # g.write('\\section*{References}\n')
        g.write(line)
