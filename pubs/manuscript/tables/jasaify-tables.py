#!/usr/bin/env python3

import os
import sys

if len(sys.argv) < 2:
    sys.exit(f'Usage: {sys.argv[0]} file1 [file2...]')

for oldfile in sys.argv[1:]:
    if not os.path.exists(oldfile):
        sys.exit(f'ERROR: file {oldfile} not found')
    newfile = os.path.join('..', oldfile)
    with open(oldfile, 'r') as f, open(newfile, 'w') as g:
        size = '\\scriptsize' if oldfile.startswith('table-rt') else ''
        for line in f:
            # replace horizontal rules
            line = line.replace('\\toprule', '\\hline\\hline')
            line = line.replace('\\bottomrule', '\\hline\\hline')
            line = line.replace('\\midrule', '\\hline')
            # replace unicode minus
            line = line.replace('−', r'\textminus{}')
            # swap order so caption goes outside tabular
            if line.startswith('\\begin{longtable}'):
                line = line.replace('longtable', 'tabular')
                captionline = f.readline()
                captionline = captionline.replace('\\\\', '')
                g.write(captionline)
                g.write(line)
                continue
            line = line.replace('longtable', 'tabular')
            table = 'table' if 'ttest' in oldfile else 'table*'
            line = line.replace('\\begin{center}', f'\\begin{{{table}}}{size}')
            line = line.replace('\\end{center}', f'\\end{{{table}}}')
            g.write(line)
