# Repository for “Auditory attention switching with listening difficulty: Behavioral and pupillometric measures”

The general analysis pipeline follows.  Optional scripts are in (parentheses):
1. `behavioral-data-cleaning.py`
1. (`behavioral-prelim-analysis.py`)
1. `behavioral-prep-for-modelling.R`
    1. (`model-accuracy-exploratory.R`)
    1. (`model-accuracy-selection.R`)
    1. `model-accuracy-final.R`
    1. `model-accuracy-posthoc.R`
    1. (`model-rt-exploratory.R`)
    1. `model-rt-final.R`
    1. `model-rt-posthoc.R`
1. `pupil-data-cleaning.py`
1. `pupil-metrics.py`
1. `pupil-stats.py`

After that, the plotting functions can be run in any order:
- `plot-trial-diagram.py`
- `plot-accuracy.py`
- `plot-rt.py`
- `plot-pupil-attn-by-space-group.py`
- `plot-pupil-group-by-space-attn.py`
- `posthocs.py`

Typesetting should all be done with the makefile `pubs/manuscript/Makefile`:
- `make draft` for a preprint
- `make supplement` for the supplement
- `make reprint` for a version formatted like the final journal article (for
    testing page counts, figure sizes, etc.)
- `make submission` for separate `.tex` and `.pdf`
- `make upload` to gather everything submittable into one folder
