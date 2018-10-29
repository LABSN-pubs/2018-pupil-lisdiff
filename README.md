# Repository for “Auditory attention switching with listening difficulty: Behavioral and pupillometric measures”

For distribution, the data have been compressed, so if you want to reproduce
the published analyses, the first step is to run `uncompress-data.sh` from that
script’s location.

The general analysis pipeline follows.  Optional scripts are in (parentheses):
1. `behavioral-data-cleaning.py`: parses the raw experimental output into two master CSV files
1. `behavioral-prep-for-modelling.R`: more data cleaning; sets up factor contrasts; saves in `.RData` format
    1. (`model-accuracy-exploratory.R`): tries different modeling approaches with `lme4::glmer`
    1. (`model-accuracy-selection.R`): model comparison / selection
    1. `model-accuracy-final.R`: runs the final model through `afex::mixed` so we get p-values
    1. `model-accuracy-posthoc.R`: runs some posthoc contrasts and writes the results to CSV files
    1. (`model-rt-exploratory.R`): tries different modeling approaches to the reaction time data
    1. `model-rt-final.R`: final reaction time model
    1. `model-rt-posthoc.R`: runs post-hoc contrasts on reaction time data and saves to CSV files
1. `pupil-data-cleaning.py`: parses the raw pupillometry data
    1. `pupil-metrics.py`: extracts summary measures from pupil data (AUC, peak latency, etc.)
    1. `pupil-stats.py`: runs the non-parametric cluster stats on the pupil data and saves results to YAML files

After that, the plotting functions can be run in any order. These generate the figures in the manuscript:
- `plot-trial-diagram.py`
- `plot-accuracy.py`
- `plot-rt.py`
- `plot-pupil-attn-by-space-group.py`
- `plot-pupil-group-by-space-attn.py`
- `posthocs.py`

Typesetting should all be done with the makefile `pubs/manuscript/Makefile`:
- `make draft` for a preprint, `make supplement` for the supplement
    - `make arxivpreprint` to merge the draft and supplement into a single PDF (requires `pdftk`)
- `make reprint` for a version formatted like the final journal article (for
    testing page counts, figure sizes, etc.)
- `make submission` for separate `.tex` and `.pdf` files, with PDF formatted
  for reviewer comfort (inline figures/tables, line numbers, double spacing)
- `make upload` or `make R1` to gather everything submittable into one folder
