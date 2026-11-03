#!/usr/bin/env bash
# Every table and Figs. 1-4 of the paper from results/ (and data/ for the dataset figures), training environment.
source "$(dirname "$0")/common.sh"
set -e
$PY analysis/make_fig_dataset.py          # Fig. 1, Fig. 2, results/dataset_summary.json (needs data/)
$PY analysis/make_fig_action_latent.py    # Fig. 3 (needs data/)
$PY analysis/make_table_bc.py             # Table 3
$PY analysis/make_table_factorial.py      # Tables 4 and 7 (needs data/ for the out-of-range column)
$PY analysis/make_table_compare.py        # Table 5
$PY analysis/make_table_eta.py            # Table 8
$PY -m omiga_rev.diagnostics.analyze      # Table 9
$PY analysis/make_fig_spread.py           # Fig. 4
