#!/usr/bin/env bash
# Executes the training + comparison notebooks in order, saving outputs inside each notebook.
set -e
cd "$(dirname "$0")/../notebooks"
for nb in 02_Simple_CNN_Baseline 03_PulmoNet_Custom_CNN 04_EfficientNetB0_Transfer_Learning 05_Model_Comparison; do
  echo "=== $(date '+%H:%M:%S') running $nb"
  ../.venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=pulmosight --ExecutePreprocessor.timeout=-1 "$nb.ipynb"
done
echo "=== $(date '+%H:%M:%S') all done"
