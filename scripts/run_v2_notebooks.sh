#!/usr/bin/env bash
# Executes the v2 (lung-crop) training + evaluation notebooks in order, saving outputs inside each notebook.
set -e
cd "$(dirname "$0")/../notebooks"
for nb in 08_PulmoNet_v2_LungCrop 09_EfficientNetB0_v2_LungCrop 10_v2_Evaluation; do
  echo "=== $(date '+%H:%M:%S') running $nb"
  ../.venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=pulmosight --ExecutePreprocessor.timeout=-1 "$nb.ipynb"
done
echo "=== $(date '+%H:%M:%S') all done"
