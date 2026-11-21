#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export PYTHONHASHSEED=2026 MPLBACKEND=Agg
export SEISDAE_V2_OUT="${SEISDAE_V2_OUT:-$PWD/runs/reproduced_v2}"
mkdir -p "$SEISDAE_V2_OUT"
python -m unittest discover -s tests -v
python scripts/execute_local.py notebooks/0[1-7]*.ipynb
python -m src.verify runs/notebook_test
python -m src.v2.stats
python -m src.v2.baseline
python -m src.v2.training
python -m src.v2.sensitivity
python -m src.v2.training --sizes
python -m src.v2.final
python -m src.v2.failures
python -m src.verify "$SEISDAE_V2_OUT" --v2
# Archived V2 report is retained; the current report is built by make assets.
