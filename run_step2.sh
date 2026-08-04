#!/usr/bin/env bash
#
# Re-run ONLY Step 2 of the NSAP pipeline — MIM (man-in-the-middle) evaluation.
# Assumes Step 1 already produced the patches in output/chexp/.
#
#   Step 2a  MIM head-flip rate       (rung2/flip_rate.py)
#   Step 2b  MIM behavioral success   (rung2/harness.py, --task all)
#
# Runs in the embench conda env (has ESCA + AI2-THOR). Override EMBENCH_PY if your
# interpreter path differs.
#
# Usage:  bash run_step2.sh
#
set -euo pipefail
cd "$(dirname "$0")"                                  # run from the NSAP root

EMBENCH_PY="${EMBENCH_PY:-/home/usi/miniconda3/envs/embench/bin/python}"
TASKS="rung2/gap_tasks_v2.json"
[ -x "$EMBENCH_PY" ] || { echo "ERROR: python not found: $EMBENCH_PY" >&2; exit 1; }

# The MIM patches must exist (Step 1). Fail early with a clear message otherwise.
if [ ! -d output/chexp ]; then
    echo "ERROR: output/chexp not found — run Step 1 first (bash run_full_pipeline.sh)." >&2
    exit 1
fi

# Fresh behavioral output (harness resumes from any existing per-task result file).
rm -rf output/rung2

echo "===== Step 2a: MIM head-flip rate (embench env) ====="
"$EMBENCH_PY" rung2/flip_rate.py --tasks-file "$TASKS" --n 6

echo "===== Step 2b: MIM behavioral task-success (embench env) ====="
# --task all: run every task in the tasks file (without it, harness defaults to the
# single id 'category_fridge', which isn't in gap_tasks_v2.json -> KeyError).
"$EMBENCH_PY" rung2/harness.py --tasks-file "$TASKS" --task all \
    --conditions base_benign,esca_benign,esca_patch

echo
echo "===== Step 2 DONE — results in output/rung2/ (sweep_summary.json + *_result.json) ====="
