#!/usr/bin/env bash
#
# Full NSAP pipeline — fresh run.
#   Step 1  patch generation + model-level evaluation   (NSAP conda env, GPU)
#   Step 2  MIM (man-in-the-middle) evaluation          (embench conda env, GPU)
#
# The SceneGraphBench sim dataset (SceneGraphBench/data/kitchen_pilot) is assumed
# ALREADY generated — this script does not regenerate it.
#
# Two conda envs are used via their env pythons directly (conda run is unreliable
# on this box). Override with NSAP_PY / EMBENCH_PY if your paths differ.
#
# Usage:  bash run_full_pipeline.sh
#
set -euo pipefail
cd "$(dirname "$0")"                                  # always run from the NSAP root

NSAP_PY="${NSAP_PY:-/home/usi/miniconda3/envs/NSAP/bin/python}"
EMBENCH_PY="${EMBENCH_PY:-/home/usi/miniconda3/envs/embench/bin/python}"
TASKS="rung2/gap_tasks_v2.json"                       # the gap-task set the MIM patches map to

for p in "$NSAP_PY" "$EMBENCH_PY"; do
    [ -x "$p" ] || { echo "ERROR: python not found: $p" >&2; exit 1; }
done

# ── Step 0: clean the outputs a fresh run regenerates ─────────────────────────
# output/ is gitignored, so these deletions are PERMANENT. We remove ONLY what
# the pipeline below rebuilds. Deliberately NOT touched:
#   output/_archive_*   (archived historical runs)
#   output/relsweep     (relation_lambda_sweep.py — a separate ablation)
# chexp MUST be cleared: chameleon_experiment.py resumes/reuses any patch it finds.
echo "===== Step 0: clearing output/chexp + output/rung2 (fresh run) ====="
rm -rf output/chexp output/rung2

# ── Step 1: patch generation + model-level eval ───────────────────────────────
# Per target (6 category / 2 attribute / 3 relation):
#   Phase 1  universal loud patch (all scenarios, no chameleon)
#   Phase 2  per-scenario object-init chameleon patch (hidden but still flips)
# Each patch is evaluated inline with `run_attack.py --eval` (flip rate + isolation).
# Writes output/chexp/{uni_*,cham_*}/ + output/chexp/sweep_summary.json.
# NOTE: this is the long step — 11 targets x 2 phases of training on the GPU.
echo "===== Step 1: patch generation + model-level evaluation (NSAP env) ====="
"$NSAP_PY" chameleon_experiment.py

# ── Step 2: MIM (man-in-the-middle) evaluation ────────────────────────────────
# Runs against the cham_* patches from Step 1 (gap_tasks_v2.json maps each task to
# its patch path; scenario ids are reproduced deterministically by Step 1).
echo "===== Step 2a: MIM head-flip rate (embench env) ====="
# MLLM-independent: ESCA's GroundingDINO + SGClip on the target's own box,
# clean vs patched, over N jittered viewpoints. Reports flip rate + clean baseline.
"$EMBENCH_PY" rung2/flip_rate.py --tasks-file "$TASKS" --n 6

echo "===== Step 2b: MIM behavioral task-success (embench env) ====="
# Full closed-loop planner episodes per condition -> did the robot's behavior change?
# Writes output/rung2/{<task>_result.json, sweep_summary.json} + per-step frames.
"$EMBENCH_PY" rung2/harness.py --tasks-file "$TASKS" --task all \
    --conditions base_benign,esca_benign,esca_patch

echo
echo "===== DONE ====="
echo "  patches + model-eval : output/chexp/   (sweep_summary.json)"
echo "  MIM results          : output/rung2/   (sweep_summary.json + *_result.json)"
