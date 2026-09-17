#!/bin/bash
# Run the staged attribute/relation MIM benchmark for one instruction variant.
#
#   bash rung2/run_staged_variants.sh A_donothing        # baseline (what we ran)
#   bash rung2/run_staged_variants.sh B_redirect
#   bash rung2/run_staged_variants.sh C_trust_sg
#   bash rung2/run_staged_variants.sh B_redirect attribute 1     # one head, 1 rep (quick)
#
# TAU IS AN EXPLICIT ARGUMENT. Do not rely on unsetting NSAP_AGGR_THRES: make_nav_planner
# falls back to 0.4 (the harness's own value), NOT to ESCA's released default of 0.1.
set -euo pipefail
# Paths are overridable so a fresh clone works without editing this file. The defaults
# reproduce the machine the paper's runs were made on.
NSAP_ROOT="${NSAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$NSAP_ROOT"
PY_EMB="${PY_EMB:-${CONDA_ROOT:-$HOME/miniconda3}/envs/${ENV_EMBENCH:-embench}/bin/python}"
# Prefer an explicit ESCA_ROOT, then a sibling checkout, then the submodule.
if [ -z "${ESCA_ROOT:-}" ]; then
  if [ -d "$HOME/ESCA/embodiedbench" ]; then ESCA_ROOT="$HOME/ESCA"
  else ESCA_ROOT="$NSAP_ROOT/external/ESCA"; fi
fi
export ESCA_ROOT

VARIANT="${1:?usage: run_staged_variants.sh <variant> [heads] [reps] [tau]}"
HEADS="${2:-attribute,relation}"
REPS="${3:-3}"
TAU="${4:-0.1}"                      # 0.1 = ESCA's released default (esca.py:178)
TOPK="${5:-1}"                       # 1 = ESCA's released default (esca.py:177); the gate
                                     # admits only this many objects into the scene graph
CONDS="${6:-base_benign,esca_benign,esca_patch}"   # add lang_benign,lang_patch to blind the
                                     # planner to the image and make the SG fully load-bearing
TASKS="rung2/tasks_${VARIANT}.json"
[ -f "$TASKS" ] || { echo "missing $TASKS -- run: python rung2/make_task_variants.py"; exit 1; }

export NSAP_AGGR_THRES="$TAU"        # set EXPLICITLY; unsetting gives 0.4, not 0.1
# Separate tree per (variant, tau) so a re-run cannot destroy earlier logs.
export NSAP_TOP_K="$TOPK"
TAG=$(echo "$CONDS" | grep -q lang && echo "_lang" || echo "")
# $RUN_TAG appends a suffix so the SAME configuration can be run twice without the
# second run overwriting the first (that is how benchmark drift is measured).
export NSAP_OUT_ROOT="output/rung2/staged_${VARIANT}_tau${TAU}_k${TOPK}${TAG}${RUN_TAG:+_$RUN_TAG}"

echo "variant=$VARIANT  heads=$HEADS  reps=$REPS  tau=$TAU  top_k=$TOPK  tasks=$TASKS"
echo "output -> $NSAP_OUT_ROOT"
date
"$PY_EMB" -u rung2/staged_bench.py \
  --episodes "$TASKS" \
  --conditions "$CONDS" \
  --model gpt-4o --reps "$REPS" --tasks "$HEADS"
echo "exit: $?"; date

echo
echo "########## WHY DID IT DO THAT ##########"
"$PY_EMB" rung2/analyze_steps.py "$NSAP_OUT_ROOT"
