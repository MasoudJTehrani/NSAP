#!/bin/bash
# Re-run the CATEGORY (EB-Nav) MIM benchmark at an explicit threshold.
#
#   bash rung2/run_category.sh              # tau=0.1, ESCA's released default, 3 reps
#   bash rung2/run_category.sh 0.4 3        # tau=0.4, what the -40 result was measured at
#   bash rung2/run_category.sh 0.1 1        # quick look, 1 rep
#
# NOTE: tau MUST be passed explicitly. make_nav_planner falls back to 0.4 when
# NSAP_AGGR_THRES is unset, which is the harness's value, not ESCA's 0.1.
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

TAU="${1:-0.3}"                      # ESCA nav evaluator uses 0.3
REPS="${2:-3}"
FAITHFUL="${3:-1}"                   # 1 = ESCA's nav evaluator config (n_shot 3, chat_history,
                                     #     multistep, tp 4, aggr_thres 0.3, top_k 1)
GD_ONLY="${4:-$FAITHFUL}"            # 1 = skip SGClip (ESCA's shipped nav setting)
                                     # 0 = SGClip in the loop, everything else unchanged

export NSAP_AGGR_THRES="$TAU"
export NSAP_ESCA_FAITHFUL="$FAITHFUL"
export NSAP_GD_ONLY="$GD_ONLY"
export NSAP_OUT_ROOT="output/rung2/ebnav_tau${TAU}_faithful${FAITHFUL}_gdonly${GD_ONLY}"

echo "CATEGORY EB-Nav MIM   tau=$TAU  reps=$REPS  faithful=$FAITHFUL  gd_only=$GD_ONLY  model=gpt-4o"
echo "output -> $NSAP_OUT_ROOT"
date
"$PY_EMB" -u rung2/eb_bench_runner.py \
  --eval-sets long_horizon,visual_appearance \
  --indices 2,3,4,7,19 \
  --conditions base_benign,esca_benign,esca_patch \
  --model gpt-4o --reps "$REPS"
echo "exit: $?"; date

echo
echo "########## WHY DID IT DO THAT ##########"
"$PY_EMB" rung2/analyze_steps.py "$NSAP_OUT_ROOT"
