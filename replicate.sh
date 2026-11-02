#!/usr/bin/env bash
# NSAP replication driver.
#
#   bash replicate.sh setup            prerequisites: weights, detector, dataset, ESCA hook
#   bash replicate.sh fetch-artifacts  download our trained patches + run logs (skip training)
#   bash replicate.sh train-universal  Stage 1: 5 universal patches      (~5 h on one GPU)
#   bash replicate.sh train-chameleon  Stage 2: 30 per-scenario patches  (~4 h)
#   bash replicate.sh train-ablation   content-loss ablation: 60 patches           (~1.5 h)
#   bash replicate.sh eval-ablation    flip rate and LPIPS for both ablation arms  (~1 h)
#   bash replicate.sh eval-model       model-level: outcome, deployment, stealth   (~2 h)
#   bash replicate.sh eval-system      system-level: staged + EB-Nav benchmarks    (~36 h, paid API)
#   bash replicate.sh numbers          recompute every paper number from output/  (seconds)
#   bash replicate.sh all              everything above, in order
#
#   DRY_RUN=1 bash replicate.sh train-chameleon     print the commands, run nothing
#
# Override anything with environment variables:
#   NSAP_ROOT    repo root                  (default: this script's directory)
#   CONDA_ROOT   conda install              (default: $HOME/miniconda3)
#   ENV_NSAP     training/model-eval env    (default: NSAP)
#   ENV_EMBENCH  ESCA + benchmark env       (default: embench)
#   ESCA_ROOT    ESCA checkout to run from  (default: $NSAP_ROOT/external/ESCA)
#
# The benchmark stage calls the OpenAI API (gpt-4o) and costs real money. It is also the
# only stage that does not reproduce our numbers: see REPLICATION.md, "what replicates".
set -euo pipefail

NSAP_ROOT="${NSAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$NSAP_ROOT"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NSAP="${ENV_NSAP:-NSAP}"
ENV_EMBENCH="${ENV_EMBENCH:-embench}"
PY_NSAP="$CONDA_ROOT/envs/$ENV_NSAP/bin/python"
PY_EMB="$CONDA_ROOT/envs/$ENV_EMBENCH/bin/python"
export ESCA_ROOT="${ESCA_ROOT:-$NSAP_ROOT/external/ESCA}"

# 8 GB cards fragment badly across the three planner instances the benchmark builds.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DRY_RUN="${DRY_RUN:-0}"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
run()  { if [ "$DRY_RUN" = "1" ]; then printf '  %s\n' "$*"; else printf '  + %s\n' "$*"; eval "$@"; fi; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
# Like run(), but a failure is reported and the loop continues. Used by the ablation,
# where one unusable scenario should not discard the other fifty-nine patches.
run_ok() { if [ "$DRY_RUN" = "1" ]; then printf '  %s\n' "$*"; else printf '  + %s\n' "$*"
           eval "$@" || printf '\033[31m  !! FAILED, continuing\033[0m\n'; fi; }

# The EB-Nav scenes live under SceneGraphBench/data when generated locally and under
# dataset/ when downloaded from the Hub. Accept either.
resolve_ebnav() {
  for d in SceneGraphBench/data/kitchen_ebnav dataset/kitchen_ebnav; do
    [ -f "$d/manifest.json" ] && { echo "$d"; return; }
  done
  echo "SceneGraphBench/data/kitchen_ebnav"   # not present yet; setup will say so
}

# ---------------------------------------------------------------- setup

stage_setup() {
  say "Checking environments"
  [ -x "$PY_NSAP" ] || die "no python at $PY_NSAP (set CONDA_ROOT / ENV_NSAP, or create the env from environment.yml)"
  [ -x "$PY_EMB"  ] || die "no python at $PY_EMB (set CONDA_ROOT / ENV_EMBENCH, or follow external/ESCA/install.sh)"
  echo "  train/model-eval : $PY_NSAP"
  echo "  ESCA/benchmark   : $PY_EMB"

  say "Downloading weights, detector and dataset"
  run "'$PY_NSAP' NSAP.py --with-detection --with-dataset"

  say "Generating the EB-Nav scene-matched dataset"
  local ebnav; ebnav=$(resolve_ebnav)
  if [ -f "$ebnav/manifest.json" ]; then
    echo "  already present: $ebnav"
  else
    # Needs a display for AI2-THOR. On a headless box: xvfb-run -a bash replicate.sh setup
    run "'$PY_EMB' rung2/gen_ebnav_dataset.py --config rung2/kitchen_ebnav.json --out SceneGraphBench/data/kitchen_ebnav"
  fi

  say "Generating the staged task variants"
  run "'$PY_NSAP' rung2/make_task_variants.py"

  say "Checking the ESCA debug hook"
  local esca_py="$ESCA_ROOT/embodiedbench/planner/navigation/esca.py"
  [ -f "$esca_py" ] || die "ESCA not found at $ESCA_ROOT (git submodule update --init --recursive)"
  if grep -q "_nsap_rel_scored" "$esca_py"; then
    echo "  present"
  else
    echo "  missing, applying replication/esca_nsap_debug_hooks.patch"
    run "git -C '$ESCA_ROOT' apply '$NSAP_ROOT/replication/esca_nsap_debug_hooks.patch'"
  fi

  say "Setup complete"
}

# ---------------------------------------------------------------- artifacts

stage_fetch_artifacts() {
  say "Fetching the published patches and run logs"
  run "'$PY_NSAP' -c 'from functions.download_model import download_artifacts; download_artifacts()'"
}

# ---------------------------------------------------------------- stage 1

stage_train_universal() {
  local only="${1:-}"
  say "Stage 1: universal patches  (replication/universal_manifest.tsv)"
  grep -v '^#' replication/universal_manifest.tsv | tail -n +2 |
  while IFS=$'\t' read -r head mode target attribute relation sim_data data_source coverage steps batch lambda_tv; do
    [ -n "${head:-}" ] || continue
    [ -n "$only" ] && [[ "$head" != *"$only"* ]] && continue
    local extra=""
    [ "$attribute"   != "-" ] && extra="$extra --attribute '$attribute'"
    [ "$mode" = "relation" ] && extra="$extra --relation '$relation'"
    [ "$data_source" != "-" ] && extra="$extra --data-source '$data_source'"
    echo "--- $head"
    run "'$PY_NSAP' run_attack.py --mode '$mode' --target '$target'$extra \
      --sim-data '$sim_data' --steps '$steps' --batch '$batch' --coverage '$coverage' \
      --lambda-tv '$lambda_tv' --no-chameleon --seed 0 --output-dir output/universal_patches"
  done
}

# ---------------------------------------------------------------- stage 2

stage_train_chameleon() {
  local only="${1:-}"
  local ebnav; ebnav=$(resolve_ebnav)
  say "Stage 2: chameleon patches  (replication/chameleon_manifest.tsv)"
  grep -v '^#' replication/chameleon_manifest.tsv | tail -n +2 |
  while IFS=$'\t' read -r head scenario mode target attribute relation sim_data coverage lambda_content steps eot_crops init_patch; do
    [ -n "${head:-}" ] || continue
    [ -n "$only" ] && [[ "$head" != *"$only"* ]] && continue
    [ -f "$init_patch" ] || die "missing Stage 1 patch $init_patch (run train-universal, or fetch-artifacts)"
    # The manifest records where the EB-Nav scenes sat when we trained; resolve it here.
    # It also labels those rows `ebnav_<id>` to keep the output paths distinct, but the
    # dataset's own ids carry no prefix, so strip it before passing --only-scenario.
    local sid="$scenario"
    if [ "$sim_data" = "SceneGraphBench/data/kitchen_ebnav" ]; then
      sim_data="$ebnav"; sid="${scenario#ebnav_}"
    fi
    if [ -f "output/chameleon_patches/$head/$scenario/$head/${head}_final.pt" ]; then
      echo "--- $head / $scenario  (already trained, skipping)"; continue
    fi
    local extra=""
    [ "$attribute" != "-" ] && extra="$extra --attribute '$attribute'"
    [ "$mode" = "relation" ] && extra="$extra --relation '$relation'"
    echo "--- $head / $scenario"
    run "'$PY_NSAP' run_attack.py --mode '$mode' --target '$target'$extra \
      --sim-data '$sim_data' --only-scenario '$sid' \
      --init-patch '$init_patch' --object-init --lambda-content '$lambda_content' \
      --steps '$steps' --batch 1 --eot-crops '$eot_crops' --coverage '$coverage' \
      --lambda-tv 0 --no-chameleon --seed 0 \
      --output-dir 'output/chameleon_patches/$head/$scenario'"
  done
}

# The content loss and the object initialisation are what make a Stage-2 patch a
# chameleon. Comparing chameleons against the Stage-1 universal patches cannot isolate
# either, because those differ in six ways at once (loss, initialisation, steps,
# coverage, training data and evaluation split). These arms change one thing at a time,
# on the same scenarios, from the same Stage-1 anchor, with the same steps, coverage,
# crops and seed as the chameleon they are compared against:
#
#   content0   object-init, lambda_content 0   -> the content loss alone (chameleon vs this)
#   noinit     no object-init, lambda_content 0 -> initialisation alone (content0 vs this)
#
stage_train_ablation() {
  local only="${1:-}"
  local ebnav; ebnav=$(resolve_ebnav)
  say "Content-loss ablation: arms content0 and noinit (replication/chameleon_manifest.tsv)"
  for arm in content0 noinit; do
    local initflag="--object-init"
    [ "$arm" = "noinit" ] && initflag=""
    grep -v '^#' replication/chameleon_manifest.tsv | tail -n +2 |
    while IFS=$'\t' read -r head scenario mode target attribute relation sim_data coverage lambda_content steps eot_crops init_patch; do
      [ -n "${head:-}" ] || continue
      [ -n "$only" ] && [[ "$head" != *"$only"* ]] && continue
      [ -f "$init_patch" ] || die "missing Stage 1 patch $init_patch (run train-universal, or fetch-artifacts)"
      local sid="$scenario"
      if [ "$sim_data" = "SceneGraphBench/data/kitchen_ebnav" ]; then
        sim_data="$ebnav"; sid="${scenario#ebnav_}"
      fi
      if [ -f "output/ablation_patches/$arm/$head/$scenario/$head/${head}_final.pt" ]; then
        echo "--- [$arm] $head / $scenario  (already trained, skipping)"; continue
      fi
      local extra=""
      [ "$attribute" != "-" ] && extra="$extra --attribute '$attribute'"
      [ "$mode" = "relation" ] && extra="$extra --relation '$relation'"
      echo "--- [$arm] $head / $scenario"
      run_ok "'$PY_NSAP' run_attack.py --mode '$mode' --target '$target'$extra \
        --sim-data '$sim_data' --only-scenario '$sid' \
        --init-patch '$init_patch' $initflag --lambda-content 0 \
        --steps '$steps' --batch 1 --eot-crops '$eot_crops' --coverage '$coverage' \
        --lambda-tv 0 --no-chameleon --seed 0 \
        --output-dir 'output/ablation_patches/$arm/$head/$scenario'"
    done
  done
}

stage_eval_ablation() {
  say "Content-loss ablation: flip rate and LPIPS for both arms"
  run "'$PY_NSAP' rung2/outcome_eval.py --arms content0,noinit --masks gt \
    --out output/ablation_outcome.json"
  run "'$PY_NSAP' rung2/stealth_eval.py --coverage 0.5 --arms content0,noinit \
    --out output/ablation_stealth.json"
}

# ---------------------------------------------------------------- model level

stage_eval_model() {
  # --all runs every patch under BOTH mask sources (ground truth and SAM2) into one file,
  # so the SAM2 comparison needs no separate call.
  say "Model level: per-head outcomes, ground-truth and SAM2 masks"
  run "'$PY_NSAP' rung2/outcome_eval.py --all --out output/outcome_eval.json"

  say "Deployment-faithful scoring, detector-driven"
  run "'$PY_EMB' rung2/deploy_eval.py --all-chameleons --out output/deploy_eval.json"

  say "Same, with the scene graph widened to 15 objects (the SGG-literature setting)"
  run "NSAP_TOP_K=15 '$PY_EMB' rung2/deploy_eval.py --all-chameleons --out output/deploy_eval_topk15.json"

  say "Random-patch control: is a trained patch better than noise of the same size?"
  run "NSAP_RANDOM_PATCH=1 '$PY_EMB' rung2/deploy_eval.py --all-chameleons --out output/deploy_eval_random.json"

  say "Stealth: LPIPS, chameleon vs universal, size-matched at coverage 0.5"
  run "'$PY_NSAP' rung2/stealth_eval.py --coverage 0.5 --out output/stealth_eval.json"
}

# ---------------------------------------------------------------- system level

stage_eval_system() {
  # One line per run behind the paper's system-level numbers, with the rep count each
  # actually used. Wall-clock times are what these took here on one RTX 4060 (8 GB)
  # against gpt-4o; the API, not the GPU, is the bottleneck.
  #
  # NOTE the two 0.1 / top_k 1 rows. They are the SAME configuration, run twice. The
  # spread between them is the paper's reproducibility evidence, so both are kept.

  say "Staged attribute and relation tasks, ESCA's released gate (aggr_thres 0.1, top_k 1)  ~4.5 h"
  run "bash rung2/run_staged_variants.sh A_donothing attribute,relation 5 0.1 1"

  say "The same configuration a second time, to measure benchmark drift  ~3.9 h"
  run "RUN_TAG=b bash rung2/run_staged_variants.sh A_donothing attribute,relation 5 0.1 1"

  say "Same tasks at the harness gate (0.4), where the category result was measured  ~7.3 h"
  run "bash rung2/run_staged_variants.sh A_donothing attribute,relation 3 0.4 1"

  say "Same, scene graph widened to 15 objects (the SGG-literature setting)  ~4.8 h"
  run "bash rung2/run_staged_variants.sh A_donothing attribute,relation 5 0.4 15"

  say "Language-only conditions: the planner is blinded to the image, so the graph is load-bearing  ~5.6 h"
  run "bash rung2/run_staged_variants.sh A_donothing attribute,relation 3 0.1 1 base_benign,esca_benign,esca_patch,lang_benign,lang_patch"

  say "Single-pot staging variant: one pot, one differently-typed distractor  ~6.3 h"
  run "bash rung2/run_staged_variants.sh E_single_pot attribute,relation 5 0.4 1"

  say "Category, EB-Nav, ESCA's own navigation evaluator config (gd_only, SGClip skipped)  ~1.8 h"
  run "bash rung2/run_category.sh 0.3 3 1 1"

  say "Category, the same config but with SGClip in the loop  ~2.0 h"
  run "bash rung2/run_category.sh 0.3 3 1 0"

  say "Category, our harness config"
  run "bash rung2/run_category.sh 0.4 3 0 0"
}

# ---------------------------------------------------------------- numbers

stage_numbers() {
  say "Every number the paper cites, recomputed from output/ with its source"
  run "'$PY_NSAP' rung2/paper_numbers.py --out output/NUMBERS.md"
}

# ---------------------------------------------------------------- dispatch

case "${1:-}" in
  setup)            stage_setup ;;
  fetch-artifacts)  stage_fetch_artifacts ;;
  train-universal)  stage_train_universal "${2:-}" ;;
  train-chameleon)  stage_train_chameleon "${2:-}" ;;
  train-ablation)   stage_train_ablation "${2:-}" ;;
  eval-ablation)    stage_eval_ablation ;;
  eval-model)       stage_eval_model ;;
  eval-system)      stage_eval_system ;;
  numbers)          stage_numbers ;;
  all)              stage_setup; stage_train_universal; stage_train_chameleon
                    stage_eval_model; stage_eval_system; stage_numbers ;;
  *) sed -n '2,23p' "$0" | sed 's/^# \?//' ; exit 1 ;;
esac
