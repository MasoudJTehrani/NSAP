#!/usr/bin/env bash
# Two sessions with BOTH determinism fixes on: frames replayed from a pose cache with
# physics frozen, and the MLLM's symbolic specification pinned per episode. Shares the
# frame cache with the fix1/fix2 runs so the only difference from those is the vocabulary.
set -euo pipefail
cd /home/usi/NSAP
export ESCA_ROOT=/home/usi/ESCA
export NSAP_FRAME_CACHE=/home/usi/NSAP/output/frame_cache
export NSAP_VOCAB_CACHE=/home/usi/NSAP/output/vocab_cache
export NSAP_PAUSE_PHYSICS=1
export NSAP_PIN_VOCAB=1
for tag in pin1 pin2; do
  echo "########## SESSION $tag  $(date) ##########"
  RUN_TAG=$tag bash rung2/run_staged_variants.sh A_donothing attribute,relation 3 0.1 1
  echo "########## SESSION $tag DONE  $(date) ##########"
done
echo "ALL_PINNED_SESSIONS_COMPLETE"
