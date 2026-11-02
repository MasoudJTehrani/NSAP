#!/usr/bin/env bash
# Two sessions with frames pinned and the symbolic specification AUTHORED from the task
# (rung2/author_specs.py) instead of frozen from the MLLM's first draw, which left 3 of
# 10 attribute episodes grounding the wrong concept. Same frame cache as pin1/pin2.
set -euo pipefail
cd /home/usi/NSAP
export ESCA_ROOT=/home/usi/ESCA
export NSAP_FRAME_CACHE=/home/usi/NSAP/output/frame_cache
export NSAP_VOCAB_CACHE=/home/usi/NSAP/output/vocab_cache_authored
export NSAP_PAUSE_PHYSICS=1
export NSAP_PIN_VOCAB=1
for tag in spec1 spec2; do
  echo "########## SESSION $tag  $(date) ##########"
  RUN_TAG=$tag bash rung2/run_staged_variants.sh A_donothing attribute,relation 3 0.1 1
  echo "########## SESSION $tag DONE  $(date) ##########"
done
echo "ALL_SPEC_SESSIONS_COMPLETE"
