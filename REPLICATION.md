# Replicating NSAP

Everything runs through one driver:

```bash
bash replicate.sh setup            # weights, detector, dataset, ESCA hook
bash replicate.sh fetch-artifacts  # our trained patches and run logs
bash replicate.sh train-universal  # Stage 1: 5 universal patches
bash replicate.sh train-chameleon  # Stage 2: 30 per-scenario patches
bash replicate.sh eval-model       # model-level analysis
bash replicate.sh eval-system      # embodied benchmarks
```

`DRY_RUN=1` in front of any of these prints the exact commands without running them.
Read that output first: it is the shortest accurate description of what each stage does.

## Requirements

| | |
|---|---|
| GPU | one CUDA card, 8 GB is enough (developed on an RTX 4060) |
| Disk | about 25 GB for weights and the dataset, plus 12 GB if you keep every frame |
| Envs | two conda envs, `NSAP` (training and model-level eval) and `embench` (ESCA and the benchmarks) |
| API | `OPENAI_API_KEY` for `eval-system` only, which calls gpt-4o |

`NSAP` comes from `environment.yml`. `embench` comes from ESCA's own `external/ESCA/install.sh`.
If your envs live elsewhere or are named differently, set `CONDA_ROOT`, `ENV_NSAP`, `ENV_EMBENCH`.

`setup` needs a display for AI2-THOR when it generates the EB-Nav scenes. On a headless
machine run it as `xvfb-run -a bash replicate.sh setup`.

## Two ways in

**Use our artifacts.** `fetch-artifacts` downloads the 35 trained patches and the nine
benchmark run directories into `output/`, from
[MasoudJTehrani/NSAP-artifacts](https://huggingface.co/datasets/MasoudJTehrani/NSAP-artifacts)
at a pinned revision. From there `eval-model` reproduces every
model-level number in the paper, and `rung2/analyze_steps.py <run-dir>` reproduces the
per-step analysis without spending a cent on the API. This is the path we recommend.

**Retrain from scratch.** `train-universal` then `train-chameleon`. The recipes are not
buried in the scripts: `replication/universal_manifest.tsv` and
`replication/chameleon_manifest.tsv` were generated from the `AttackConfig` stored inside
each saved patch, so they are a record of what actually ran rather than a description of it.
Stage 2 warm-starts from Stage 1, so the order matters.

## What replicates, and what does not

This matters more than usual here, because one of the paper's findings is about
reproducibility itself.

**Replicates.** Everything at the model level. The patches are trained with a fixed seed,
and the evaluation is deterministic given a patch. Flip rates, the outcome taxonomy, the
isolation split, the deployment-faithful scoring, the random-patch control and the LPIPS
stealth numbers should all come back the same.

**Does not replicate exactly.** The embodied benchmark. gpt-4o is not deterministic at the
temperature ESCA uses, and the episode outcome is a long chain of sampled decisions. We
measured this directly: across six sessions with an identical configuration, the unattacked
`base_benign` baseline ranged from 0% to 44% success. Two runs of one configuration are
therefore kept in `eval-system` on purpose, and the published logs keep both.

Read the system-level results as the paper reads them: as an analysis of what happens to the
scene graph, which is recorded exactly in `steps.jsonl` and is not subject to that noise, and
not as a success-rate delta.

## Configuration values that decide the result

Four settings change the answer more than anything in the attack does. All four are
ESCA's, not ours, and all four are overridable from the environment.

| setting | ESCA's value | what it does |
|---|---|---|
| `top_k` | 1 | how many objects clear the gate into the scene graph. At 1, the graph holds a single object, so relations have nothing to relate and an isolation score has nothing to isolate. |
| `aggr_thres` | 0.1 planner, 0.3 nav evaluator, 0.75 ALFRED | the aggregation gate. Three different released defaults for the same model. |
| `gd_only` | `True` in the nav evaluator | when set, `predict_laser` is skipped entirely, so SGClip never runs and the scene graph is GroundingDINO's category output alone. |
| `NSAP_LASER_ATTR_REL` | 0 | our addition. With it off, attributes and relations never reach the scene graph text at all. |

`rung2/run_category.sh` and `rung2/run_staged_variants.sh` take the first three as positional
arguments and refuse to guess. That is deliberate: an unset `NSAP_AGGR_THRES` falls back to
0.4, which is the harness value, not any of ESCA's.

## Traps

Things that cost us time and will silently give you wrong numbers.

- **`predict_laser` is case-sensitive.** It keeps a box only when the name is in
  `target_names`, and those names are lowercase. Passing `'Pot'` or `'Television'` from a
  config grounds nothing, and the relation head then reports zero flips with no error.
- **The scene graph is one object deep at ESCA's defaults.** Before reading any relation
  result, check how many objects actually cleared the gate.
- **Flip definitions are not interchangeable.** "The target pair changed" and "no edge
  survives" give very different rates on the same data. `rung2/deploy_eval.py` uses the
  pair-level definition, matching the original evaluation.
- **Patch size confounds stealth.** Coverage is applied per side, so coverage 0.5 pastes
  2.78 times the area of coverage 0.3. Compare LPIPS only at matched coverage, which is what
  `stealth_eval.py --coverage 0.5` enforces.
- **Patches are scene-matched.** A chameleon is overfit to one scenario. Running it in
  another FloorPlan is not a weaker attack, it is a different experiment.
- **Check `frames_with_patch` before believing a row.** If it is 0 the target was never
  visible, the patch never injected, and whatever happened that episode is not attributable
  to the attack.

## Published data

Both are pinned to a commit sha in `functions/download_model.py`, so a later change to
either repo cannot silently alter what you download. Override with `$NSAP_DATA_REVISION`
and `$NSAP_ARTIFACTS_REVISION`.

| repo | what | pulled by |
|---|---|---|
| [NSAP-sim-data](https://huggingface.co/datasets/MasoudJTehrani/NSAP-sim-data) | 202 AI2-THOR scenarios with ground-truth scene graphs, plus the 12 EB-Nav scene-matched ones | `replicate.sh setup` |
| [NSAP-artifacts](https://huggingface.co/datasets/MasoudJTehrani/NSAP-artifacts) | 35 trained patches, 9 benchmark run directories, model-level results | `replicate.sh fetch-artifacts` |

## Layout

```
replicate.sh                        the driver
replication/universal_manifest.tsv  Stage 1 recipes, extracted from the saved patches
replication/chameleon_manifest.tsv  Stage 2 recipes, same provenance
replication/esca_nsap_debug_hooks.patch   read-only hooks applied to the ESCA submodule
rung2/                              benchmark harness, evaluation and analysis
functions/                          patch training, losses, evaluation primitives
external/ESCA, SceneGraphBench      submodules, pinned
```

Output lands in `output/`, which is gitignored. `output/rung2/<run>/<rep>/<condition>/<episode>/steps.jsonl`
is the file to read when you want to know what the planner actually saw and did.
