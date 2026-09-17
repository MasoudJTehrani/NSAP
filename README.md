# NSAP

Adversarial patches against the symbolic grounding of a scene-graph model, and an
analysis of what the embodied agent built on top of it actually reads.

NSAP trains patches that flip individual heads of **SGClip/VINE** (category, attribute,
relation), then injects them into the perception stream of **ESCA**, an embodied agent
that turns the resulting scene graph into text for an MLLM planner. The model-level
attack and the system-level consequence come apart, and most of what this repository
contains is the measurement of that gap.

## Start here

**[REPLICATION.md](REPLICATION.md)** is the entry point: requirements, the two ways to
reproduce (download our artifacts, or retrain), what does and does not replicate, and the
four ESCA configuration values that decide the answer.

```bash
git clone --recurse-submodules <this repo> && cd NSAP
bash replicate.sh setup
bash replicate.sh fetch-artifacts
bash replicate.sh eval-model
```

`DRY_RUN=1 bash replicate.sh <stage>` prints what a stage would run, without running it.

## Layout

| path | what |
|---|---|
| `replicate.sh` | the driver for every stage |
| `replication/` | training recipes, extracted from the saved patches, and the ESCA hook patch |
| `functions/attack/` | patch parameterisation, losses, EOT, anchor selection |
| `functions/eval/` | scene-graph prediction and flip-rate primitives |
| `rung2/` | the embodied harness: injection, benchmarks, evaluation, per-step logging |
| `external/ESCA`, `SceneGraphBench` | pinned submodules |

Trained patches, benchmark logs and result JSON live in `output/`, which is gitignored and
published separately as
[NSAP-artifacts](https://huggingface.co/datasets/MasoudJTehrani/NSAP-artifacts).
`bash replicate.sh fetch-artifacts` puts them back. The dataset is
[NSAP-sim-data](https://huggingface.co/datasets/MasoudJTehrani/NSAP-sim-data), fetched by
`replicate.sh setup`. Both are pinned to a commit sha.
