# SceneGraphBench

A **reproducible scene-graph dataset / benchmark generator** built on the
[AI2-THOR](https://ai2thor.allenai.org/) embodied simulator.

It renders a robot's-eye view of task-relevant objects and, for every frame, emits
a **pixel-perfect ground-truth scene graph straight from the simulator** —
per-object *category*, *attributes* (open/closed, sliced, on/off, …) and *relations*
(on / in / near). No perception model is involved in generation, so the output is a
**model-agnostic benchmark**: bring your own scene-graph / detector / VLM model, run
it on the frames, and score it against exact ground truth — no human labels, no
self-labeling circularity.

Why a simulator? Ground truth is free and exact, viewpoints match deployment (the
agent navigates *to* each object), and physical states (a fridge open vs closed, an
apple whole vs sliced) are controllable — so you can build balanced, parametric test
sets and regenerate them deterministically from a seed.

---

## Replication — follow these steps exactly

### 1. Get the code and a Python environment
```bash
git clone <your-fork-url> SceneGraphBench
cd SceneGraphBench
conda create -n sgbench python=3.10 -y
conda activate sgbench
pip install -e .          # installs ai2thor, numpy, pillow + the sgbench package
```

### 2. Download the simulator (headless rendering)
AI2-THOR ships a Unity build that is **downloaded automatically on first run** (~0.5 GB).
Headless Linux uses Vulkan (CloudRendering) — no X server, no `sudo` display setup:
```bash
# one-time: make sure a Vulkan loader is present (NVIDIA driver provides the ICD)
vulkaninfo | head -5        # if missing: sudo apt install -y vulkan-tools libvulkan1
nvidia-smi | head -12       # confirm the GPU driver is up
```
> On macOS / a desktop with a display, AI2-THOR works out of the box; you can drop
> the `platform=CloudRendering` argument if you prefer windowed rendering.

### 3. Smoke-test the simulator
```bash
python scripts/00_check_thor.py
# expect: frame (300,300,3) | objects ~70 | instance masks ~55 | OK ...
```
The first run also triggers the Unity-build download. If this prints, you are ready.

### 4. Generate a dataset
```bash
python scripts/01_generate.py --out data/kitchen_pilot --config configs/kitchen_pilot.json
```
This stages each target object (default pilot: **pot, kettle, toaster, coffee
machine**) across all 30 kitchens, navigates the agent to face it from 6 jittered
viewpoints, and writes frames + ground-truth scene graphs. Any config field can be
overridden on the CLI, e.g.:
```bash
# target your own categories, more viewpoints, with material/lighting randomization
python scripts/01_generate.py --out data/custom \
    --categories Fridge Microwave Mug Apple --scenes 1 2 3 \
    --viewpoints 6 --randomize-materials
```

### 5. Inspect the ground truth
```bash
python scripts/02_visualize.py --scenario data/kitchen_pilot/scenarios/scenario_00000
# writes overlay/*.png: green box = target, labels show category [attributes],
# blue lines list relations
```

### 6. Read the format
The dataset is plain PNG + JSON — see [`docs/FORMAT.md`](docs/FORMAT.md) for the full
schema and a snippet to recover per-object masks from the instance maps.

### 7. Use it / extend it
- **Benchmark your model:** run it on `scenarios/*/frames/*.png`, compare predictions
  to `scene_graph.json` (`objects[*].category` / `attributes` / `relations`).
- **Retarget:** edit `sgbench/vocab.py` (`TARGET_CATEGORIES`, `ATTRIBUTE_FLAGS`,
  `ENCLOSING_RECEPTACLES`) — categories, attributes and relations are pure data tables.
- **Determinism:** a fixed `--seed` + scene list reproduces the same dataset.

---

## Package layout
```
sgbench/
  vocab.py        # category/attribute/relation tables + objectType→word mapping
  scene_graph.py  # ground-truth scene-graph extraction from a THOR event
  framing.py      # object-centric camera placement
  generator.py    # the dataset generator (generate(GeneratorConfig))
scripts/
  00_check_thor.py 01_generate.py 02_visualize.py
configs/
  kitchen_pilot.json
docs/
  FORMAT.md
```

## Notes & scope
- Default kitchen scenes are AI2-THOR `FloorPlan1..30`. Living-room / bedroom /
  bathroom plans exist too — just list them in `--scenes`.
- The pilot category set is the cluster a CLIP-based scene-graph model reliably reads
  on AI2-THOR renders; broaden via `--all-targets` and validate per your own model.
- License: TODO. Citation: TODO.
