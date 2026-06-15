# Dataset format

A generated dataset is a self-describing directory. Everything is plain PNG + JSON
so it is model- and framework-agnostic.

```
<dataset>/
├── manifest.json          # dataset-level index
├── vocab.json             # category / attribute / relation vocabularies used
└── scenarios/
    ├── scenario_00000/
    │   ├── meta.json          # scenario summary
    │   ├── scene_graph.json   # per-frame ground-truth scene graph
    │   ├── frames/000.png …   # robot's-eye RGB frames (the model input)
    │   └── instance/000.png … # instance-segmentation maps (color-coded)
    └── scenario_00001/ …
```

A **scenario** is one target object observed from a few viewpoints — i.e. a short
clip. One target = one scenario.

## `manifest.json`
```json
{
  "version": "0.1.0",
  "config": { ...the GeneratorConfig used... },
  "scenes": [1, 2, 3],
  "scenarios": [
    {"scenario_id": "scenario_00000", "scene": "FloorPlan1",
     "target_category": "pot", "n_frames": 4}
  ]
}
```

## `scenario_XXXXX/scene_graph.json`
```json
{
  "scenario_id": "scenario_00000",
  "scene": "FloorPlan1",
  "target": {"id": 12, "category": "pot", "thor_id": "Pot|+01.2|...", "thor_type": "Pot"},
  "frames": [
    {
      "frame": "frames/000.png",
      "instance_map": "instance/000.png",
      "objects": [
        {"id": 12, "category": "pot", "thor_id": "Pot|...", "color": [r, g, b],
         "bbox": [x1, y1, x2, y2], "attributes": ["empty"]},
        {"id": 7,  "category": "countertop", "color": [r, g, b],
         "bbox": [x1, y1, x2, y2], "attributes": []}
      ],
      "relations": [
        {"subject": 12, "predicate": "on", "object": 7}
      ]
    }
  ]
}
```

Field notes:
- **`id`** — integer object id, stable across all frames of a scenario (so relation
  endpoints are consistent over time).
- **`category`** — ground-truth class as a natural-language word (e.g. `pot`,
  `coffee machine`), mapped from AI2-THOR's `objectType`.
- **`attributes`** — active object states as words: `open`/`closed`, `sliced`,
  `on`/`off`, `cooked`, `dirty`/`clean`, `filled`, `broken`. Empty list = none active.
- **`bbox`** — `[x1, y1, x2, y2]` in pixels, tight bound of the object's mask.
- **`color`** — the object's RGB color in `instance_map`. Recover the exact mask:
  ```python
  import numpy as np
  from PIL import Image
  inst = np.array(Image.open("instance/000.png"))
  mask = np.all(inst == np.array(color), axis=-1)   # bool [H, W]
  ```
- **`relations`** — `subject`/`object` are object `id`s; `predicate` ∈ `on`, `in`,
  `near`. Emitted only when both endpoints are visible in that frame.
- Only objects that are **visible and ≥ `min_area` pixels** are listed — the regime
  an embodied agent actually perceives.

## How to use it as a benchmark
Run your scene-graph / detector / classifier on each `frames/*.png`, then compare
your predictions to `objects[*].category` / `attributes` / `relations`. Because the
ground truth is the simulator's own state, scores are exact — no human labels, no
self-labeling circularity.
