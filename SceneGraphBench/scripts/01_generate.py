#!/usr/bin/env python
"""
Step 3: generate a scene-graph dataset.

  python scripts/01_generate.py --out data/kitchen_pilot --config configs/kitchen_pilot.json

Any CLI flag overrides the corresponding field in --config.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sgbench.generator import GeneratorConfig, generate
from sgbench.vocab import PILOT_CATEGORIES, TARGET_CATEGORIES


def main():
    ap = argparse.ArgumentParser(description="Generate an AI2-THOR scene-graph dataset")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--config", help="JSON config; CLI flags below override its fields")
    ap.add_argument("--scenes", type=int, nargs="+")
    ap.add_argument("--viewpoints", type=int)
    ap.add_argument("--resolution", type=int)
    ap.add_argument("--min-area", type=int)
    ap.add_argument("--max-targets", type=int, help="targets per scene (0 = all)")
    ap.add_argument("--categories", nargs="+", help="explicit AI2-THOR objectTypes to target")
    ap.add_argument("--all-targets", action="store_true", help="use the full TARGET_CATEGORIES set")
    ap.add_argument("--randomize-materials", action="store_true", help="randomize materials/lighting per scene")
    ap.add_argument("--no-randomize-states", action="store_true", help="keep objects in their default states")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()

    cfg = json.load(open(a.config)) if a.config else {}
    cfg["out_dir"] = a.out
    for key, val in (("scenes", a.scenes), ("viewpoints", a.viewpoints),
                     ("resolution", a.resolution), ("min_area", a.min_area),
                     ("max_targets_per_scene", a.max_targets), ("seed", a.seed)):
        if val is not None:
            cfg[key] = val
    if a.randomize_materials:
        cfg["randomize_materials"] = True
    if a.no_randomize_states:
        cfg["randomize_states"] = False

    if a.categories:
        cfg["target_categories"] = set(a.categories)
    elif a.all_targets:
        cfg["target_categories"] = set(TARGET_CATEGORIES)
    elif "target_categories" in cfg:
        cfg["target_categories"] = set(cfg["target_categories"])
    else:
        cfg["target_categories"] = set(PILOT_CATEGORIES)

    generate(GeneratorConfig(**cfg))


if __name__ == "__main__":
    main()
