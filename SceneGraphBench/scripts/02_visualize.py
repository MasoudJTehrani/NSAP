#!/usr/bin/env python
"""
Step 4: draw the ground-truth scene graph onto a scenario's frames for inspection.

  python scripts/02_visualize.py --scenario data/kitchen_pilot/scenarios/scenario_00000
"""

import os
import sys
import json
import argparse

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser(description="Overlay GT scene graph on a scenario's frames")
    ap.add_argument("--scenario", required=True, help="path to a scenario_XXXXX folder")
    ap.add_argument("--out", default=None, help="output dir (default: <scenario>/overlay)")
    a = ap.parse_args()

    sg = json.load(open(os.path.join(a.scenario, "scene_graph.json")))
    out = a.out or os.path.join(a.scenario, "overlay")
    os.makedirs(out, exist_ok=True)
    target_id = sg["target"]["id"]

    for fr in sg["frames"]:
        img = Image.open(os.path.join(a.scenario, fr["frame"])).convert("RGB")
        d = ImageDraw.Draw(img)
        for o in fr["objects"]:
            is_target = (o["id"] == target_id)
            col = (30, 180, 30) if is_target else (220, 150, 30)
            x1, y1, x2, y2 = o["bbox"]
            d.rectangle([x1, y1, x2, y2], outline=col, width=3 if is_target else 1)
            label = o["category"] + (f" [{','.join(o['attributes'])}]" if o["attributes"] else "")
            d.text((x1 + 2, max(0, y1 - 10)), label, fill=col)
        ry = 2
        for r in fr["relations"][:8]:
            d.text((2, ry), f'{r["subject"]} {r["predicate"]} {r["object"]}', fill=(0, 200, 255))
            ry += 11
        img.save(os.path.join(out, os.path.basename(fr["frame"])))

    print(f"Saved {len(sg['frames'])} overlays → {out}/")


if __name__ == "__main__":
    main()
