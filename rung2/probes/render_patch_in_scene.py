"""Render a trained patch onto its object in the full camera frame, with no annotation.

`outcome_eval.py --viz` already saves patched frames, but it draws a red box around the
target and a text banner over the top, which is the wrong thing to show when the question
is how well the patch blends in. This writes the same composite with nothing drawn on it,
alongside the clean frame for comparison.

  python rung2/probes/render_patch_in_scene.py --patch <p.pt> --scenario <sid> \
      --sim-data dataset/sim --out output/probes/in_scene --frame 0
"""
import argparse, os, sys

NSAP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, NSAP)
os.chdir(NSAP)
import numpy as np
import torch
from PIL import Image

from functions.attack.train import AttackConfig                              # noqa: E402,F401
from functions.attack.patch import AdversarialPatch                          # noqa: E402
from functions.attack.sim_data import SimAttackDataset, explode_to_frames    # noqa: E402
from functions.eval._common import _patch_object_across_frames, _cat_ok      # noqa: E402
from functions.attack.data import largest_object_id                          # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def to_img(frame):
    f = frame.detach().cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
    return Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--sim-data", default="dataset/sim")
    ap.add_argument("--out", default="output/probes/in_scene")
    ap.add_argument("--frame", type=int, default=0, help="which viewpoint of the scenario")
    ap.add_argument("--tag", default="", help="basename for the written files")
    a = ap.parse_args()

    ck = torch.load(a.patch, map_location=DEV, weights_only=False)
    cfg = ck["config"]; anchor = ck["anchor"]
    cfg.sim_data_dir = a.sim_data
    patch = AdversarialPatch(ck["w"].shape[1], ck["w"].shape[2], torch.device(DEV))
    patch.w.data.copy_(ck["w"]); patch.eval()
    mode = cfg.attack_mode

    ds = SimAttackDataset(
        a.sim_data,
        target_category=cfg.target_category if mode != "relation" else None,
        target_attribute=cfg.target_attribute if mode == "attribute" else None,
        target_relation=cfg.target_relation if mode == "relation" else None,
        split=None, max_clips=getattr(cfg, "max_clips", 0),
        only_scenarios=[a.scenario], data_source=getattr(cfg, "data_source", ""))
    items = [fr for it in [ds[i] for i in range(len(ds))] for fr in explode_to_frames(it)]
    if not items:
        raise SystemExit(f"no frames for scenario {a.scenario}")
    it = items[min(a.frame, len(items) - 1)]

    masks, bboxes = it["masks"], it["bboxes"]
    gt = it.get("gt_categories") or {}
    cand = set(masks.get(0, {}).keys())
    if cfg.target_category and gt:
        keep = {o for o in cand if _cat_ok(gt.get(o, ""), cfg.target_category)}
        cand = keep or cand
    oid = largest_object_id(masks, cand)
    if oid is None:
        raise SystemExit("no target object in this frame")

    patched = _patch_object_across_frames(it["frames"], bboxes, patch, anchor, cfg,
                                          torch.device(DEV), oid)
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or f"{mode}_{a.scenario}"
    to_img(it["frames"][0]).save(os.path.join(a.out, f"{tag}_clean.png"))
    to_img(patched[0]).save(os.path.join(a.out, f"{tag}_patched.png"))
    # the object's box, so a figure can crop around it without guessing
    bb = (bboxes.get(0) or {}).get(oid)
    if bb:
        with open(os.path.join(a.out, f"{tag}_box.txt"), "w") as f:
            f.write(f"{bb['x1']} {bb['y1']} {bb['x2']} {bb['y2']}\n")
    print(f"wrote {a.out}/{tag}_clean.png and _patched.png"
          + (f"  box={bb}" if bb else ""))


if __name__ == "__main__":
    main()
