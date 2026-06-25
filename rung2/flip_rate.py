#!/usr/bin/env python
"""
Robust per-target flip RATES over N viewpoints, MLLM-independent (calls ESCA's
GroundingDINO + SGClip perception directly, with the cleaned detection/vocab).

Two metrics per target, clean vs patched, averaged over N jittered viewpoints:
  GT-box flip : run SGClip on the TARGET's own THOR instance-mask box -> does the
                actual target object stop classifying as the target? (noise-free)
  SG-drop     : run the full GD+SGClip pipeline -> does the target category leave
                the grounded scene graph? (the pipeline-level effect, post-cleanup)

Run: conda run -n embench python rung2/flip_rate.py --use-uni --n 6
"""
import os, sys, json, argparse
sys.path.insert(0, "/home/usi/NSAP/rung2")
import numpy as np, torch
from harness import make_planner, load_patch, ACTIONS, NSAP
from env import Rung2Env
from functions.attack.patch import paste_patch, compute_patch_size
from functions.attack.anchor import anchor_center
from embodiedbench.planner.alfred.esca import convert_gd_boxes_to_pixel

TASKS = {t["id"]: t for t in json.load(open("rung2/tasks.json"))["tasks"]}
CAT = {"Fridge": "fridge", "Pot": "pot", "Toaster": "toaster", "Microwave": "microwave",
       "CoffeeMachine": "coffee machine", "Kettle": "kettle"}


def inject(frame, mask, bbox, patch):
    w, anchor, cov = patch
    phc = (127.5 * (torch.tanh(w) + 1.0)).permute(1, 2, 0)
    ph, pw = compute_patch_size(bbox, cov)
    cx, cy = anchor_center(anchor, bbox, ph, pw)
    out = paste_patch(torch.from_numpy(frame.astype(np.float32)), phc, cx, cy, ph, pw)
    return out.clamp(0, 255).numpy().astype(np.uint8)


def sg_has_target(planner, image_rgb, target_cat):
    """Full GD+SGClip pipeline -> is target_cat among the kept grounded labels?"""
    labels, boxes = planner.predict_gd(image_rgb, [target_cat] + planner.common_objects,
                                       box_threshold=planner.gd_box_threshold,
                                       text_threshold=planner.gd_text_threshold)
    if boxes is None or len(boxes) == 0:
        return False
    h, w = image_rgb.shape[:2]
    bp = convert_gd_boxes_to_pixel(boxes, w, h)
    tl, *_ = planner.predict_laser(image_rgb, bp, masks=None, target_names=[target_cat],
                                   query_names=[target_cat] + planner.common_objects,
                                   threshold=planner.aggr_thres)
    return any(target_cat in l.split("::")[-1].lower() for l in tl)


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def target_box_class(planner, image_rgb, target_cat, gt_bbox):
    """Run the full GD+SGClip pipeline, find the detected box that best OVERLAPS the
    target object's GT bbox, and return whether THAT box reads as the target
    (True/False), or None if nothing overlaps it. Ignores spurious/nearby boxes."""
    labels, boxes = planner.predict_gd(image_rgb, [target_cat] + planner.common_objects,
                                       box_threshold=planner.gd_box_threshold,
                                       text_threshold=planner.gd_text_threshold)
    if boxes is None or len(boxes) == 0:
        return None
    h, w = image_rgb.shape[:2]
    bp = convert_gd_boxes_to_pixel(boxes, w, h)
    planner.predict_laser(image_rgb, bp, masks=None, target_names=[target_cat],
                          query_names=[target_cat], threshold=0.0)
    full = getattr(planner, "last_boxes_full", [])
    if not full:
        return None
    gb = [gt_bbox["x1"], gt_bbox["y1"], gt_bbox["x2"], gt_bbox["y2"]]
    best = max(full, key=lambda x: _iou(x[0], gb))
    if _iou(best[0], gb) < 0.25:
        return None
    return target_cat in best[1].lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="category_fridge,category_toaster,category_pot,category_coffee_machine")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--use-uni", action="store_true")
    args = ap.parse_args()
    planner = make_planner(scene_graph_text=True, language_only=False)
    planner.set_actions(ACTIONS)

    print(f"\n{'task':26}{'TargetBox flip':>16}{'SG-drop':>10}{'(clean baselines)':>28}")
    for tid in args.tasks.split(","):
        task = TASKS[tid]
        cat = CAT[task["target_type"]]
        ppath = f"output/chexp/uni_{tid}/{tid}/{tid}_final.pt" if args.use_uni else task["patch"]
        patch = load_patch(os.path.join(NSAP, ppath))
        env = Rung2Env(task["scene"], resolution=600)
        if task.get("init"):
            env.run_init(task["init"])
        tobj = env._nearest(task["target_type"])
        tid_thor = tobj["objectId"] if tobj else None
        n = tbflip = sgdrop = tb_clean_ok = sg_clean_ok = 0
        for seed in range(args.n):
            frame, mask, bbox = env.view_target(tid_thor, seed=seed * 7 + 1, force_jitter=True)
            if frame is None:
                continue
            patched = inject(frame, mask, bbox, patch)
            tc = target_box_class(planner, frame, cat, bbox)      # box overlapping target
            tp = target_box_class(planner, patched, cat, bbox)
            sc = sg_has_target(planner, frame, cat)
            sp = sg_has_target(planner, patched, cat)
            n += 1
            tb_clean_ok += (tc is True); sg_clean_ok += sc
            tbflip += (tc is True and tp is False)
            sgdrop += (sc and not sp)
        env.stop()
        print(f"{tid:26}{f'{tbflip}/{n}':>16}{f'{sgdrop}/{n}':>10}"
              f"{f'(TBok {tb_clean_ok}/{n} SGok {sg_clean_ok}/{n})':>28}", flush=True)


if __name__ == "__main__":
    main()
