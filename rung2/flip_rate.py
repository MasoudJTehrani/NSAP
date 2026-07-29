#!/usr/bin/env python
"""
Per-target, per-ATTACKED-HEAD flip rate for the MIM patch — MLLM-independent (calls
ESCA's GroundingDINO + SGClip directly, no planner episode). For each task, over N
jittered viewpoints of the target, it measures whether the head the task attacks
changes its label ON THE TARGET, clean -> patched:

  mode=category   -> SGClip category head on the target's own box
                     (box picked by overlap with the target's GT bbox: instance-specific)
  mode=attribute  -> SGClip unary  head on the target  (e.g. open -> clean/closed)
  mode=relation   -> SGClip binary head on the target  (e.g. on   -> near)

Output per task: HEAD-flip rate + the clean baseline (was the head correct/grounded at
all cleanly), so a low flip rate is not confused with a plain detection failure.

This replaces the old category-only TargetBox-flip + whole-SG "SG-drop" pair: SG-drop
("is the category anywhere in the SG") was confounded by spurious boxes / multiple
instances, and only ever looked at the category head. Behavioral task-success is
measured separately by harness.py.

Run (embench env; conda run may fail -> use the env python directly):
  <miniconda>/envs/embench/bin/python rung2/flip_rate.py --tasks-file rung2/gap_tasks_v2.json --n 6
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from harness import make_planner, load_patch, ACTIONS, NSAP
from env import Rung2Env
from functions.attack.patch import paste_patch, compute_patch_size
from functions.attack.anchor import anchor_center
from embodiedbench.planner.alfred.esca import convert_gd_boxes_to_pixel

CAT = {"Fridge": "fridge", "Pot": "pot", "Toaster": "toaster", "Microwave": "microwave",
       "CoffeeMachine": "coffee machine", "Kettle": "kettle"}


def inject(frame, mask, bbox, patch):
    """Paste the patch onto the target's bbox (same math as harness.render_inject)."""
    w, anchor, cov = patch
    phc = (127.5 * (torch.tanh(w) + 1.0)).permute(1, 2, 0)
    ph, pw = compute_patch_size(bbox, cov)
    cx, cy = anchor_center(anchor, bbox, ph, pw)
    out = paste_patch(torch.from_numpy(frame.astype(np.float32)), phc, cx, cy, ph, pw)
    return out.clamp(0, 255).numpy().astype(np.uint8)


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ground(planner, image, target_cat):
    """Run the real GD+SGClip pipeline (all three heads) on `image`. Returns
      boxes_full = [(bbox_xyxy, category, conf), ...]      (SGClip category head, per box)
      obj_attrs  = [[attr, obj_name], ...]                 (SGClip unary head)
      obj_rels   = [[subj_name, rel, obj_name], ...]       (SGClip binary head)
    Empty structures if GD detects nothing."""
    labels, boxes = planner.predict_gd(image, [target_cat] + planner.common_objects,
                                       box_threshold=planner.gd_box_threshold,
                                       text_threshold=planner.gd_text_threshold)
    if boxes is None or len(boxes) == 0:
        planner.last_boxes_full = []
        return [], [], []
    h, w = image.shape[:2]
    bp = convert_gd_boxes_to_pixel(boxes, w, h)
    out = planner.predict_laser(image, bp, masks=None,
                                target_names=[target_cat],
                                query_names=[target_cat] + planner.common_objects,
                                threshold=0.0,
                                query_attributes=list(getattr(planner, "attr_vocab", []) or []),
                                query_relation_kws=list(getattr(planner, "rel_vocab", []) or []))
    obj_attrs = out[3] if len(out) > 3 else []
    obj_rels  = out[4] if len(out) > 4 else []
    return getattr(planner, "last_boxes_full", []) or [], obj_attrs, obj_rels


def target_head_label(mode, target_cat, gt_bbox, boxes_full, obj_attrs, obj_rels):
    """The ATTACKED head's label ON THE TARGET, or None if the target isn't grounded
    on that head. category is instance-specific (GT-bbox overlap); attribute/relation
    are keyed by the target's category name (single-instance assumption — SGClip's
    unary/binary outputs are per category name, not per box)."""
    if mode == "category":
        if not boxes_full:
            return None
        gb = [gt_bbox["x1"], gt_bbox["y1"], gt_bbox["x2"], gt_bbox["y2"]]
        best = max(boxes_full, key=lambda x: _iou(x[0], gb))
        return best[1].lower() if _iou(best[0], gb) >= 0.25 else None
    if mode == "attribute":
        hits = [a[0].lower() for a in obj_attrs if target_cat in str(a[1]).lower()]
        return hits[0] if hits else None
    if mode == "relation":
        hits = [r[1].lower() for r in obj_rels if target_cat in str(r[0]).lower()]
        return hits[0] if hits else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", default="rung2/gap_tasks_v2.json")
    ap.add_argument("--tasks", default="", help="comma list of task ids (default: all in the file)")
    ap.add_argument("--n", type=int, default=6, help="jittered viewpoints per target")
    ap.add_argument("--use-uni", action="store_true",
                    help="use the loud universal patch (output/chexp/uni_<id>/...) instead of the task's patch")
    args = ap.parse_args()
    tasks = {t["id"]: t for t in json.load(open(args.tasks_file))["tasks"]}
    ids = args.tasks.split(",") if args.tasks else list(tasks)

    planner = make_planner(scene_graph_text=True, language_only=False)
    planner.set_actions(ACTIONS)

    print(f"\n{'task':22}{'mode':11}{'attacked-head flip':>20}{'clean-grounded':>16}")
    for tid in ids:
        task = tasks[tid]
        mode = task["mode"]
        cat = CAT.get(task["target_type"], task["target_type"].lower())
        ppath = f"output/chexp/uni_{tid}/{tid}/{tid}_final.pt" if args.use_uni else task["patch"]
        patch = load_patch(os.path.join(NSAP, ppath))

        env = Rung2Env(task["scene"], resolution=600)
        if task.get("init"):
            env.run_init(task["init"])
        tobj = env._nearest(task["target_type"])
        tid_thor = tobj["objectId"] if tobj else None

        n = flips = clean_ok = 0
        for seed in range(args.n):
            frame, mask, bbox = env.view_target(tid_thor, seed=seed * 7 + 1, force_jitter=True)
            if frame is None:
                continue
            patched = inject(frame, mask, bbox, patch)
            lc = target_head_label(mode, cat, bbox, *ground(planner, frame, cat))    # clean label
            lp = target_head_label(mode, cat, bbox, *ground(planner, patched, cat))  # patched label
            n += 1
            # baseline: the target was correctly grounded on the attacked head, clean.
            # category demands it read as the target; attribute/relation just need a label.
            base_ok = (lc == cat) if mode == "category" else (lc is not None)
            clean_ok += base_ok
            # flip: given a clean baseline, the label changed (or disappeared) under the patch.
            flips += (base_ok and lp != lc)
        env.stop()
        print(f"{tid:22}{mode:11}{f'{flips}/{n}':>20}{f'{clean_ok}/{n}':>16}", flush=True)


if __name__ == "__main__":
    main()
