#!/usr/bin/env python
"""Probe SGClip's classification of a target across preprocessing variants, to test the
hypothesis: cropping to the GT bbox ZOOMS the object -> misread (pot->toaster, toaster->
coffee machine), whereas whitening the background at TRUE scale classifies it correctly.

SGClip's category head does  extract_single_object(mask)  THEN  crop_to_bbox(box)  --
whitening is driven by the MASK, the zoom by the BOX, and predict_laser takes them
independently. So we sweep, for each task's target (faced, isolated, CLEAN unless --patch):

  (a) tight GT-bbox  + rect mask   -> whiten-outside-rect, crop-to-bbox   (the misreading crop)
  (b) FULL frame     + real mask   -> whiten-outside-object, NO zoom      (the idea)
  (c) tight GT-bbox  + real mask   -> isolate: mask helps, or is it the crop?
  (d) GroundingDINO  + rect mask   -> what the current gate / planner path sees

Reports SGClip's argmax + conf for the target box in each. Run with the embench python:
  <miniconda>/envs/embench/bin/python rung2/gate_probe.py --tasks toast_cat,pot_cat
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from harness import (make_planner, load_patch, ACTIONS, NSAP, THOR2VINE,
                     _target_bbox, _paste_on_target, convert_gd_boxes_to_pixel)
from env import Rung2Env


def _laser_one(planner, frame, boxes, masks, vine_cat, thr):
    """predict_laser then return the highest-conf (cat, conf) among the input boxes."""
    planner.predict_laser(image=frame, boxes=boxes, masks=masks,
                          target_names=[vine_cat], threshold=thr)
    lb = list(getattr(planner, "last_boxes_full", []))
    if not lb:
        return None, 0.0, lb
    _b, cat, conf = max(lb, key=lambda x: x[2])
    return cat, conf, lb


def _whiten_outside_bbox(frame, bbox, alpha=1.0):
    """Blend everything OUTSIDE the target's bbox toward white (alpha=1 => hard white),
    leaving the bbox region untouched. Removes other objects from VINE's view at TRUE scale."""
    out = frame.astype(np.float32)
    keep = np.zeros(frame.shape[:2], dtype=bool)
    keep[bbox["y1"]:bbox["y2"] + 1, bbox["x1"]:bbox["x2"] + 1] = True
    white = np.full_like(out, 255.0)
    bg = ~keep
    out[bg] = out[bg] * (1.0 - alpha) + white[bg] * alpha
    return out.astype(np.uint8)


def _gd_ground(planner, frame, vine_cat, thr):
    """Full planner grounding: GroundingDINO detect over the fixed vocab, then SGClip
    classify each box. Returns [(cat, conf), ...] -- i.e. the object list the SG would show,
    so we can count how many boxes come back as the target (false positives vs the real one)."""
    W, H = frame.shape[1], frame.shape[0]
    gd_vocab = [c for c in (list(planner.classify_vocab) + list(planner.common_objects))
                if c != "other"]
    _lab, gboxes = planner.predict_gd(frame, gd_vocab,
                                      box_threshold=planner.gd_box_threshold,
                                      text_threshold=planner.gd_text_threshold)
    if gboxes is None or len(gboxes) == 0:
        return "no GD detections"
    gpix = convert_gd_boxes_to_pixel(gboxes, W, H)
    _c, _cf, lb = _laser_one(planner, frame, gpix, None, vine_cat, thr)
    return [(c, round(float(cf), 3)) for _b, c, cf in lb]


def probe_task(planner, task, thr, use_patch):
    env = Rung2Env(task["scene"], resolution=600)
    if task.get("init"):
        env.run_init(task["init"])
    tid = (env._nearest(task["target_type"]) or {}).get("objectId")
    if tid:
        env.isolate_target(tid)
    frame, mask, bbox = env.view_target(tid)
    env.stop()
    if frame is None:
        return None
    H, W = frame.shape[:2]
    vine_cat = THOR2VINE.get(task["target_type"], task["target_type"].lower())
    if use_patch:
        w_a, anch, cov = load_patch(os.path.join(NSAP, task["patch"]))
        frame, _ = _paste_on_target(frame, bbox, (w_a, anch, cov))

    box_tight = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
    box_full  = [0, 0, W - 1, H - 1]
    m = mask.astype(np.uint8)

    out = {"target": vine_cat, "bbox": box_tight, "frame": f"{W}x{H}", "variants": {}}
    for name, boxes, masks in [
        ("(a) tight bbox + rect mask", [box_tight], None),
        ("(b) FULL frame + real mask", [box_full],  [m]),
        ("(c) tight bbox + real mask", [box_tight], [m]),
    ]:
        cat, conf, _ = _laser_one(planner, frame, boxes, masks, vine_cat, thr)
        out["variants"][name] = (cat, round(float(conf), 3))

    # THE REAL TEST of the false-positive problem: full GD->SGClip grounding on the
    # NORMAL frame (d) vs the WHITENED-outside-bbox frame (e). Count how many boxes come
    # back as the target when other objects are visible vs whitened out.
    out["variants"]["(d) GD grounding, NORMAL frame"] = _gd_ground(planner, frame, vine_cat, thr)
    frame_w = _whiten_outside_bbox(frame, bbox, alpha=1.0)
    out["variants"]["(e) GD grounding, WHITENED bg"] = _gd_ground(planner, frame_w, vine_cat, thr)
    os.makedirs("output/rung2/probe", exist_ok=True)
    Image.fromarray(frame_w).save(f"output/rung2/probe/{task['id']}_whitened.png")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", default="rung2/gap_tasks_v2.json")
    ap.add_argument("--tasks", default="", help="comma list of task ids (default: all)")
    ap.add_argument("--patch", action="store_true", help="paste the trained patch first (default: CLEAN)")
    args = ap.parse_args()
    TASKS = {t["id"]: t for t in json.load(open(args.tasks_file))["tasks"]}
    planner = make_planner(scene_graph_text=True, language_only=False)
    planner.set_actions(ACTIONS)
    thr = getattr(planner, "aggr_thres", 0.4)
    ids = args.tasks.split(",") if args.tasks else list(TASKS)
    for tid in ids:
        print(f"\n################ {tid}  (patch={args.patch}) ################", flush=True)
        r = probe_task(planner, TASKS[tid], thr, args.patch)
        if r is None:
            print("  (target not visible)"); continue
        print(f"  target VINE name = '{r['target']}'   bbox={r['bbox']}   frame={r['frame']}")
        for name, val in r["variants"].items():
            if isinstance(val, tuple):
                flag = "  <-- CORRECT" if val[0] == r["target"] else "  (misread)"
                print(f"    {name:34s}: {val}{flag}")
            elif isinstance(val, list):
                n_tgt = sum(1 for c, cf in val if c == r["target"] and cf >= 0.4)
                print(f"    {name:34s}: {len(val)} boxes, {n_tgt} kept as '{r['target']}'>=0.4  ->  {val}")
            else:
                print(f"    {name:34s}: {val}")


if __name__ == "__main__":
    main()
