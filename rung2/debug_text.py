#!/usr/bin/env python
"""
Debug: show ESCA's grounded scene-graph TEXT, clean vs patched, on the SAME frame,
across many scenarios (category/attribute/relation). After Fix C the grounded text
should be CLEAN (one entry per object). We want to see, per scenario:
  - the MLLM's own object list (its vision), clean vs patched
  - ESCA's grounded objects / attributes / relations, clean vs patched
Run (embench python directly; conda run may fail if CONDA_EXE is unset):
  <miniconda>/envs/embench/bin/python rung2/debug_text.py --tasks-file rung2/gap_tasks_v2.json
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
from PIL import Image
from harness import make_planner, load_patch, ACTIONS, NSAP
from env import Rung2Env
from functions.attack.patch import paste_patch, compute_patch_size
from functions.attack.anchor import anchor_center


def _draw_boxes(src_png, boxes_full, out_png):
    """Draw SGClip's ACTUAL per-box (category, conf) on the real frame — the truthful
    annotated view of what VINE classified. ESCA's own obs_bbox.png overlay is drawn on
    a clean/cached frame with GD's labels, so it disagrees with the real grounding; this
    draws last_boxes_full (pixel bbox + SGClip category + conf) on the patched frame."""
    from PIL import ImageDraw
    img = Image.open(src_png).convert("RGB")
    d = ImageDraw.Draw(img)
    for bbox, cat, conf in boxes_full:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        d.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=2)
        d.text((x1 + 2, max(0, y1 - 10)), f"{cat} {conf:.2f}", fill=(255, 60, 60))
    img.save(out_png)


def run_task(planner, task, isolate=True):
    env = Rung2Env(task["scene"], resolution=600)
    if task.get("init"):
        env.run_init(task["init"])
    tid = (env._nearest(task["target_type"]) or {}).get("objectId")
    if isolate and tid:   # match the harness: strip confusable objects before viewing
        n = env.isolate_target(tid)
        print(f"  [isolate] removed {n} movable objects (kept {task['target_type']} + structure)")
    frame, mask, bbox = env.view_target(tid)
    env.stop()
    if frame is None:
        return None
    outdir = f"output/rung2/text_debug/{task['id']}"
    os.makedirs(outdir, exist_ok=True)
    Image.fromarray(frame).save(f"{outdir}/clean.png")
    w, anchor, cov = load_patch(os.path.join(NSAP, task["patch"]))
    patch_hwc = (127.5 * (torch.tanh(w) + 1.0)).permute(1, 2, 0)
    ph, pw = compute_patch_size(bbox, cov)
    cx, cy = anchor_center(anchor, bbox, ph, pw)
    patched = paste_patch(torch.from_numpy(frame.astype(np.float32)), patch_hwc, cx, cy, ph, pw)
    patched = patched.clamp(0, 255).numpy().astype(np.uint8)
    Image.fromarray(patched).save(f"{outdir}/patched.png")

    def grab(img_rel):
        planner.reset()
        _action, out_str = planner.act(img_rel, task["instruction"])
        # Draw SGClip's REAL per-box classification on the actual frame -> <img>_planner.png
        # (truthful; ESCA's own obs_bbox.png overlay is drawn on a clean/cached frame).
        _draw_boxes(img_rel, getattr(planner, "last_boxes_full", []),
                    img_rel.replace(".png", "_planner.png"))
        try:                       # out_str is the full MLLM plan JSON (esca.py act() line 2635)
            plan = json.loads(out_str)
        except Exception:
            plan = out_str          # keep raw string if it didn't parse
        sg = getattr(planner, "last_sg_info", {}) or {}
        return {"mllm_objects": sg.get("current_objects"),
                "grounded_text": getattr(planner, "last_sg_text", "") or "",
                # full planner output: visual_state_description, reasoning_and_reflection,
                # language_plan, executable_plan (the {action_id, action_name} steps), scene_graph
                "plan": plan,
                "n_boxes": getattr(planner, "last_n_boxes", 0),
                "boxes_full": getattr(planner, "last_boxes_full", []),
                "box_dump": getattr(planner, "last_box_dump", [])}

    clean = grab(f"{outdir}/clean.png")
    patched_r = grab(f"{outdir}/patched.png")
    # one comprehensive JSON: task meta + instruction + (grounded SG, MLLM objects, plan) x {clean, patched}
    json.dump({"task": task["id"], "instruction": task["instruction"], "scene": task["scene"],
               "mode": task["mode"], "target_type": task["target_type"],
               "clean": clean, "patched": patched_r},
              open(f"{outdir}/all_outputs.json", "w"), indent=2, default=str)
    return clean, patched_r


def line_with(txt, key):
    for ln in txt.splitlines():
        if key in ln:
            return ln.strip()
    return "(absent)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", default="rung2/gap_tasks_v2.json")
    ap.add_argument("--tasks", default="", help="comma list of task ids (default: all in the file)")
    ap.add_argument("--patch", default=None, help="override each task's patch path (e.g. a new EOT patch)")
    ap.add_argument("--use-uni", action="store_true", help="use each task's universal EOT patch output/chexp/uni_<id>/...")
    ap.add_argument("--no-isolate", action="store_false", dest="isolate",
                    help="Debug the FULL scene (default: isolate the target like the harness).")
    args = ap.parse_args()
    TASKS = {t["id"]: t for t in json.load(open(args.tasks_file))["tasks"]}
    planner = make_planner(scene_graph_text=True, language_only=False)
    planner.set_actions(ACTIONS)
    for tid in (args.tasks.split(",") if args.tasks else list(TASKS)):
        task = TASKS[tid]
        if args.use_uni:
            task = {**task, "patch": f"output/chexp/uni_{tid}/{tid}/{tid}_final.pt"}
        if args.patch:
            task = {**task, "patch": args.patch}
        print(f"\n\n################ {tid}  ({task['scene']}, {task['target_type']}, mode={task['mode']}) ################")
        print(f"instruction: {task['instruction']}")
        r = run_task(planner, task, isolate=args.isolate)
        if r is None:
            print("  (target not visible from any viewpoint)")
            continue
        clean, patched_r = r
        def fmt_boxes(d):
            return f"{d['n_boxes']} GD boxes -> " + ", ".join(
                f"{n}:{p}{'*' if k else ''}" for n, p, k in d["box_dump"][:30])
        print(f"  PER-BOX (cat:conf, *=kept)  CLEAN : {fmt_boxes(clean)}")
        print(f"  PER-BOX (cat:conf, *=kept)  PATCH : {fmt_boxes(patched_r)}")
        print(f"  MLLM's OWN objects  CLEAN: {clean['mllm_objects']}   PATCHED: {patched_r['mllm_objects']}")
        for key in ["The objects are", "The attributes are", "The relations are"]:
            print(f"  [{key}]")
            print(f"     CLEAN  : {line_with(clean['grounded_text'], key)}")
            print(f"     PATCHED: {line_with(patched_r['grounded_text'], key)}")

        def plan_acts(d):
            p = d.get("plan")
            if isinstance(p, dict):
                ep = p.get("executable_plan") or p.get("plan") or []
                return [s.get("action_name", s) if isinstance(s, dict) else s for s in ep]
            return "(unparsed)"
        print("  [executable_plan]")
        print(f"     CLEAN  : {plan_acts(clean)}")
        print(f"     PATCHED: {plan_acts(patched_r)}")
    print("\n(frames + all_outputs.json saved under output/rung2/text_debug/<task>/)")


if __name__ == "__main__":
    main()
