#!/usr/bin/env python
"""
Debug: show ESCA's grounded scene-graph TEXT, clean vs patched, on the SAME frame,
across many scenarios (category/attribute/relation). After Fix C the grounded text
should be CLEAN (one entry per object). We want to see, per scenario:
  - the MLLM's own object list (its vision), clean vs patched
  - ESCA's grounded objects / attributes / relations, clean vs patched
Run: conda run -n embench python rung2/debug_text.py --tasks category_fridge,category_pot,...
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
from PIL import Image
from harness import make_planner, load_patch, ACTIONS, NSAP
from env import Rung2Env
from functions.attack.patch import paste_patch, compute_patch_size
from functions.attack.anchor import anchor_center

TASKS = {t["id"]: t for t in json.load(open("rung2/tasks.json"))["tasks"]}


def run_task(planner, task):
    env = Rung2Env(task["scene"], resolution=600)
    if task.get("init"):
        env.run_init(task["init"])
    tid = (env._nearest(task["target_type"]) or {}).get("objectId")
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
        planner.act(img_rel, task["instruction"])
        sg = getattr(planner, "last_sg_info", {}) or {}
        return {"mllm_objects": sg.get("current_objects"),
                "grounded_text": getattr(planner, "last_sg_text", "") or "",
                "n_boxes": getattr(planner, "last_n_boxes", 0),
                "box_dump": getattr(planner, "last_box_dump", [])}

    clean = grab(f"{outdir}/clean.png")
    patched_r = grab(f"{outdir}/patched.png")
    json.dump({"clean": clean, "patched": patched_r}, open(f"{outdir}/texts.json", "w"),
              indent=2, default=str)
    return clean, patched_r


def line_with(txt, key):
    for ln in txt.splitlines():
        if key in ln:
            return ln.strip()
    return "(absent)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="category_fridge,category_toaster,category_pot,"
                    "category_coffee_machine,attribute_fridge_open,attribute_microwave_open,relation_on_pot")
    ap.add_argument("--patch", default=None, help="override each task's patch path (e.g. a new EOT patch)")
    ap.add_argument("--use-uni", action="store_true", help="use each task's universal EOT patch output/chexp/uni_<id>/...")
    args = ap.parse_args()
    planner = make_planner(scene_graph_text=True, language_only=False)
    planner.set_actions(ACTIONS)
    for tid in args.tasks.split(","):
        task = TASKS[tid]
        if args.use_uni:
            task = {**task, "patch": f"output/chexp/uni_{tid}/{tid}/{tid}_final.pt"}
        if args.patch:
            task = {**task, "patch": args.patch}
        print(f"\n\n################ {tid}  ({task['scene']}, {task['target_type']}, mode={task['mode']}) ################")
        print(f"instruction: {task['instruction']}")
        r = run_task(planner, task)
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
    print("\n(frames + texts.json saved under output/rung2/text_debug/<task>/)")


if __name__ == "__main__":
    main()
