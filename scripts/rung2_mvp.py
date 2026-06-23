#!/usr/bin/env python
"""
Rung-2 harness MVP: does the FROZEN per-scenario chameleon patch flip SGClip's
grounded scene graph on a LIVE AI2-THOR-5.0 agent-viewpoint frame of its own
scenario's kitchen+object?  (The rung-1 -> rung-2 transfer question.)

Pipeline: load patch -> THOR 5.0 CloudRendering FloorPlan -> face the real target
object -> SGClip on clean frame -> inject patch exactly as eval (paste_patch +
anchor_center + coverage) -> SGClip on patched frame -> compare.

Run (embench env, with NSAP on PYTHONPATH):
  conda run -n embench python scripts/rung2_mvp.py
"""
import os, sys, json
NSAP = "/home/usi/NSAP"
for p in (NSAP, f"{NSAP}/src/LASER", f"{NSAP}/src/LASER/laser/models", f"{NSAP}/SceneGraphBench"):
    sys.path.insert(0, p)
os.chdir(NSAP)

import numpy as np, torch
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from llava_clip_model_v3 import PredicateModel
from safetensors.torch import load_file
from functions.attack.patch import paste_patch, compute_patch_size
from functions.attack.anchor import anchor_center, Anchor          # noqa: F401 (Anchor for unpickle)
from functions.attack.train import AttackConfig                    # noqa: F401 (for unpickle)
from sgbench.framing import face_pose, jitter

# --- target: the fridge category chameleon patch + its scenario ---
PATCH = "output/chexp/cham_category_fridge_scenario_00081/category_fridge/category_fridge_final.pt"
SCENE, TARGET_ID, SCN = "FloorPlan13", "Fridge|-00.32|00.00|+03.60", "scenario_00081"
RES = 600
ATTR = ['open', 'closed', 'on', 'off', 'sliced', 'cooked', 'clean', 'dirty', 'filled', 'broken']

# --- load frozen patch (w -> pixels), anchor, coverage ---
ck = torch.load(PATCH, map_location="cpu", weights_only=False)
w, anchor = ck["w"], ck["anchor"]
coverage = getattr(ck.get("config"), "coverage_ratio", 0.3)
patch_hwc = (127.5 * (torch.tanh(w) + 1.0)).permute(1, 2, 0)        # [ph,pw,3] in [0,255]
print(f"patch {tuple(w.shape)}  coverage={coverage}  anchor=({anchor.dx},{anchor.dy})")

# --- candidate categories = this scenario's category set (deployment-faithful) ---
sg = json.load(open(f"SceneGraphBench/data/kitchen_pilot/scenarios/{SCN}/scene_graph.json"))
consts = sorted({o["category"] for fr in sg["frames"] for o in fr["objects"]})

# --- SGClip (= VINE), patch32 backbone + trained weights ---
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PredicateModel(128, 1, dev, "openai/clip-vit-base-patch32").to(dev)
model.load_state_dict(load_file("model/vine/model.safetensors"), strict=False)
model.eval()


def sg_target(image_uint8, mask, bbox):
    """Run SGClip on a single object (the target) -> (category, attribute) argmax."""
    masks_in = [mask[:, :, None]]
    boxes = [[bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]]
    with torch.no_grad():
        cate, unary, _b, *_ = model(
            batched_video_ids=[0], batched_videos=[image_uint8], batched_masks=masks_in,
            batched_bboxes=boxes, batched_names=[consts], batched_object_ids=[[0, 0, 0]],
            batched_unary_kws=[ATTR], batched_binary_kws=[[]], batched_obj_pairs=[],
            batched_video_splits=[0], batched_binary_predicates=[[]])
    _f = lambda p: p.item() if hasattr(p, "item") else float(p)
    cat = max(((n, _f(p)) for (o, n), p in cate[0].items() if o == 0 and n != "other"),
              key=lambda x: x[1], default=None)
    at = max(((a, _f(p)) for (f, o, a), p in unary[0].items() if o == 0),
             key=lambda x: x[1], default=None)
    return cat, at


# --- THOR 5.0: load scene, face the real target, find a viewpoint where it's visible ---
c = Controller(platform=CloudRendering, scene=SCENE, width=RES, height=RES,
               renderInstanceSegmentation=True)
reach = c.step("GetReachablePositions").metadata["actionReturn"]
obj = next(o for o in c.last_event.metadata["objects"] if o["objectId"] == TARGET_ID)
pose = face_pose(obj, reach)
assert pose is not None, "no reachable pose faces the target"
base_pos, base_yaw, base_hor = pose

import random
rng = random.Random(0)
ev = None
for v in range(8):
    yaw, hor = (base_yaw, base_hor) if v == 0 else jitter(base_yaw, base_hor, rng)
    ev = c.step("TeleportFull", position=base_pos, rotation=dict(x=0, y=yaw, z=0),
                horizon=hor, standing=True)
    m = ev.instance_masks.get(TARGET_ID)
    if m is not None and int(m.sum()) > 600:
        break
mask = ev.instance_masks.get(TARGET_ID)
assert mask is not None, "target not visible from any tried viewpoint"
ys, xs = np.where(mask)
bbox = {"x1": int(xs.min()), "y1": int(ys.min()), "x2": int(xs.max()), "y2": int(ys.max())}
frame = np.array(ev.frame)                                          # uint8 RGB
print(f"viewpoint found: target mask area={int(mask.sum())} bbox={bbox}")

# --- clean SG ---
clean = sg_target(frame, mask, bbox)

# --- inject patch exactly as eval (size to bbox, anchor-center clamp, paste) ---
ph, pw = compute_patch_size(bbox, coverage)
cx, cy = anchor_center(anchor, bbox, ph, pw)
ft = torch.from_numpy(frame.astype(np.float32))
patched = paste_patch(ft, patch_hwc, cx, cy, ph, pw).clamp(0, 255).numpy().astype(np.uint8)
patched_sg = sg_target(patched, mask, bbox)

# --- save + report ---
os.makedirs("output/rung2", exist_ok=True)
Image.fromarray(frame).save("output/rung2/fridge_clean.png")
Image.fromarray(patched).save("output/rung2/fridge_patched.png")
c.stop()
print("\n==================  RUNG-2 LIVE-FRAME SG FLIP  ==================")
print(f"  scene={SCENE}  target=Fridge (category attack)")
print(f"  CLEAN   -> category={clean[0]}  attribute={clean[1]}")
print(f"  PATCHED -> category={patched_sg[0]}  attribute={patched_sg[1]}")
flip = clean[0] and patched_sg[0] and clean[0][0] != patched_sg[0][0]
print(f"  CATEGORY FLIPPED: {flip}")
print("  images: output/rung2/fridge_clean.png , fridge_patched.png")
