"""
VINE-on-sim probe — dataset-creation Step D4 go/no-go.

Renders AI2-THOR frames, runs VINE *exactly as the eval does* (via
predict_scene_graph), and reports whether VINE cleanly recognizes the
ground-truth object categories the simulator hands us for free. The simulator's
instance masks + objectType are the ground truth; VINE's top-1 category is the
prediction.

Sampling is OBJECT-CENTRIC: for each target-category object we navigate the agent
close to it and face it, then score only objects that are actually prominent in
frame (mask area >= --min-frac of the image). This mirrors deployment — in ESCA
the robot navigates *to* an object to manipulate it, so VINE sees it large and
centered, never as a 5-pixel speck across the room. (An earlier random-teleport
version mostly framed ceilings / cabinet fronts and badly understated accuracy.)

Vanilla ai2thor Controller (CloudRendering), decoupled from ESCA's ALFRED wrapper
— no X server / ALFRED task data needed. NOTE: EB-ALFRED proper pins
ai2thor==2.1.0; modern renders are if anything cleaner, so a PASS here is a
reasonable (not pixel-exact) predictor for the pinned stack.

Usage:
    python -m functions.sim.probe --scenes 1 2 3 4 5 --per-scene 12 --out output/thor_probe
"""

import os
import re
import json
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import torch


# ── attackable target categories (skip structural drawers/cabinets/walls/floor) ─

TARGET_TYPES = {
    # appliances
    'Fridge', 'Microwave', 'Toaster', 'CoffeeMachine', 'Kettle',
    # tableware / cookware
    'Mug', 'Cup', 'Bowl', 'Plate', 'Pot', 'Pan',
    # food
    'Apple', 'Bread', 'Tomato', 'Lettuce', 'Potato', 'Egg',
    # utensils
    'Knife', 'ButterKnife', 'Fork', 'Spoon', 'Spatula', 'Ladle',
    # other manipulable
    'WineBottle', 'SoapBottle', 'Vase', 'Statue', 'Book', 'Laptop',
    'CellPhone', 'SaltShaker', 'PepperShaker', 'HousePlant',
}


# ── sim objectType → CLIP-friendly word ───────────────────────────────────────

_SPECIAL = {
    'CounterTop': 'countertop', 'SoapBottle': 'soap bottle', 'SprayBottle': 'spray bottle',
    'GarbageCan': 'garbage can', 'DiningTable': 'dining table', 'SideTable': 'side table',
    'CoffeeTable': 'coffee table', 'StoveBurner': 'stove burner', 'StoveKnob': 'stove knob',
    'ButterKnife': 'butter knife', 'PepperShaker': 'pepper shaker', 'SaltShaker': 'salt shaker',
    'DishSponge': 'dish sponge', 'WineBottle': 'wine bottle', 'CellPhone': 'cellphone',
    'RemoteControl': 'remote control', 'HousePlant': 'house plant', 'CreditCard': 'credit card',
    'TissueBox': 'tissue box', 'LightSwitch': 'light switch', 'SinkBasin': 'sink',
    'Glassbottle': 'glass bottle', 'CoffeeMachine': 'coffee machine',
}

# Fixed candidate vocabulary (the classification "consts"); each visible target's
# GT word is unioned in so the right answer is always available, but distractors
# keep the test honest.
_KITCHEN_VOCAB = [
    'apple', 'mug', 'bowl', 'plate', 'knife', 'fork', 'spoon', 'bread', 'tomato',
    'lettuce', 'potato', 'egg', 'pan', 'pot', 'cup', 'bottle', 'fridge',
    'microwave', 'toaster', 'kettle', 'book', 'laptop', 'cellphone',
    'spatula', 'vase', 'coffee machine', 'house plant', 'statue',
]


def thor_type_to_word(otype: str) -> str:
    if otype in _SPECIAL:
        return _SPECIAL[otype]
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', otype).lower()


# ── object-centric framing ────────────────────────────────────────────────────

def face_pose(obj, reach, eye_offset=0.9):
    """Reachable position near `obj` + yaw/horizon to face it."""
    op = obj['position']
    near = [p for p in reach if 0.5 <= math.hypot(op['x'] - p['x'], op['z'] - p['z']) <= 1.8]
    pool = near or reach
    if not pool:
        return None
    best = min(pool, key=lambda p: math.hypot(op['x'] - p['x'], op['z'] - p['z']))
    dx, dz = op['x'] - best['x'], op['z'] - best['z']
    dist = math.hypot(dx, dz)
    yaw = math.degrees(math.atan2(dx, dz)) % 360                       # 0=+z, CW
    horizon = math.degrees(math.atan2((best['y'] + eye_offset) - op['y'], max(dist, 1e-3)))
    horizon = max(-30.0, min(60.0, horizon))                          # THOR limits
    return best, yaw, horizon


# ── one rendered view → eval-format inputs ────────────────────────────────────

def _bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return {'x1': int(xs.min()), 'y1': int(ys.min()),
            'x2': int(xs.max()), 'y2': int(ys.max())}


def build_inputs_from_event(event, keep_types, min_frac, bgr=False):
    """frame tensor + masks/bboxes for prominent, visible objects of keep_types."""
    frame = event.frame
    if bgr:
        frame = frame[:, :, ::-1]   # RGB->BGR, so VINE's internal BGR2RGB lands as true RGB at CLIP
    frame_t = torch.from_numpy(np.ascontiguousarray(frame).astype(np.float32))  # [H,W,3] 0-255
    H, W = frame.shape[0], frame.shape[1]
    min_px = max(400, int(min_frac * H * W))
    instance_masks = getattr(event, 'instance_masks', {}) or {}

    masks, bboxes, id2type, id2word = {}, {}, {}, {}
    next_id = 0
    for obj in event.metadata['objects']:
        if not obj.get('visible', False):
            continue
        if keep_types is not None and obj['objectType'] not in keep_types:
            continue
        oid = obj['objectId']
        if oid not in instance_masks:
            continue
        m = np.asarray(instance_masks[oid]).astype(bool)
        if m.ndim != 2 or int(m.sum()) < min_px:
            continue
        bb = _bbox_from_mask(m)
        if bb is None:
            continue
        masks[next_id] = torch.from_numpy(m.copy())
        bboxes[next_id] = bb
        id2type[next_id] = obj['objectType']
        id2word[next_id] = thor_type_to_word(obj['objectType'])
        next_id += 1
    return frame_t, masks, bboxes, id2type, id2word


def save_overlay(frame_t, bboxes, id2word, pred, conf, path):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    img = Image.fromarray(frame_t.numpy().clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for oid, bb in bboxes.items():
        gt, pr = id2word.get(oid, '?'), pred.get(oid)
        hit = (pr is not None and gt in pr)
        col = (30, 180, 30) if hit else (210, 40, 40)
        draw.rectangle([bb['x1'], bb['y1'], bb['x2'], bb['y2']], outline=col, width=2)
        c = conf.get(oid)
        draw.text((bb['x1'] + 2, max(0, bb['y1'] - 10)),
                  f"{gt}|{pr}{'' if c is None else f' {c:.2f}'}", fill=col)
    img.save(path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="VINE clean-accuracy probe on AI2-THOR (object-centric)")
    ap.add_argument('--scenes', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--per-scene', type=int, default=12, help="target objects framed per scene")
    ap.add_argument('--res', type=int, default=600)
    ap.add_argument('--min-frac', type=float, default=0.008, help="min mask area as fraction of frame")
    ap.add_argument('--all-types', action='store_true', help="don't restrict to TARGET_TYPES")
    ap.add_argument('--bgr', action='store_true', help="pre-flip RGB->BGR (test VINE color order)")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default='output/thor_probe')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering
    from run_attack import load_vine
    from functions.eval.scene_graph import predict_scene_graph

    device = torch.device(args.device)
    vine = load_vine(args.device)
    keep_types = None if args.all_types else TARGET_TYPES

    controller = Controller(platform=CloudRendering, scene=f"FloorPlan{args.scenes[0]}",
                            width=args.res, height=args.res, renderInstanceSegmentation=True)

    per_cat = defaultdict(lambda: [0, 0])
    confs, n_obj, n_views, seen = [], 0, 0, set()

    for sc in args.scenes:
        controller.reset(scene=f"FloorPlan{sc}")
        reach = controller.step("GetReachablePositions").metadata.get("actionReturn", []) or []
        targets = [o for o in controller.last_event.metadata['objects']
                   if (keep_types is None or o['objectType'] in keep_types)]
        random.shuffle(targets)

        for obj in targets[:args.per_scene]:
            fp = face_pose(obj, reach)
            if fp is None:
                continue
            pos, yaw, hor = fp
            ev = controller.step(action="TeleportFull", position=pos,
                                 rotation=dict(x=0, y=yaw, z=0), horizon=hor, standing=True)
            if not ev.metadata.get("lastActionSuccess", True):
                continue

            frame_t, masks, bboxes, id2type, id2word = build_inputs_from_event(ev, keep_types, args.min_frac, args.bgr)
            if not masks:
                continue

            consts = sorted(set(_KITCHEN_VOCAB) | set(id2word.values()))
            sg = predict_scene_graph(vine, [frame_t], {0: masks}, {0: bboxes},
                                     {'consts': consts, 'unary_kws': [], 'binary_kws': []}, device)

            pred = {oid: sg.categories.get(oid) for oid in masks}
            conf = {oid: sg.category_conf.get(oid) for oid in masks}
            for oid in masks:
                # dedupe the same sim object across overlapping framings
                key = (sc, id2type[oid], bboxes[oid]['x1'], bboxes[oid]['y1'])
                if key in seen:
                    continue
                seen.add(key)
                gt, pr = id2word[oid], pred[oid]
                per_cat[gt][0] += int(pr is not None and gt in pr)
                per_cat[gt][1] += 1
                if conf[oid] is not None:
                    confs.append(conf[oid])
                n_obj += 1

            save_overlay(frame_t, bboxes, id2word, pred, conf,
                         os.path.join(args.out, f"fp{sc}_{obj['objectType']}.png"))
            n_views += 1

    controller.stop()

    tot_hit = sum(h for h, _ in per_cat.values())
    acc = tot_hit / max(n_obj, 1)
    mean_conf = float(np.mean(confs)) if confs else 0.0

    print("\n================  VINE-on-THOR probe (object-centric)  ================")
    print(f"views={n_views}  objects={n_obj}  overall top-1 acc={acc:.3f}  mean conf={mean_conf:.3f}")
    print("per-category (hits/total) acc:")
    for word in sorted(per_cat, key=lambda w: -per_cat[w][1]):
        h, t = per_cat[word]
        print(f"  {word:<16} {h:3d}/{t:<3d}  {h / t:.2f}")
    print("======================================================================")
    print("GO if the categories you intend to attack score well (>= ~0.6). You only")
    print("need a handful of strong target categories, not all of them.")

    json.dump({'overall_top1_acc': acc, 'mean_conf': mean_conf, 'n_views': n_views,
               'n_objects': n_obj,
               'per_category': {w: {'hits': h, 'total': t} for w, (h, t) in per_cat.items()},
               'args': vars(args)},
              open(os.path.join(args.out, 'probe_summary.json'), 'w'), indent=2)
    print(f"Saved overlays + probe_summary.json → {args.out}/")


if __name__ == '__main__':
    main()
