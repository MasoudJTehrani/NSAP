#!/usr/bin/env python
"""
Disambiguation attack — a task where perception is genuinely LOAD-BEARING.

Scene: FloorPlan1 with TWO pots (spawned), one on the stove, one on a counter.
Instruction: "pick up the pot that is on the stove." The instruction names no
instance, so the agent MUST perceive WHICH pot is on the stove. We resolve the
target instance through ESCA's GD+SGClip perception (the attackable path), not the
simulator's nearest-GT lookup — mirroring the real EB-ALFRED multi-instance handling.

Measures, clean vs patched (patch injected on the on-stove pot):
  - on-stove pot perceived as a pot?  -> if not, "the pot on the stove" is unresolvable
  - counter pot still perceived?      -> control: only the TARGET instance should drop
This is the perception-resolution proof; the GT executor previously hid exactly this.

Run:  conda run -n embench python rung2/disambig.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from harness import make_planner, load_patch, ACTIONS, NSAP
from env import Rung2Env
from flip_rate import inject, _iou
from embodiedbench.planner.alfred.esca import convert_gd_boxes_to_pixel

POT_PATCH = "output/chexp/uni_category_pot/category_pot/category_pot_final.pt"  # strong pot-dropper


def on_stove(o):
    return any('StoveBurner' in (p or '') for p in (o.get('parentReceptacles') or []))


def gt_bbox(mask):
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def perceived_pots(planner, image):
    """Boxes that ESCA's GD+SGClip classify as 'pot' -> [(bbox_xyxy, cat, conf), ...]."""
    labels, boxes = planner.predict_gd(image, ["pot"] + planner.common_objects,
                                       box_threshold=planner.gd_box_threshold,
                                       text_threshold=planner.gd_text_threshold)
    if boxes is None or len(boxes) == 0:
        return []
    h, w = image.shape[:2]
    bp = convert_gd_boxes_to_pixel(boxes, w, h)
    planner.predict_laser(image, bp, masks=None, target_names=["pot"],
                          query_names=["pot"], threshold=0.0)
    full = getattr(planner, "last_boxes_full", []) or []
    return [b for b in full if 'pot' in b[1].lower()]


def setup(env):
    """Reset FloorPlan1 and spawn a 2nd pot; return (on-stove target, counter distractor)."""
    env.c.reset("FloorPlan1")
    env.c.step(action="InitialRandomSpawn", randomSeed=0, forceVisible=True,
               numDuplicatesOfType=[{"objectType": "Pot", "count": 2}])
    env.reach = env.c.step("GetReachablePositions").metadata.get("actionReturn", []) or []
    pots = [o for o in env.objects() if o['objectType'] == 'Pot']
    if sum(on_stove(p) for p in pots) != 1 or len(pots) < 2:
        return None, None
    return next(p for p in pots if on_stove(p)), next(p for p in pots if not on_stove(p))


def resolve_on_stove_pot(planner, img, tgt_gt):
    """Perception-driven: is a pot perceived at the on-stove location? (IoU>0.25)."""
    pp = perceived_pots(planner, img)
    tgt_iou = max((_iou(b[0], tgt_gt) for b in pp), default=0.0)
    return len(pp), tgt_iou, tgt_iou > 0.25


def main():
    planner = make_planner(scene_graph_text=True, language_only=False)
    planner.set_actions(ACTIONS)
    patch = load_patch(os.path.join(NSAP, POT_PATCH))
    env = Rung2Env("FloorPlan1", resolution=600)

    print("TASK: 'pick up the pot on the stove'  (2 pots present; instruction names no instance)\n")
    print(f"{'cond':7}{'#pots seen':>12}{'on-stove pot':>16}{'resolved?':>12}{'on-stove PICKED':>18}{'  TASK':>8}")
    for tag, do_inject in [("CLEAN", False), ("PATCH", True)]:
        target, other = setup(env)
        if target is None:
            print(f"{tag}: scene setup failed"); continue
        tid = target['objectId']
        frame, mask, bbox = env.view_target(tid)
        if frame is None:
            print(f"{tag}: on-stove pot not visible"); continue
        tgt_gt = gt_bbox(mask)
        img = inject(frame, mask, bbox, patch) if do_inject else frame

        n_pots, tgt_iou, resolved = resolve_on_stove_pot(planner, img, tgt_gt)

        # Perception-driven executor: only act if the referent was resolved by perception.
        if resolved:
            o = env.obj_by_id(tid)
            env.navigate_to(o)
            env.c.step(action="PickupObject", objectId=tid, forceAction=True)
        picked = bool(env.obj_by_id(tid).get("isPickedUp"))
        tgt_str = ("YES" if resolved else "NO") + f"({tgt_iou:.2f})"
        print(f"{tag:7}{n_pots:>12}{tgt_str:>16}{str(resolved):>12}{str(picked):>18}"
              f"{('SUCCESS' if picked else 'FAIL'):>8}", flush=True)
    env.stop()


if __name__ == "__main__":
    main()
