"""Absolute-confidence attribute probe. For each (object, attribute) we set the object to its
POSITIVE then NEGATIVE state and read VINE's unary score through the PLANNER-faithful forward
(bbox mask + actual bbox + query_names). Unlike AUC (ranking), this reveals whether VINE reads
the attribute CONFIDENTLY (positive >> 0.5, negative << 0.5) -- the only case that has a signal
to attack. open/closed was a coin flip (0.516); this hunts for one that isn't."""
import os, sys, math, json, argparse
sys.path.insert(0, "rung2"); os.environ["NSAP_LASER_ATTR_REL"] = "1"
import harness, numpy as np                                       # noqa: E402
from nav_harness import make_nav_planner                          # noqa: E402
from embodiedbench.planner.alfred.esca import bbox_to_mask        # noqa: E402
from ai2thor.controller import Controller                         # noqa: E402
from ai2thor.platform import CloudRendering                       # noqa: E402

p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o-mini")
c = Controller(platform=CloudRendering, width=600, height=600, renderInstanceSegmentation=True)
QN = ["object", "television", "lamp", "bread", "stove burner", "microwave", "counter top"]

def uscore(img, box, attr):
    h, w = img.shape[:2]; m = [bbox_to_mask(box, h, w)]
    _, u, _, _ = p.predicate_model(batched_video_ids=[0], batched_videos=[img], batched_masks=m,
        batched_bboxes=[box], batched_names=[QN], batched_object_ids=[[0, 0, 0]],
        batched_unary_kws=[[attr, f"not {attr}"]], batched_binary_kws=[[]], batched_obj_pairs=[],
        batched_video_splits=[0], batched_binary_predicates=[None])
    d = {kw: v.item() for (f, o, kw), v in u[0].items()}
    return d[attr]

def views(pos, n=3):
    reach = c.step("GetReachablePositions").metadata["actionReturn"]
    d = lambda q: math.hypot(q["x"] - pos["x"], q["z"] - pos["z"])
    band = sorted([q for q in reach if 0.5 <= d(q) <= 1.6] or sorted(reach, key=d)[:n], key=d)[:n]
    out = []
    for q in band:
        yaw = math.degrees(math.atan2(pos["x"] - q["x"], pos["z"] - q["z"])) % 360
        for hor in (0, 20):
            out.append((q, yaw, hor))
    return out

def measure(obj_type, attr, scene, set_pos, set_neg):
    """Return (pos_scores, neg_scores) for the object's attribute across close views."""
    def read_state(setter):
        c.reset(scene=scene)
        objs = [o for o in c.last_event.metadata["objects"] if o["objectType"] == obj_type]
        if not objs:
            return []
        obj = objs[0]
        if setter and not setter(obj):
            return []
        oid = obj["objectId"]; sc = []
        for (q, yaw, hor) in views(obj["position"]):
            ev = c.step(action="TeleportFull", position=q, rotation=dict(x=0, y=yaw, z=0), horizon=hor, standing=True)
            if not ev.metadata.get("lastActionSuccess", True):
                continue
            box = ev.instance_detections2D.get(oid)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if (x2 - x1) * (y2 - y1) < 500:
                continue
            sc.append(uscore(np.array(ev.frame), list(box), attr))
        return sc
    return read_state(set_pos), read_state(set_neg)

def toggle_on(o):  return c.step(action="ToggleObjectOn",  objectId=o["objectId"], forceAction=True).metadata["lastActionSuccess"]
def toggle_off(o): return c.step(action="ToggleObjectOff", objectId=o["objectId"], forceAction=True).metadata["lastActionSuccess"]
def slice_it(o):   return c.step(action="SliceObject",     objectId=o["objectId"], forceAction=True).metadata["lastActionSuccess"]

# (obj_type, [attr phrasings], set_pos, set_neg, [scenes])
CANDS = [
    ("FloorLamp",   ["on", "turned on", "lit"], toggle_on, toggle_off, ["FloorPlan201","FloorPlan203","FloorPlan205","FloorPlan209"]),
    ("DeskLamp",    ["on", "turned on", "lit"], toggle_on, toggle_off, ["FloorPlan205","FloorPlan301","FloorPlan303","FloorPlan305"]),
    ("Television",  ["on", "turned on"],        toggle_on, toggle_off, ["FloorPlan201","FloorPlan203","FloorPlan205","FloorPlan207"]),
    ("StoveBurner", ["on", "turned on", "lit"], toggle_on, toggle_off, ["FloorPlan1","FloorPlan3","FloorPlan5","FloorPlan7"]),
    ("Bread",       ["sliced", "cut"],          slice_it,  None,       ["FloorPlan1","FloorPlan3","FloorPlan5","FloorPlan7"]),
]

print("\n" + "=" * 78)
print(f"{'object':12s} {'attribute':11s} {'pos(mean)':>10s} {'neg(mean)':>10s} {'gap':>7s} {'n+':>4s} {'n-':>4s}  verdict")
print("=" * 78)
for obj_type, attrs, sp, sn, scenes in CANDS:
    for attr in attrs:
        POS, NEG = [], []
        for scene in scenes:
            try:
                pos, neg = measure(obj_type, attr, scene, sp, sn)
                POS += pos; NEG += neg
            except Exception as e:
                print(f"  ! {obj_type}/{attr}/{scene}: {type(e).__name__}: {str(e)[:50]}")
        if not POS:
            print(f"{obj_type:12s} {attr:11s} {'(no views)':>10s}"); continue
        pm = float(np.mean(POS)); nm = float(np.mean(NEG)) if NEG else float("nan")
        gap = pm - nm if NEG else float("nan")
        conf = pm > 0.65 and (not NEG or nm < 0.45)
        vd = "ATTACKABLE ✓" if conf else ("weak" if (pm > 0.55 or (NEG and gap > 0.15)) else "chance ✗")
        print(f"{obj_type:12s} {attr:11s} {pm:>10.3f} {nm:>10.3f} {gap:>7.3f} {len(POS):>4d} {len(NEG):>4d}  {vd}")
c.stop()
print("=" * 78)
