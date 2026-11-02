"""Can SGClip ground the facts the ALFRED spatial / visual_appearance tasks depend on?

Before building any harness, test the two things those task sets need:

  A. COLOUR attributes ("yellow bottle", "black table"). Ground truth = the mean colour
     of the object's own pixels, taken from the simulator's instance mask.
  B. SPATIAL relations ("to the right of the sink"). Ground truth = image-space
     geometry of the two objects' boxes, which is unambiguous.

Both are asked through the DEPLOYMENT path (predict_gd -> predict_laser), because that
is the only path the attack can reach. A fact SGClip cannot ground here is a fact no
amount of task design will make attackable.
"""
import os, sys, json, math, collections
sys.path.insert(0, "rung2")
os.environ.setdefault("NSAP_AGGR_THRES", "0.4")
# esca.py:276 defaults NSAP_LASER_ATTR_REL to "0", which silently discards every
# relation keyword. staged_bench.py sets it to "1". Without it this probe measures nothing.
os.environ["NSAP_LASER_ATTR_REL"] = "1"
import numpy as np
import ai2thor.controller
from nav_harness import make_nav_planner

RES = 500
KITCHENS  = ["FloorPlan1", "FloorPlan6", "FloorPlan10"]
BATHROOMS = ["FloorPlan401", "FloorPlan405", "FloorPlan410"]
SCENES = KITCHENS + BATHROOMS

BASIC = {"red":(180,40,40), "green":(50,140,60), "blue":(50,80,180), "yellow":(220,200,60),
         "white":(225,225,225), "black":(40,40,40), "grey":(130,130,130),
         "brown":(120,80,50), "silver":(190,190,195), "orange":(225,140,50)}

def colour_name(rgb):
    return min(BASIC, key=lambda k: sum((a-b)**2 for a,b in zip(rgb, BASIC[k])))

def iou(a, b):
    ix = max(0, min(a[2],b[2])-max(a[0],b[0])); iy = max(0, min(a[3],b[3])-max(a[1],b[1]))
    inter = ix*iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def main():
    p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o")
    c = ai2thor.controller.Controller(scene=SCENES[0], width=RES, height=RES,
                                      renderInstanceSegmentation=True)
    print(f"  use_laser_attr_rel = {getattr(p, 'use_laser_attr_rel', None)}  (must be True)\n", flush=True)

    attr_stats = collections.Counter()
    rel_stats  = collections.Counter()
    attr_rows, rel_rows = [], []

    for scene in SCENES:
        c.reset(scene=scene)
        reach = c.step("GetReachablePositions").metadata.get("actionReturn", []) or []
        if not reach:
            continue
        # a few viewpoints per scene
        for k in range(0, len(reach), max(1, len(reach)//4))[:4] if False else [i for i in range(0, len(reach), max(1, len(reach)//4))][:4]:
            pos = reach[k]
            for rot in (0, 90, 180, 270):
                c.step(action="Teleport", position=pos, rotation={"x":0,"y":rot,"z":0},
                       horizon=0, standing=True)
                ev = c.last_event
                det = ev.instance_detections2D or {}
                frame = ev.frame
                seg = ev.instance_masks or {}
                vis = [(oid, b) for oid, b in det.items()
                       if (b[2]-b[0])*(b[3]-b[1]) > 2500 and "|" in oid]
                if len(vis) < 2:
                    continue
                names = sorted({oid.split("|")[0].lower() for oid, _ in vis})[:8]
                labels, boxes = p.predict_gd(frame, names, box_threshold=0.1, text_threshold=0.1)
                if boxes is None or len(boxes) == 0:
                    continue

                # ---------- A. colour ----------
                for oid, b in vis[:3]:
                    m = seg.get(oid)
                    if m is None or m.sum() < 400:
                        continue
                    mean = tuple(int(x) for x in frame[m].mean(axis=0))
                    true_c = colour_name(mean)
                    wrong_c = "green" if true_c != "green" else "red"
                    tname = oid.split("|")[0].lower()
                    _, sb, _, sa = p.predict_laser(image=frame, boxes=boxes,
                        target_names=[tname], query_names=names+["other"],
                        query_attributes=[true_c, wrong_c], query_relation_kws=[])
                    got = set()
                    if sa:
                        for lst in sa:
                            got |= {str(x).lower() for x in (lst or [])}
                    hit_true  = true_c in got
                    hit_wrong = wrong_c in got
                    attr_stats["n"] += 1
                    attr_stats["true_granted"] += hit_true
                    attr_stats["false_granted"] += hit_wrong
                    attr_rows.append((scene, tname, mean, true_c, sorted(got)))

                # ---------- B. spatial relation ----------
                for i in range(min(2, len(vis))):
                    for j in range(len(vis)):
                        if i == j: continue
                        (oa, ba), (ob, bb) = vis[i], vis[j]
                        na, nb = oa.split("|")[0].lower(), ob.split("|")[0].lower()
                        if na == nb: continue
                        ca, cb = (ba[0]+ba[2])/2, (bb[0]+bb[2])/2
                        if abs(ca-cb) < 60: continue          # too close to call
                        truth = "to the right of" if ca > cb else "to the left of"
                        false = "to the left of" if ca > cb else "to the right of"
                        _, sb2, sr, _ = p.predict_laser(image=frame, boxes=boxes,
                            target_names=[na, nb], query_names=names+["other"],
                            query_attributes=[], query_relation_kws=[truth, false])
                        rels = [str(x).lower() for x in (sr or [])]
                        blob = " ".join(rels)
                        rel_stats["n"] += 1
                        rel_stats["any_relation"] += bool(rels)
                        rel_stats["true_present"] += truth in blob
                        rel_stats["false_present"] += false in blob
                        if len(rel_rows) < 12:
                            rel_rows.append((scene, na, nb, truth, str(rels)[:120]))
                        break
                if attr_stats["n"] > 40 and rel_stats["n"] > 40:
                    break
            if attr_stats["n"] > 40 and rel_stats["n"] > 40:
                break
        print(f"{scene}: attr n={attr_stats['n']} rel n={rel_stats['n']}", flush=True)

    print("\n================ COLOUR ATTRIBUTE ================")
    n = max(attr_stats["n"], 1)
    print(f"  objects tested            : {attr_stats['n']}")
    print(f"  correct colour granted    : {attr_stats['true_granted']} ({attr_stats['true_granted']/n:.0%})")
    print(f"  WRONG colour also granted : {attr_stats['false_granted']} ({attr_stats['false_granted']/n:.0%})")
    for r in attr_rows[:10]:
        print(f"    {r[0]:13s} {r[1]:14s} mean_rgb={r[2]} truth={r[3]:7s} granted={r[4]}")

    print("\n================ SPATIAL RELATION ================")
    n = max(rel_stats["n"], 1)
    print(f"  ordered pairs tested      : {rel_stats['n']}")
    print(f"  any relation grounded     : {rel_stats['any_relation']} ({rel_stats['any_relation']/n:.0%})")
    print(f"  TRUE relation present     : {rel_stats['true_present']} ({rel_stats['true_present']/n:.0%})")
    print(f"  FALSE relation present    : {rel_stats['false_present']} ({rel_stats['false_present']/n:.0%})")
    for r in rel_rows:
        print(f"    {r[0]:13s} ({r[1]}, ?, {r[2]}) truth='{r[3]}' -> {r[4]}")
    json.dump({"attr": dict(attr_stats), "rel": dict(rel_stats)},
              open("output/probes/attr_rel_probe.json","w"), indent=2)
    c.stop()

if __name__ == "__main__":
    main()
