"""Follow-up. Two questions the first probe raised:

  1. SGClip grounded ZERO left/right relations, yet it grounds "on" for pots. So which
     relation predicates does it support at all? Query a battery and count how often
     each is emitted, regardless of truth.
  2. The colour test was dominated by walls, floors and doorframes. ALFRED's
     visual_appearance tasks are about SMALL MANIPULABLE objects. Retest colour on
     those only.
"""
import os, sys, collections, json
sys.path.insert(0, "rung2")
os.environ.setdefault("NSAP_AGGR_THRES", "0.4")
# WITHOUT THIS the relation head is disabled and every predicate returns nothing:
# esca.py:276 reads NSAP_LASER_ATTR_REL and defaults it to "0". staged_bench.py sets it to "1".
os.environ["NSAP_LASER_ATTR_REL"] = "1"
import numpy as np
import ai2thor.controller
from nav_harness import make_nav_planner

RES = 500
SCENES = ["FloorPlan1", "FloorPlan6", "FloorPlan10", "FloorPlan401", "FloorPlan405"]
PREDS = ["on", "on top of", "in", "inside", "next to", "under", "above", "behind",
         "in front of", "near", "to the right of", "to the left of"]
SMALL = {"spraybottle","soapbottle","bowl","mug","cup","cloth","vase","pot","plate",
         "pan","bread","apple","tomato","lettuce","potato","kettle","toaster","book",
         "candle","soapbar","toiletpaper","towel","handtowel","papertowelroll","statue"}
BASIC = {"red":(180,40,40),"green":(50,140,60),"blue":(50,80,180),"yellow":(220,200,60),
         "white":(225,225,225),"black":(40,40,40),"grey":(130,130,130),
         "brown":(120,80,50),"silver":(190,190,195),"orange":(225,140,50)}
def cname(rgb): return min(BASIC, key=lambda k: sum((a-b)**2 for a,b in zip(rgb,BASIC[k])))

def main():
    p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o")
    print(f"  use_laser_attr_rel = {getattr(p, 'use_laser_attr_rel', None)}  (must be True)", flush=True)
    c = ai2thor.controller.Controller(scene=SCENES[0], width=RES, height=RES,
                                      renderInstanceSegmentation=True)
    pred_hits = collections.Counter(); pred_tries = collections.Counter()
    a_n = a_true = a_false = 0; arows = []
    for scene in SCENES:
        c.reset(scene=scene)
        reach = c.step("GetReachablePositions").metadata.get("actionReturn", []) or []
        for k in [i for i in range(0, len(reach), max(1, len(reach)//6))][:6]:
            for rot in (0, 90, 180, 270):
                c.step(action="Teleport", position=reach[k], rotation={"x":0,"y":rot,"z":0},
                       horizon=0, standing=True)
                ev = c.last_event; frame = ev.frame
                det = ev.instance_detections2D or {}; seg = ev.instance_masks or {}
                vis = [(o,b) for o,b in det.items()
                       if "|" in o and (b[2]-b[0])*(b[3]-b[1]) > 2000]
                if len(vis) < 2: continue
                names = sorted({o.split("|")[0].lower() for o,_ in vis})[:8]
                labels, boxes = p.predict_gd(frame, names, box_threshold=0.1, text_threshold=0.1)
                if boxes is None or len(boxes) == 0: continue

                # 1. which predicates ever ground
                for pred in PREDS:
                    _,_,sr,_ = p.predict_laser(image=frame, boxes=boxes,
                        target_names=names, query_names=names+["other"],
                        query_attributes=[], query_relation_kws=[pred])
                    pred_tries[pred] += 1
                    if sr: pred_hits[pred] += 1

                # 2. colour on SMALL objects only
                for oid, b in vis:
                    t = oid.split("|")[0].lower()
                    if t not in SMALL: continue
                    m = seg.get(oid)
                    if m is None or m.sum() < 300: continue
                    mean = tuple(int(x) for x in frame[m].mean(axis=0))
                    tc = cname(mean); wc = "green" if tc != "green" else "red"
                    _,_,_,sa = p.predict_laser(image=frame, boxes=boxes,
                        target_names=[t], query_names=names+["other"],
                        query_attributes=[tc, wc], query_relation_kws=[])
                    got = set()
                    for lst in (sa or []): got |= {str(x).lower() for x in (lst or [])}
                    a_n += 1; a_true += tc in got; a_false += wc in got
                    if len(arows) < 14: arows.append((scene, t, mean, tc, sorted(got)))
                if sum(pred_tries.values()) > 260 and a_n > 30: break
            if sum(pred_tries.values()) > 260 and a_n > 30: break
        print(f"{scene}: pred_tries={sum(pred_tries.values())} colour_n={a_n}", flush=True)

    print("\n========= WHICH RELATION PREDICATES GROUND AT ALL =========")
    for pred in PREDS:
        t = pred_tries[pred]; h = pred_hits[pred]
        print(f"  {pred:18s} grounded in {h:>3}/{t:<3} frames ({h/max(t,1):>4.0%})")
    print("\n========= COLOUR ON SMALL MANIPULABLE OBJECTS =========")
    print(f"  objects tested            : {a_n}")
    print(f"  correct colour granted    : {a_true} ({a_true/max(a_n,1):.0%})")
    print(f"  WRONG colour also granted : {a_false} ({a_false/max(a_n,1):.0%})")
    for r in arows:
        print(f"    {r[0]:12s} {r[1]:14s} rgb={r[2]} truth={r[3]:7s} granted={r[4]}")
    json.dump({"pred_hits":dict(pred_hits),"pred_tries":dict(pred_tries),
               "colour":{"n":a_n,"true":a_true,"false":a_false}},
              open("output/probes/vocab_probe.json","w"), indent=2)
    c.stop()

if __name__ == "__main__":
    main()
