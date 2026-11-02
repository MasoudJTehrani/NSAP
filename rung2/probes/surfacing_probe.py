"""Two measurements on our own surfacing code (not stock ESCA).

A. THE ATTRIBUTE GUARD. esca.py only writes attributes into the scene-graph text when
   len(target_attributes_grounded) == len(target_labels); otherwise it drops ALL of them.
   How often does that guard fail?

B. THE RELATION SELECTION RULE. Today each relation keyword is kept whenever it beats its
   own negation, independently. So "to the right of" and "to the left of" can both survive
   for the same pair, and both get printed. Measure the alternative: for each ordered pair,
   keep only the ARGMAX relation across the queried keywords. Does that pick the true
   direction?

Ground truth for direction is image-space box geometry, which is unambiguous.
"""
import os, sys, collections, json
sys.path.insert(0, "rung2")
os.environ.setdefault("NSAP_AGGR_THRES", "0.4")
os.environ["NSAP_LASER_ATTR_REL"] = "1"
import numpy as np
import ai2thor.controller
from nav_harness import make_nav_planner

RES = 500
SCENES = ["FloorPlan1","FloorPlan6","FloorPlan10","FloorPlan401","FloorPlan405","FloorPlan201"]
DIRS = ["to the right of", "to the left of"]

def main():
    p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o")
    print(f"  use_laser_attr_rel = {getattr(p,'use_laser_attr_rel',None)}\n", flush=True)
    c = ai2thor.controller.Controller(scene=SCENES[0], width=RES, height=RES,
                                      renderInstanceSegmentation=True)
    g = collections.Counter()      # attribute guard
    r = collections.Counter()      # relation rule
    rows = []
    for scene in SCENES:
        c.reset(scene=scene)
        reach = c.step("GetReachablePositions").metadata.get("actionReturn", []) or []
        for k in [i for i in range(0, len(reach), max(1,len(reach)//6))][:6]:
            for rot in (0,90,180,270):
                c.step(action="Teleport", position=reach[k], rotation={"x":0,"y":rot,"z":0},
                       horizon=0, standing=True)
                ev=c.last_event; frame=ev.frame
                det=ev.instance_detections2D or {}
                vis=[(o,b) for o,b in det.items() if "|" in o and (b[2]-b[0])*(b[3]-b[1])>2000]
                if len(vis)<2: continue
                names=sorted({o.split("|")[0].lower() for o,_ in vis})[:8]
                labels, boxes = p.predict_gd(frame, names, box_threshold=0.1, text_threshold=0.1)
                if boxes is None or len(boxes)==0: continue
                p._nsap_rel_scored = {}; p._nsap_rel_names = {}
                lab, sb, rels, attrs = p.predict_laser(image=frame, boxes=boxes,
                    target_names=names, query_names=names+["other"],
                    query_attributes=["open","clean"], query_relation_kws=DIRS)

                # ---- A. attribute guard ----
                if lab:
                    g["frames_with_objects"] += 1
                    if len(attrs or []) == len(lab): g["guard_passes"] += 1
                    else:
                        g["guard_FAILS_attrs_dropped"] += 1
                        g[f"mismatch {len(attrs or [])} vs {len(lab)}"] += 1

                # ---- B. relation rule ----
                scored = getattr(p, "_nsap_rel_scored", {}) or {}
                nmap   = getattr(p, "_nsap_rel_names", {}) or {}
                if not scored: continue
                gt_center = {}
                for oid,b in vis:
                    gt_center[oid.split("|")[0].lower()] = (b[0]+b[2])/2
                pairs = collections.defaultdict(dict)
                for (pair, rel), prob in scored.items():
                    if rel in DIRS: pairs[pair][rel] = prob
                for pair, d in pairs.items():
                    if len(d) < len(DIRS): continue
                    soid, ooid = pair
                    sname, oname = nmap.get(soid), nmap.get(ooid)
                    if not sname or not oname or sname==oname: continue
                    if sname not in gt_center or oname not in gt_center: continue
                    dx = gt_center[sname]-gt_center[oname]
                    if abs(dx) < 60: continue
                    truth = "to the right of" if dx>0 else "to the left of"
                    false = "to the left of" if dx>0 else "to the right of"
                    r["pairs"] += 1
                    # current rule: each keyword kept if it beats its own negation
                    cur_true  = d[truth] > scored.get((pair, f"not {truth}"), 1e9)
                    cur_false = d[false] > scored.get((pair, f"not {false}"), 1e9)
                    if cur_true:  r["current_emits_true"]  += 1
                    if cur_false: r["current_emits_false"] += 1
                    if cur_true and cur_false: r["current_emits_BOTH"] += 1
                    # proposed rule: argmax across the queried directions
                    pick = max(DIRS, key=lambda x: d[x])
                    if pick == truth: r["argmax_correct"] += 1
                    else: r["argmax_wrong"] += 1
                    if len(rows)<12:
                        rows.append((sname, oname, truth, round(d[truth],3), round(d[false],3), pick==truth))
        print(f"{scene}: guard frames={g['frames_with_objects']} rel pairs={r['pairs']}", flush=True)

    print("\n========== A. ATTRIBUTE GUARD ==========")
    n=max(g["frames_with_objects"],1)
    print(f"  frames with grounded objects : {g['frames_with_objects']}")
    print(f"  guard passes (attrs written) : {g['guard_passes']} ({g['guard_passes']/n:.0%})")
    print(f"  guard FAILS (attrs dropped)  : {g['guard_FAILS_attrs_dropped']} ({g['guard_FAILS_attrs_dropped']/n:.0%})")
    for kk,v in g.items():
        if kk.startswith("mismatch"): print(f"    {kk}: {v}")

    print("\n========== B. RELATION SELECTION RULE ==========")
    n=max(r["pairs"],1)
    print(f"  ordered pairs with both directions scored : {r['pairs']}")
    print(f"  CURRENT rule emits the TRUE direction     : {r['current_emits_true']} ({r['current_emits_true']/n:.0%})")
    print(f"  CURRENT rule emits the FALSE direction    : {r['current_emits_false']} ({r['current_emits_false']/n:.0%})")
    print(f"  CURRENT rule emits BOTH (contradiction)   : {r['current_emits_BOTH']} ({r['current_emits_BOTH']/n:.0%})")
    print(f"  PROPOSED argmax-per-pair is CORRECT       : {r['argmax_correct']} ({r['argmax_correct']/n:.0%})")
    print(f"  PROPOSED argmax-per-pair is WRONG         : {r['argmax_wrong']} ({r['argmax_wrong']/n:.0%})")
    print("\n  examples (subject, object, truth, p_true, p_false, argmax right?):")
    for x in rows: print(f"    {x[0]:14s} {x[1]:14s} {x[2]:16s} {x[3]:.3f} {x[4]:.3f}  {x[5]}")
    json.dump({"guard":dict(g),"relation":dict(r)},
              open("output/probes/surfacing_probe.json","w"), indent=2)
    c.stop()

if __name__ == "__main__":
    main()
