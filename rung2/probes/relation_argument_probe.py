"""Does the relation head pick the correct OBJECT ARGUMENT?

The left/right probe tested choosing between relation keywords. The pot task tests
something different: for a FIXED relation ("on"), does the head pick the right receptacle,
counter rather than stove? That is choosing among candidate objects.

Ground truth comes from AI2-THOR's own `parentReceptacles`, so we know exactly what each
object is resting on. We then read the raw pair scores for "on" and ask whether the true
receptacle outranks the visible distractors.
"""
import os, sys, collections, json
sys.path.insert(0, "rung2")
os.environ.setdefault("NSAP_AGGR_THRES", "0.4")
os.environ["NSAP_LASER_ATTR_REL"] = "1"
import ai2thor.controller
from nav_harness import make_nav_planner

RES = 500
SCENES = ["FloorPlan1","FloorPlan6","FloorPlan10","FloorPlan15","FloorPlan20",
          "FloorPlan401","FloorPlan405"]

def main():
    p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o")
    print(f"  use_laser_attr_rel = {getattr(p,'use_laser_attr_rel',None)}\n", flush=True)
    c = ai2thor.controller.Controller(scene=SCENES[0], width=RES, height=RES,
                                      renderInstanceSegmentation=True)
    st = collections.Counter(); ranks = collections.Counter(); rows = []
    for scene in SCENES:
        c.reset(scene=scene)
        reach = c.step("GetReachablePositions").metadata.get("actionReturn", []) or []
        for k in [i for i in range(0, len(reach), max(1,len(reach)//6))][:6]:
            for rot in (0,90,180,270):
                c.step(action="Teleport", position=reach[k], rotation={"x":0,"y":rot,"z":0},
                       horizon=0, standing=True)
                ev=c.last_event; frame=ev.frame
                det=ev.instance_detections2D or {}
                meta={o["objectId"]: o for o in ev.metadata["objects"]}
                vis={}
                for oid,b in det.items():
                    if "|" not in oid: continue
                    if (b[2]-b[0])*(b[3]-b[1])<1500: continue
                    vis[oid]=b
                if len(vis)<3: continue
                vis_types={oid.split("|")[0].lower() for oid in vis}
                # objects whose TRUE receptacle is also visible
                targets=[]
                for oid in vis:
                    o=meta.get(oid)
                    if not o: continue
                    pr=o.get("parentReceptacles") or []
                    if not pr: continue
                    rt=pr[0].split("|")[0].lower()
                    if rt in vis_types and rt != oid.split("|")[0].lower():
                        targets.append((oid.split("|")[0].lower(), rt))
                if not targets: continue
                names=sorted(vis_types)[:8]
                labels, boxes = p.predict_gd(frame, names, box_threshold=0.1, text_threshold=0.1)
                if boxes is None or len(boxes)==0: continue
                p._nsap_rel_scored={}; p._nsap_rel_names={}
                p.predict_laser(image=frame, boxes=boxes, target_names=names,
                                query_names=names+["other"], query_attributes=[],
                                query_relation_kws=["on"])
                scored=getattr(p,"_nsap_rel_scored",{}) or {}
                nmap=getattr(p,"_nsap_rel_names",{}) or {}
                if not scored: continue
                for tname, true_rt in targets:
                    cand={}
                    for (pair, rel), prob in scored.items():
                        if rel != "on": continue
                        soid, ooid = pair
                        if nmap.get(soid) != tname: continue
                        on2 = nmap.get(ooid)
                        if not on2 or on2 == tname: continue
                        cand[on2] = max(cand.get(on2, -1), prob)
                    if true_rt not in cand or len(cand) < 2: continue
                    st["cases"] += 1
                    order = sorted(cand.items(), key=lambda kv: -kv[1])
                    rank = [n for n,_ in order].index(true_rt) + 1
                    ranks[rank if rank<=3 else 4] += 1
                    if rank == 1: st["true_receptacle_is_argmax"] += 1
                    # does it even beat its own negation?
                    if len(rows) < 14:
                        rows.append((scene, tname, true_rt, [(n, round(v,3)) for n,v in order[:4]], rank))
        print(f"{scene}: cases={st['cases']}", flush=True)

    n=max(st["cases"],1)
    print("\n========== DOES THE HEAD PICK THE RIGHT RECEPTACLE? ==========")
    print(f"  cases (true receptacle visible, >=2 candidates) : {st['cases']}")
    print(f"  true receptacle ranked #1 (argmax)              : {st['true_receptacle_is_argmax']} ({st['true_receptacle_is_argmax']/n:.0%})")
    for rk in (1,2,3,4):
        lbl = f"rank {rk}" if rk<4 else "rank 4+"
        print(f"    {lbl:8s}: {ranks[rk]:>4} ({ranks[rk]/n:.0%})")
    print("\n  examples (object, TRUE receptacle, top scored candidates, rank of truth):")
    for r in rows:
        print(f"    {r[0]:12s} {r[1]:13s} true={r[2]:13s} {r[3]}  rank={r[4]}")
    json.dump({"stats":dict(st),"ranks":dict(ranks)},
              open("output/probes/rel_argument.json","w"), indent=2)
    c.stop()

if __name__ == "__main__":
    main()
