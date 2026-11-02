"""The question the 38% false-grant rate leaves open.

Granting a wrong colour alongside a right one is only fatal if it breaks the thing the
task needs: telling two same-category objects apart by colour. "Put the two YELLOW
bottles on the toilet" only works if the graph calls the yellow bottle yellow and the
other bottle something else.

So: find frames holding two objects of the SAME category with DIFFERENT true colours,
ask for both colours at once, match the returned boxes back to the two objects, and see
which attribute landed on which instance.

  discriminated : each object got its own colour and not the other's  <- what we need
  swapped       : colours assigned the wrong way round
  collapsed     : both objects got the same colour, or both got both  <- fatal
  incomplete    : one or both got nothing
"""
import os, sys, json, collections
sys.path.insert(0, "rung2")
os.environ.setdefault("NSAP_AGGR_THRES", "0.4")
os.environ["NSAP_LASER_ATTR_REL"] = "1"          # else predict_laser takes a different path
import numpy as np
import ai2thor.controller
from nav_harness import make_nav_planner

RES = 500
SCENES = ([f"FloorPlan{i}" for i in (1,6,10,15,20)] +
          [f"FloorPlan{i}" for i in (401,405,410)] +
          [f"FloorPlan{i}" for i in (201,205,301,305)])
BASIC = {"red":(180,40,40),"green":(50,140,60),"blue":(50,80,180),"yellow":(220,200,60),
         "white":(225,225,225),"black":(40,40,40),"grey":(130,130,130),
         "brown":(120,80,50),"silver":(190,190,195),"orange":(225,140,50)}
def cname(rgb): return min(BASIC, key=lambda k: sum((a-b)**2 for a,b in zip(rgb,BASIC[k])))
def cdist(a,b): return sum((x-y)**2 for x,y in zip(a,b))**0.5

def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=ix*iy; ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def main():
    p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o")
    print(f"  use_laser_attr_rel = {getattr(p,'use_laser_attr_rel',None)} (must be True)\n", flush=True)
    c = ai2thor.controller.Controller(scene=SCENES[0], width=RES, height=RES,
                                      renderInstanceSegmentation=True)
    stats = collections.Counter(); rows = []; pairs_found = 0
    for scene in SCENES:
        c.reset(scene=scene)
        reach = c.step("GetReachablePositions").metadata.get("actionReturn", []) or []
        for k in [i for i in range(0, len(reach), max(1,len(reach)//8))][:8]:
            for rot in (0,90,180,270):
                c.step(action="Teleport", position=reach[k], rotation={"x":0,"y":rot,"z":0},
                       horizon=0, standing=True)
                ev=c.last_event; frame=ev.frame
                det=ev.instance_detections2D or {}; seg=ev.instance_masks or {}
                # group visible objects by category
                by=collections.defaultdict(list)
                for oid,b in det.items():
                    if "|" not in oid: continue
                    if (b[2]-b[0])*(b[3]-b[1]) < 1200: continue
                    m=seg.get(oid)
                    if m is None or m.sum()<250: continue
                    by[oid.split("|")[0].lower()].append((oid,b,tuple(int(x) for x in frame[m].mean(axis=0))))
                for cat, objs in by.items():
                    if len(objs)<2: continue
                    # pick the two most colour-different instances
                    best=None
                    for i in range(len(objs)):
                        for j in range(i+1,len(objs)):
                            d=cdist(objs[i][2],objs[j][2])
                            if best is None or d>best[0]: best=(d,objs[i],objs[j])
                    if best is None or best[0] < 60: continue      # too similar to be a colour task
                    _,(oa,ba,ca),(ob,bb,cb) = best
                    na, nb = cname(ca), cname(cb)
                    if na == nb: continue
                    pairs_found += 1
                    labels, boxes = p.predict_gd(frame, [cat], box_threshold=0.1, text_threshold=0.1)
                    if boxes is None or len(boxes)==0: stats["no_detection"]+=1; continue
                    _, sboxes, _, sattr = p.predict_laser(image=frame, boxes=boxes,
                        target_names=[cat], query_names=[cat,"other"],
                        query_attributes=[na, nb], query_relation_kws=[])
                    if sboxes is None or len(sboxes)==0: stats["not_grounded"]+=1; continue
                    sb=[list(map(int,x)) for x in (sboxes.tolist() if hasattr(sboxes,"tolist") else sboxes)]
                    def attrs_for(gt):
                        bi,bv=-1,0.0
                        for idx,box in enumerate(sb):
                            v=iou(gt,box)
                            if v>bv: bi,bv=idx,v
                        if bi<0 or bv<0.2: return None
                        lst=(sattr[bi] if sattr and bi<len(sattr) else []) or []
                        return {str(x).lower() for x in lst}
                    A,B = attrs_for(ba), attrs_for(bb)
                    if A is None or B is None: stats["unmatched_box"]+=1; continue
                    a_ok = (na in A) and (nb not in A)
                    b_ok = (nb in B) and (na not in B)
                    a_sw = (nb in A) and (na not in A)
                    b_sw = (na in B) and (nb not in B)
                    if a_ok and b_ok: k2="discriminated"
                    elif a_sw and b_sw: k2="swapped"
                    elif not A and not B: k2="incomplete"
                    elif A==B and A: k2="collapsed"
                    else: k2="partial"
                    stats[k2]+=1
                    if len(rows)<16:
                        rows.append((scene,cat,na,nb,sorted(A),sorted(B),k2))
                if pairs_found>60: break
            if pairs_found>60: break
        print(f"{scene}: same-category colour pairs so far = {pairs_found}", flush=True)
        if pairs_found>60: break

    print("\n=========== TWO SAME-CATEGORY OBJECTS, DIFFERENT COLOURS ===========")
    print(f"  qualifying pairs found : {pairs_found}")
    tot=sum(stats[k] for k in ("discriminated","swapped","collapsed","partial","incomplete"))
    for k in ("discriminated","swapped","collapsed","partial","incomplete"):
        print(f"  {k:14s} : {stats[k]:>3} ({stats[k]/max(tot,1):.0%})")
    for k in ("no_detection","not_grounded","unmatched_box"):
        if stats[k]: print(f"  ({k}: {stats[k]} pairs dropped before scoring)")
    print()
    for r in rows:
        print(f"  {r[0]:12s} {r[1]:12s} truth={r[2]:7s}/{r[3]:7s}  got={r[4]} / {r[5]}   -> {r[6]}")
    json.dump(dict(stats), open("output/probes/colour_disambig.json","w"), indent=2)
    c.stop()

if __name__ == "__main__":
    main()
