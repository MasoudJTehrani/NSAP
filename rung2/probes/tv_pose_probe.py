"""Find start poses where the television is VISIBLE but too small for the planner to
judge its screen state.

The attribute attack fails because gpt-4o can see a lit screen and never needs the
symbol. The category attack works on visual_appearance for the mirror-image reason.
So we want poses where:
  - the TV is detected (a bbox exists), else the patch never injects and SGClip has
    nothing to ground;
  - the TV's bbox is SMALL in the frame, so the screen state is not readable by eye;
  - the TV is reachable inside the 20-step budget (0.25 m per move).

Reports, per scene, the CURRENT pose and the best far candidates.
"""
import json, math, os, sys
sys.path.insert(0, "rung2")
import ai2thor.controller

SCENES = [f"FloorPlan{200+i}" for i in range(1, 11)]
RES = 500
OUT = "output/probes/tv_poses.json"

cur = {}
raw = json.load(open("rung2/staged_tasks.json"))
for t in (raw["tasks"] if isinstance(raw, dict) else raw):
    if t.get("mode") == "attribute":
        cur[t["scene"]] = t["agentPose"]

c = ai2thor.controller.Controller(scene=SCENES[0], width=RES, height=RES,
                                  renderInstanceSegmentation=True)
print("controller up", flush=True)

def tv_bbox(tv_id):
    d = c.last_event.instance_detections2D or {}
    return d.get(tv_id)

def probe(pos, rot, horizon, tv_id):
    ev = c.step(action="Teleport", position=pos, rotation={"x":0,"y":rot,"z":0},
                horizon=horizon, standing=True)
    if not ev.metadata.get("lastActionSuccess"):
        return None
    b = tv_bbox(tv_id)
    if b is None:
        return None
    w, h = b[2]-b[0], b[3]-b[1]
    return {"area": int(w*h), "w": int(w), "h": int(h),
            "frac": round(w*h/(RES*RES), 5)}

results = {}
for scene in SCENES:
    c.reset(scene=scene)
    tvs = [o for o in c.last_event.metadata["objects"] if o["objectType"] == "Television"]
    if not tvs:
        print(f"{scene}: no TV"); continue
    tv = tvs[0]
    c.step(action="ToggleObjectOn", objectId=tv["objectId"], forceAction=True)
    tvpos = tv["position"]; tvid = tv["objectId"]
    reach = c.step("GetReachablePositions").metadata.get("actionReturn", []) or []

    # current pose, for reference
    cp = cur.get(scene)
    cur_info = None
    if cp:
        r = probe(cp["position"], cp["rotation"], cp.get("horizon", 0), tvid)
        d = math.hypot(cp["position"]["x"]-tvpos["x"], cp["position"]["z"]-tvpos["z"])
        cur_info = {"dist": round(d,2), **(r or {"area": 0, "note": "TV NOT VISIBLE"})}

    cands = []
    for p in reach:
        d = math.hypot(p["x"]-tvpos["x"], p["z"]-tvpos["z"])
        if not (3.0 <= d <= 6.0):        # far, but reachable in 20 x 0.25 m moves
            continue
        yaw = (math.degrees(math.atan2(tvpos["x"]-p["x"], tvpos["z"]-p["z"])) + 360) % 360
        rot = round(yaw/90.0)*90 % 360   # snap: the skill set rotates in 90 deg steps
        for hz in (0, -15):
            r = probe(p, rot, hz, tvid)
            if r and r["area"] > 150:    # detectable at all
                cands.append({"position": p, "rotation": rot, "horizon": hz,
                              "dist": round(d,2), **r})
                break
    cands.sort(key=lambda x: (x["area"], -x["dist"]))
    results[scene] = {"tv_id": tvid, "current": cur_info, "n_candidates": len(cands),
                      "smallest": cands[:5], "farthest": sorted(cands, key=lambda x:-x["dist"])[:3]}
    cu = f"dist={cur_info['dist']}m area={cur_info['area']}px" if cur_info else "n/a"
    bs = cands[0] if cands else None
    print(f"{scene}: current {cu} | {len(cands)} far candidates"
          + (f" | smallest {bs['area']}px at {bs['dist']}m" if bs else ""), flush=True)

json.dump(results, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
c.stop()
