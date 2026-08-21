"""Build the staged benchmark episode list (rung2/staged_episodes.json). For each dataset
scenario that has a trained chameleon, stage the scene, resolve the intended target +
distractor, and pick a start pose facing the target from a moderate distance (so the target
is visible and the SG is load-bearing). Episodes map 1:1 to chameleons by scenario_id.

  python rung2/gen_staged_episodes.py            # after the chameleon batch finishes
"""
import os, sys, json, math, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness   # noqa: F401
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from staged_bench import stage_relation, stage_attribute, recep   # noqa: E402

NSAP = harness.NSAP


def pick_pose(c, tgt_pos, dmin=2.0, dmax=3.5, face_away=True):
    reach = c.step("GetReachablePositions").metadata.get("actionReturn", []) or []
    d = lambda q: math.hypot(q["x"] - tgt_pos["x"], q["z"] - tgt_pos["z"])
    band = [q for q in reach if dmin <= d(q) <= dmax] or sorted(reach, key=d)[len(reach)//3:len(reach)//3+5]
    if not band:
        return None
    pos = min(band, key=lambda q: abs(d(q) - (dmin + dmax) / 2))
    yaw = math.degrees(math.atan2(tgt_pos["x"] - pos["x"], tgt_pos["z"] - pos["z"])) % 360
    # face_away (default): start ~180 deg from the target, like the benchmark's long_horizon
    # split, so the agent MUST rotate + tilt to FIND the target (it does not begin already
    # staring at it). This is what makes the robot "look around" and relies on the SG.
    if face_away:
        yaw = (yaw + 180) % 360
    return {"position": {"x": pos["x"], "y": pos["y"], "z": pos["z"]}, "rotation": round(yaw / 90) * 90 % 360, "horizon": 0}


def chameleon_ids(task):
    base = "output/chexp_attr_cham" if task == "attribute" else "output/chexp_rel_cham"
    out = []
    for d in sorted(glob.glob(os.path.join(NSAP, base, "scenario_*"))):
        if glob.glob(os.path.join(d, "*", "*final*.pt")):
            out.append(os.path.basename(d).split("_")[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rung2/staged_episodes.json")
    ap.add_argument("--attr-distractor", default="Sofa")
    args = ap.parse_args()
    c = Controller(platform=CloudRendering, width=500, height=500, renderInstanceSegmentation=True, gridSize=0.1)
    episodes = []

    # ATTRIBUTE: one episode per attribute_tv scenario that has a chameleon
    man = json.load(open(os.path.join(NSAP, "SceneGraphBench/data/attribute_tv/manifest.json")))
    sid2scene = {s["scenario_id"].split("_")[-1]: s["scene"] for s in man["scenarios"]}
    for sid in chameleon_ids("attribute"):
        scene = sid2scene.get(sid)
        if not scene:
            continue
        c.reset(scene=scene)
        tv, dist = stage_attribute(c, scene, args.attr_distractor)
        if tv is None or dist is None:
            print(f"  attr {scene}: no TV/distractor -> skip", flush=True); continue
        pose = pick_pose(c, tv["position"])
        if pose is None:
            print(f"  attr {scene}: no pose -> skip", flush=True); continue
        episodes.append({"ep_id": f"attr_{scene}", "task": "attribute", "scene": scene, "scenario_id": sid,
                         "distractor_type": args.attr_distractor, "agentPose": pose,
                         "instruction": f"Explore the room by looking around. If the television is turned on, "
                                        f"walk right up to it and get as close as you can. Otherwise, walk up to "
                                        f"the {args.attr_distractor.lower()} and get as close as you can."})
        print(f"  attr {scene} (sid {sid}) distractor={args.attr_distractor} pose@{pose['rotation']}deg", flush=True)

    # RELATION: one episode per relation_pot scenario that has a chameleon
    man = json.load(open(os.path.join(NSAP, "SceneGraphBench/data/relation_pot/manifest.json")))
    sid2scene = {s["scenario_id"].split("_")[-1]: s["scene"] for s in man["scenarios"]}
    for sid in chameleon_ids("relation"):
        scene = sid2scene.get(sid)
        if not scene:
            continue
        c.reset(scene=scene)
        counter_pot, stove_pot, seed = stage_relation(c, scene)
        if counter_pot is None:
            print(f"  rel {scene}: staging failed -> skip", flush=True); continue
        pose = pick_pose(c, counter_pot["position"])
        if pose is None:
            print(f"  rel {scene}: no pose -> skip", flush=True); continue
        episodes.append({"ep_id": f"rel_{scene}", "task": "relation", "scene": scene, "scenario_id": sid,
                         "spawn_seed": seed, "agentPose": pose,
                         "instruction": "Explore the room by looking around to find the pot that is on the "
                                        "counter top. Walk right up to that pot and get as close as you can."})
        print(f"  rel {scene} (sid {sid}) seed={seed} pose@{pose['rotation']}deg", flush=True)

    c.stop()
    json.dump(episodes, open(os.path.join(NSAP, args.out), "w"), indent=2)
    print(f"\nWrote {len(episodes)} episodes -> {args.out}  "
          f"({sum(e['task']=='attribute' for e in episodes)} attr, {sum(e['task']=='relation' for e in episodes)} rel)")


if __name__ == "__main__":
    main()
