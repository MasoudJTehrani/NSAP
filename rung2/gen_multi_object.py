#!/usr/bin/env python
"""Generate SceneGraphBench training data for the RELATION ('on') and ATTRIBUTE ('open')
attacks. Unlike the category generator, each scene is STAGED: spawn a duplicate of the
target category (so there are two), set states / read receptacles, and pick the CORRECT
target -- the pot ON a CounterTop, or the OPEN laptop. Then render close-framed views of
that target and record the simulator GT scene graph (which includes the (pot,on,counter)
relation / (laptop,open) attribute the patch trainer attacks).

  python rung2/gen_multi_object.py --config rung2/relation_pot.json  --out SceneGraphBench/data/relation_pot
  python rung2/gen_multi_object.py --config rung2/attribute_laptop.json --out SceneGraphBench/data/attribute_laptop
"""
import os, sys, json, math, random, argparse
NSAP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(NSAP, "SceneGraphBench"))
import numpy as np                                                          # noqa: E402
from sgbench.generator import GeneratorConfig, _make_controller, _save_png  # noqa: E402
from sgbench.scene_graph import build_id_map, extract_frame_graph           # noqa: E402
from sgbench.framing import approach_poses                                  # noqa: E402
from sgbench.vocab import thor_type_to_word                                 # noqa: E402


def recep(o):
    return [r.split("|")[0] for r in (o.get("parentReceptacles") or [])]


def stage_scene(c, cfg, scene):
    """Spawn 2 of the target category + set states; return (target_obj, note) or (None, why)."""
    cat = cfg["target_category"]
    for seed in range(cfg.get("spawn_seed", 3), cfg.get("spawn_seed", 3) + 8):   # retry seeds
        c.reset(scene=f"FloorPlan{scene}")
        ev = c.step(action="InitialRandomSpawn", randomSeed=seed, forceVisible=True,
                    numPlacementAttempts=12, numDuplicatesOfType=[{"objectType": cat, "count": 2}])
        if not ev.metadata.get("lastActionSuccess"):
            continue
        objs = [o for o in c.last_event.metadata["objects"] if o["objectType"] == cat]
        if len(objs) < 2:
            continue
        if cfg["mode"] == "attribute":
            # open the first, close the second -> target = the OPEN one
            c.step(action="OpenObject", objectId=objs[0]["objectId"], forceAction=True)
            c.step(action="CloseObject", objectId=objs[1]["objectId"], forceAction=True)
            tgt = next((o for o in c.last_event.metadata["objects"]
                        if o["objectId"] == objs[0]["objectId"]), None)
            if tgt and tgt.get("isOpen"):
                return tgt, f"open@seed{seed}"
        else:  # relation: need one ON a CounterTop and one ON a StoveBurner (disambiguation)
            want = set(cfg["relation_object_types"]); distract = set(cfg.get("distractor_object_types", []))
            on_want = [o for o in objs if want & set(recep(o))]
            on_dist = [o for o in objs if distract & set(recep(o))]
            if on_want and on_dist:
                return on_want[0], f"on_{list(want)[0]}@seed{seed}"
    return None, "no valid staging"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    fr = cfg["framing"]; res, min_area, seed = cfg["resolution"], cfg["min_area"], cfg["seed"]
    out = args.out
    os.makedirs(os.path.join(out, "scenarios"), exist_ok=True)
    rng = random.Random(seed)
    gc = GeneratorConfig(out_dir=out, resolution=res, min_area=min_area, randomize_states=False, seed=seed)

    manifest = {"version": "0.1.0", "config": {k: cfg[k] for k in ("mode", "target_category", "predicate")},
                "scenes": cfg["scenes"], "scenarios": []}
    controller, sidx = None, 0
    for si, scene in enumerate(cfg["scenes"]):
        if controller is None or (si % 6 == 0 and si > 0):
            if controller is not None:
                controller.stop()
            controller = _make_controller(gc, scene)
        tgt, note = stage_scene(controller, cfg, scene)
        if tgt is None:
            print(f"  SKIP FloorPlan{scene}: {note}", flush=True); continue

        objs = controller.last_event.metadata["objects"]
        id_map = build_id_map(objs)
        reach = controller.step("GetReachablePositions").metadata.get("actionReturn", []) or []
        target_int = id_map[tgt["objectId"]]
        poses = approach_poses(tgt, reach, fr["dists"], rng, views_per_dist=fr["views_per_dist"], tol=fr["tol"])

        sid_str = f"scenario_{sidx:05d}"
        sdir = os.path.join(out, "scenarios", sid_str)
        os.makedirs(os.path.join(sdir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(sdir, "instance"), exist_ok=True)
        frames_rec = []
        for (pos, yaw, hor) in poses:
            ev = controller.step(action="TeleportFull", position=pos, rotation=dict(x=0, y=yaw, z=0),
                                 horizon=hor, standing=True)
            if not ev.metadata.get("lastActionSuccess", True):
                continue
            graph = extract_frame_graph(ev, id_map, min_area)
            tobj = next((o for o in graph["objects"] if o["id"] == target_int), None)
            if tobj is None:
                continue
            # require the attacked predicate to actually be GT-present on the target in this frame
            if cfg["mode"] == "attribute":
                if not any(cfg["predicate"] in a.lower() for a in tobj.get("attributes", [])):
                    continue
            else:
                if not any(r["subject"] == target_int and cfg["predicate"] in r["predicate"].lower()
                           for r in graph["relations"]):
                    continue
            fn = f"{len(frames_rec):03d}.png"
            _save_png(ev.frame, os.path.join(sdir, "frames", fn))
            _save_png(ev.instance_segmentation_frame, os.path.join(sdir, "instance", fn))
            frames_rec.append({"frame": f"frames/{fn}", "instance_map": f"instance/{fn}", **graph})

        if not frames_rec:
            print(f"  SKIP FloorPlan{scene}: no frames with GT '{cfg['predicate']}' on target", flush=True); continue
        meta = {"scenario_id": sid_str, "scene": f"FloorPlan{scene}", "seed": seed,
                "target": {"id": target_int, "category": thor_type_to_word(tgt["objectType"]),
                           "thor_id": tgt["objectId"], "thor_type": tgt["objectType"],
                           "mode": cfg["mode"], "predicate": cfg["predicate"], "staged": note},
                "n_frames": len(frames_rec)}
        json.dump(meta, open(os.path.join(sdir, "meta.json"), "w"), indent=2)
        json.dump({"scenario_id": sid_str, "scene": meta["scene"], "target": meta["target"], "frames": frames_rec},
                  open(os.path.join(sdir, "scene_graph.json"), "w"), indent=2)
        manifest["scenarios"].append({"scenario_id": sid_str, "scene": meta["scene"],
                                      "target_category": meta["target"]["category"], "n_frames": len(frames_rec)})
        json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)
        print(f"  {sid_str}  FloorPlan{scene:<3} {tgt['objectType']:<8} {len(frames_rec):>2} frames  ({note})", flush=True)
        sidx += 1

    if controller is not None:
        controller.stop()
    print(f"\nDone: {sidx} scenarios -> {out}", flush=True)


if __name__ == "__main__":
    main()
