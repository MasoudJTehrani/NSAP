#!/usr/bin/env python
"""LOCAL (NSAP-only) generator: a pilot-style SceneGraphBench dataset SCENE-MATCHED to the
EB-Navigation scenarios we evaluate. One scenario per exact EB-Nav target instance, framed
from a moderate 'pilot band' (~0.8-1.5m). Reuses the sgbench building blocks so the output
schema is identical to kitchen_pilot / kitchen_approach (manifest.json + scenarios/*), which
the patch-training pipeline reads directly.

  python rung2/gen_ebnav_dataset.py --config rung2/kitchen_ebnav.json --out SceneGraphBench/data/kitchen_ebnav

Kept local (not in the SceneGraphBench repo) per project decision; run from the NSAP root.
"""
import os, sys, json, math, random, argparse

NSAP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(NSAP, "SceneGraphBench"))   # import the sgbench package

import numpy as np                                           # noqa: E402
from sgbench.generator import GeneratorConfig, _make_controller, _save_png   # noqa: E402
from sgbench.scene_graph import build_id_map, extract_frame_graph            # noqa: E402
from sgbench.framing import approach_poses                                   # noqa: E402
from sgbench.vocab import thor_type_to_word                                  # noqa: E402


def find_target(objs, category, object_id, pos):
    """Exact EB-Nav instance by objectId; fall back to the nearest same-category object to
    the EB-Nav position (robust to any THOR id/position drift). Returns (obj, matched_how)."""
    for o in objs:
        if o["objectId"] == object_id:
            return o, "exact"
    cands = [o for o in objs if o["objectType"] == category]
    if not cands:
        return None, "missing"
    best = min(cands, key=lambda o: math.hypot(o["position"]["x"] - pos[0], o["position"]["z"] - pos[1]))
    d = math.hypot(best["position"]["x"] - pos[0], best["position"]["z"] - pos[1])
    return best, f"nearest({d:.2f}m)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rung2/kitchen_ebnav.json")
    ap.add_argument("--out", default="SceneGraphBench/data/kitchen_ebnav")
    ap.add_argument("--restart-every", type=int, default=6)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    fr = cfg["framing"]
    res, min_area, seed = cfg["resolution"], cfg["min_area"], cfg["seed"]
    out = args.out
    os.makedirs(os.path.join(out, "scenarios"), exist_ok=True)
    rng = random.Random(seed)

    # a bare GeneratorConfig just to drive _make_controller (resolution etc.)
    gc = GeneratorConfig(out_dir=out, resolution=res, min_area=min_area, randomize_states=False,
                         randomize_materials=False, seed=seed)

    manifest = {"version": "0.1.0",
                "config": {"style": "ebnav_" + fr["style"], "framing": fr, "resolution": res,
                           "min_area": min_area, "randomize_states": False, "seed": seed},
                "scenes": sorted({t["scene"] for t in cfg["targets"]}), "scenarios": []}

    controller = None
    sidx = 0
    for ti, t in enumerate(cfg["targets"]):
        scene = t["scene"]
        if controller is None or (ti % args.restart_every == 0 and ti > 0):
            if controller is not None:
                controller.stop()
            controller = _make_controller(gc, scene)
        controller.reset(scene=f"FloorPlan{scene}")   # DEFAULT state: no randomize_states

        reach = controller.step("GetReachablePositions").metadata.get("actionReturn", []) or []
        objs = controller.last_event.metadata["objects"]
        id_map = build_id_map(objs)
        obj, how = find_target(objs, t["category"], t["object_id"], t["pos"])
        if obj is None:
            print(f"  SKIP FloorPlan{scene} {t['category']}: not found", flush=True)
            continue
        target_int = id_map[obj["objectId"]]
        poses = approach_poses(obj, reach, fr["dists"], rng,
                               views_per_dist=fr["views_per_dist"], tol=fr["tol"])

        sid_str = t.get("scenario_id") or f"scenario_{sidx:05d}"   # explicit id (in-place regen) or sequential
        sdir = os.path.join(out, "scenarios", sid_str)
        os.makedirs(os.path.join(sdir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(sdir, "instance"), exist_ok=True)
        frames_rec = []
        for (pos, yaw, hor) in poses:
            ev = controller.step(action="TeleportFull", position=pos,
                                 rotation=dict(x=0, y=yaw, z=0), horizon=hor, standing=True)
            if not ev.metadata.get("lastActionSuccess", True):
                continue
            graph = extract_frame_graph(ev, id_map, min_area)
            if not any(o["id"] == target_int for o in graph["objects"]):
                continue   # target not adequately visible from this viewpoint
            fn = f"{len(frames_rec):03d}.png"
            _save_png(ev.frame, os.path.join(sdir, "frames", fn))
            _save_png(ev.instance_segmentation_frame, os.path.join(sdir, "instance", fn))
            frames_rec.append({"frame": f"frames/{fn}", "instance_map": f"instance/{fn}", **graph})

        if not frames_rec:
            print(f"  SKIP FloorPlan{scene} {t['category']}: no frames with the target visible", flush=True)
            continue
        meta = {"scenario_id": sid_str, "scene": f"FloorPlan{scene}", "seed": seed,
                "target": {"id": target_int, "category": thor_type_to_word(obj["objectType"]),
                           "thor_id": obj["objectId"], "thor_type": obj["objectType"],
                           "eb_nav_object_id": t["object_id"], "matched": how},
                "n_frames": len(frames_rec)}
        json.dump(meta, open(os.path.join(sdir, "meta.json"), "w"), indent=2)
        json.dump({"scenario_id": meta["scenario_id"], "scene": meta["scene"],
                   "target": meta["target"], "frames": frames_rec},
                  open(os.path.join(sdir, "scene_graph.json"), "w"), indent=2)
        manifest["scenarios"].append({"scenario_id": meta["scenario_id"], "scene": meta["scene"],
                                      "target_category": meta["target"]["category"],
                                      "n_frames": len(frames_rec)})
        json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)  # incremental
        print(f"  {sid_str}  FloorPlan{scene:<2}  {obj['objectType']:<14} "
              f"{len(frames_rec):>2} frames  match={how}", flush=True)
        sidx += 1

    if controller is not None:
        controller.stop()
    print(f"\nDone: {sidx}/{len(cfg['targets'])} scenarios -> {out}", flush=True)


if __name__ == "__main__":
    main()
