#!/usr/bin/env python
"""Generate an ATTRIBUTE dataset for a TOGGLE attribute (e.g. a TV that is 'displaying').
Unlike gen_multi_object (which spawns duplicates + open/close), this toggles the naturally
present object ON and records close-framed views with the target attribute GT-present. VINE
grounds open/closed at chance, but reads an active TV screen confidently as 'displaying'
(0.782) -- a real signal to attack. The target object's attribute list is overridden to the
requested word(s) when the sim reports isToggled, since AI2-THOR has no 'displaying' label.

  python rung2/gen_toggle_attr.py --config rung2/attribute_tv.json --out SceneGraphBench/data/attribute_tv
"""
import os, sys, json, random, argparse
NSAP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(NSAP, "SceneGraphBench"))
import numpy as np                                                          # noqa: E402
from sgbench.generator import GeneratorConfig, _make_controller, _save_png  # noqa: E402
from sgbench.scene_graph import build_id_map, extract_frame_graph           # noqa: E402
from sgbench.framing import approach_poses                                  # noqa: E402
from sgbench.vocab import thor_type_to_word                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    cat = cfg["target_category"]; attrs = cfg["attribute_words"]; primary = attrs[0]
    fr = cfg["framing"]; res, min_area, seed = cfg["resolution"], cfg["min_area"], cfg["seed"]
    out = args.out
    os.makedirs(os.path.join(out, "scenarios"), exist_ok=True)
    rng = random.Random(seed)
    gc = GeneratorConfig(out_dir=out, resolution=res, min_area=min_area, randomize_states=False, seed=seed)

    manifest = {"version": "0.1.0", "config": {"mode": "attribute", "target_category": cat, "predicate": primary},
                "scenes": cfg["scenes"], "scenarios": []}
    controller, sidx = None, 0
    for si, scene in enumerate(cfg["scenes"]):
        if controller is None or (si % 6 == 0 and si > 0):
            if controller is not None:
                controller.stop()
            controller = _make_controller(gc, scene)
        controller.reset(scene=f"FloorPlan{scene}")
        objs0 = [o for o in controller.last_event.metadata["objects"] if o["objectType"] == cat]
        if not objs0:
            print(f"  SKIP FloorPlan{scene}: no {cat}", flush=True); continue
        tgt0 = objs0[0]
        # set the POSITIVE state: toggle ON
        controller.step(action="ToggleObjectOn", objectId=tgt0["objectId"], forceAction=True)
        tgt = next((o for o in controller.last_event.metadata["objects"] if o["objectId"] == tgt0["objectId"]), None)
        if not tgt or not tgt.get("isToggled"):
            print(f"  SKIP FloorPlan{scene}: {cat} would not toggle on", flush=True); continue

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
            # override the target's attribute list with the requested toggle word(s)
            tobj["attributes"] = sorted(set((tobj.get("attributes") or []) + attrs))
            fn = f"{len(frames_rec):03d}.png"
            _save_png(ev.frame, os.path.join(sdir, "frames", fn))
            _save_png(ev.instance_segmentation_frame, os.path.join(sdir, "instance", fn))
            frames_rec.append({"frame": f"frames/{fn}", "instance_map": f"instance/{fn}", **graph})

        if not frames_rec:
            print(f"  SKIP FloorPlan{scene}: no framed views", flush=True); continue
        meta = {"scenario_id": sid_str, "scene": f"FloorPlan{scene}", "seed": seed,
                "target": {"id": target_int, "category": thor_type_to_word(tgt["objectType"]),
                           "thor_id": tgt["objectId"], "thor_type": tgt["objectType"],
                           "mode": "attribute", "predicate": primary, "staged": "toggled_on"},
                "n_frames": len(frames_rec)}
        json.dump(meta, open(os.path.join(sdir, "meta.json"), "w"), indent=2)
        json.dump({"scenario_id": sid_str, "scene": meta["scene"], "target": meta["target"], "frames": frames_rec},
                  open(os.path.join(sdir, "scene_graph.json"), "w"), indent=2)
        manifest["scenarios"].append({"scenario_id": sid_str, "scene": meta["scene"],
                                      "target_category": meta["target"]["category"], "n_frames": len(frames_rec)})
        json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)
        print(f"  {sid_str}  FloorPlan{scene:<3} {cat:<12} {len(frames_rec):>2} frames  (toggled on)", flush=True)
        sidx += 1

    if controller is not None:
        controller.stop()
    print(f"\nDone: {sidx} scenarios -> {out}", flush=True)


if __name__ == "__main__":
    main()
