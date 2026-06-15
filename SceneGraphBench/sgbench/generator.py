"""
The dataset generator.

For every target-category object in every scene, it navigates the agent to face
the object from a few jittered viewpoints, renders the robot's-eye frame, and
writes the simulator's ground-truth scene graph for that frame. One target object
across its viewpoints = one "scenario" (a short clip). Output schema: docs/FORMAT.md.
"""

import os
import json
import random
from dataclasses import dataclass, field, asdict
from typing import List, Set

import numpy as np

from .framing import face_pose, jitter
from .scene_graph import build_id_map, extract_frame_graph
from .vocab import TARGET_CATEGORIES, thor_type_to_word, ATTRIBUTE_FLAGS


@dataclass
class GeneratorConfig:
    out_dir: str
    scenes: List[int] = field(default_factory=lambda: list(range(1, 6)))   # FloorPlan kitchens
    target_categories: Set[str] = field(default_factory=lambda: set(TARGET_CATEGORIES))
    viewpoints: int = 4                # frames per target object (1 facing + jitters)
    resolution: int = 600
    min_area: int = 600                # min target mask px to record a viewpoint
    randomize_states: bool = True      # toggle openable/toggleable target states for attribute variety
    randomize_materials: bool = False  # RandomizeMaterials/Lighting per scene (real-render EoT)
    max_targets_per_scene: int = 0     # 0 = all
    seed: int = 0


def _save_png(arr, path):
    from PIL import Image
    Image.fromarray(np.asarray(arr).astype(np.uint8)).save(path)


def _stage_state(controller, obj, rng):
    """Randomly set a target's openable/toggleable state, for attribute coverage."""
    oid = obj['objectId']
    try:
        if obj.get('openable'):
            act = 'OpenObject' if rng.random() < 0.5 else 'CloseObject'
            controller.step(action=act, objectId=oid, forceAction=True)
        if obj.get('toggleable'):
            act = 'ToggleObjectOn' if rng.random() < 0.5 else 'ToggleObjectOff'
            controller.step(action=act, objectId=oid, forceAction=True)
    except Exception:
        pass


def generate(config: GeneratorConfig) -> dict:
    """Generate the dataset described by `config`. Returns the manifest dict."""
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    rng = random.Random(config.seed)
    os.makedirs(os.path.join(config.out_dir, "scenarios"), exist_ok=True)

    controller = Controller(platform=CloudRendering, scene=f"FloorPlan{config.scenes[0]}",
                            width=config.resolution, height=config.resolution,
                            renderInstanceSegmentation=True)

    cfg_dict = asdict(config)
    cfg_dict["target_categories"] = sorted(cfg_dict["target_categories"])   # set -> list for JSON
    manifest = {"version": "0.1.0", "config": cfg_dict,
                "scenes": config.scenes, "scenarios": []}
    sidx = 0

    for scene in config.scenes:
        controller.reset(scene=f"FloorPlan{scene}")
        if config.randomize_materials:
            for act in ("RandomizeMaterials", "RandomizeLighting"):
                try:
                    controller.step(action=act)
                except Exception:
                    pass

        reach = controller.step("GetReachablePositions").metadata.get("actionReturn", []) or []
        objs = controller.last_event.metadata['objects']
        id_map = build_id_map(objs)
        targets = [o for o in objs if o['objectType'] in config.target_categories]
        rng.shuffle(targets)
        if config.max_targets_per_scene:
            targets = targets[:config.max_targets_per_scene]

        for obj in targets:
            if config.randomize_states:
                _stage_state(controller, obj, rng)
                obj = next((o for o in controller.last_event.metadata['objects']
                            if o['objectId'] == obj['objectId']), obj)   # refresh state

            pose = face_pose(obj, reach)
            if pose is None:
                continue
            base_pos, base_yaw, base_hor = pose
            target_int = id_map[obj['objectId']]

            frames_rec = []
            for v in range(config.viewpoints):
                yaw, hor = (base_yaw, base_hor) if v == 0 else jitter(base_yaw, base_hor, rng)
                ev = controller.step(action="TeleportFull", position=base_pos,
                                     rotation=dict(x=0, y=yaw, z=0), horizon=hor, standing=True)
                if not ev.metadata.get("lastActionSuccess", True):
                    continue
                graph = extract_frame_graph(ev, id_map, config.min_area)
                if not any(o['id'] == target_int for o in graph['objects']):
                    continue   # target not adequately visible from this viewpoint

                sdir = os.path.join(config.out_dir, "scenarios", f"scenario_{sidx:05d}")
                os.makedirs(os.path.join(sdir, "frames"), exist_ok=True)
                os.makedirs(os.path.join(sdir, "instance"), exist_ok=True)
                fname = f"{len(frames_rec):03d}.png"
                _save_png(ev.frame, os.path.join(sdir, "frames", fname))
                _save_png(ev.instance_segmentation_frame, os.path.join(sdir, "instance", fname))
                frames_rec.append({"frame": f"frames/{fname}", "instance_map": f"instance/{fname}",
                                   **graph})

            if not frames_rec:
                continue

            sdir = os.path.join(config.out_dir, "scenarios", f"scenario_{sidx:05d}")
            meta = {"scenario_id": f"scenario_{sidx:05d}", "scene": f"FloorPlan{scene}",
                    "seed": config.seed,
                    "target": {"id": target_int, "category": thor_type_to_word(obj['objectType']),
                               "thor_id": obj['objectId'], "thor_type": obj['objectType']},
                    "n_frames": len(frames_rec)}
            json.dump(meta, open(os.path.join(sdir, "meta.json"), "w"), indent=2)
            json.dump({"scenario_id": meta["scenario_id"], "scene": meta["scene"],
                       "target": meta["target"], "frames": frames_rec},
                      open(os.path.join(sdir, "scene_graph.json"), "w"), indent=2)
            manifest["scenarios"].append({"scenario_id": meta["scenario_id"],
                                          "scene": meta["scene"],
                                          "target_category": meta["target"]["category"],
                                          "n_frames": len(frames_rec)})
            sidx += 1
            print(f"  scenario_{sidx-1:05d}  FloorPlan{scene}  {obj['objectType']:<14}  {len(frames_rec)} frames")

    controller.stop()

    # dataset-level files
    attr_words = sorted({w for (_cap, w_true, w_false) in ATTRIBUTE_FLAGS.values()
                         for w in (w_true, w_false) if w})
    json.dump({"target_categories": sorted(config.target_categories),
               "attribute_words": attr_words,
               "relations": ["on", "in", "near"]},
              open(os.path.join(config.out_dir, "vocab.json"), "w"), indent=2)
    json.dump(manifest, open(os.path.join(config.out_dir, "manifest.json"), "w"), indent=2)
    print(f"\nDone: {sidx} scenarios → {config.out_dir}/")
    return manifest
