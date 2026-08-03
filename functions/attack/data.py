"""
Frame + mask helpers shared across NSAP training and evaluation.

NSAP's data source is the self-contained SceneGraphBench (AI2-THOR) sim dataset
— see functions/attack/sim_data.py::SimAttackDataset, which stores each frame as
a PNG and carries a ground-truth scene graph. This module keeps only the small
helpers that operate on those instances / their masks:

  - load_frame        : lazy-load an instance's PNG frame as a float32 RGB tensor
  - object_areas      : per-object max mask area across a clip's frames
  - largest_object_id : pick the largest-area object among candidate obj_ids

The former YouTube/ESCA video ingestion (EscaAttackDataset, cv2 video decoding,
per-clip mask .pkl parsing, video download/split helpers) has been removed —
the sim dataset replaces it entirely.
"""
import torch
import numpy as np
from functools import lru_cache
from typing import Dict, Optional


@lru_cache(maxsize=128)
def _cached_png(path: str):
    """Load a single PNG frame (sim datasets store frames as PNGs, not videos).

    Bounded LRU cache (most-recently-used frames only): repeated sampling across
    training steps reuses the decoded array instead of re-reading from disk, while
    the bound keeps CPU-RAM within budget (frames are still loaded lazily, not all
    at once). Callers copy via .astype(...), so the cached array is never mutated."""
    from PIL import Image
    return np.array(Image.open(path).convert('RGB'))   # HxWx3 uint8 RGB


def load_frame(inst: dict) -> torch.Tensor:
    """Lazy-load the frame for an instance as a float32 RGB tensor [H,W,3].

    The instance carries its sim PNG frame path in inst['_png_path']; it is
    decoded to RGB [H,W,3] float32 in 0-255.
    """
    png = inst.get('_png_path')
    if not png:
        raise ValueError(
            "load_frame: instance has no '_png_path'. NSAP loads frames only from "
            "the sim (SceneGraphBench) dataset; video-backed instances are no "
            "longer supported."
        )
    return torch.from_numpy(_cached_png(png).astype(np.float32))


def object_areas(masks: Dict[int, Dict[int, torch.Tensor]]) -> Dict[int, int]:
    """Max mask area (pixel count) per obj_id across all frames of a clip."""
    areas: Dict[int, int] = {}
    for obj_masks in masks.values():
        for obj_id, m in obj_masks.items():
            a = int(m.sum().item())
            if a > areas.get(obj_id, 0):
                areas[obj_id] = a
    return areas


def largest_object_id(masks: Dict[int, Dict[int, torch.Tensor]], candidates) -> Optional[int]:
    """
    Return the candidate obj_id with the largest mask area, or None.

    The mask set is heavily over-segmented (SAM2/GroundingDINO emit dozens of
    objects per clip, many of them noise specks). There is no ground-truth
    obj_id->category map, so the target is chosen as the largest-area object among
    those VINE classifies as the target category — i.e. the actual subject, not a
    fragment. Used by both training and eval so they pick the SAME object.
    """
    areas = object_areas(masks)
    best, best_area = None, 0
    for obj_id in candidates:
        a = areas.get(obj_id, 0)
        if a > best_area:
            best, best_area = obj_id, a
    return best
