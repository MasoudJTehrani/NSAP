"""Standalone MIM patch injector for the EB-Navigation benchmark runner.

Pastes a trained NSAP patch onto the target's GT bounding box in a frame, using the
SAME placement math the patch was trained/evaluated with (functions.attack.patch +
functions.attack.anchor). Kept dependency-light so it can be imported from a runner
that also drives the ESCA benchmark env.
"""
import os
import torch

NSAP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if NSAP not in sys.path:
    sys.path.insert(0, NSAP)

from functions.attack.patch import inject_patch                      # noqa: E402


def load_patch(path):
    """Return (w, anchor, coverage_ratio) from a saved patch checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cov = getattr(ck.get("config"), "coverage_ratio", 0.3)
    return ck["w"], ck["anchor"], cov


def paste_on_bbox(frame_rgb, bbox_xyxy, patch):
    """frame_rgb: HxWx3 uint8 (RGB). bbox_xyxy: (x1,y1,x2,y2) in pixels.
    patch: (w, anchor, coverage_ratio). Returns (patched_frame_uint8, injected_bool).
    Thin xyxy->dict adapter over functions.attack.patch.inject_patch (the shared math)."""
    if patch is None or bbox_xyxy is None:
        return frame_rgb, False
    x1, y1, x2, y2 = [int(v) for v in list(bbox_xyxy)[:4]]
    if x2 <= x1 or y2 <= y1:
        return frame_rgb, False
    w, anchor, cov = patch
    return inject_patch(frame_rgb, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, w, anchor, cov)
