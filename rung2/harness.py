"""
Rung-2 shared setup + patch-injection utilities.

Importing this module (a) puts ESCA + LASER on sys.path, sets the OPENAI / GD / VINE
env vars, and chdirs to the repo root -- the reason nav_harness / eb_bench_runner /
staged_bench all `import harness` -- and (b) exposes the man-in-the-middle injection
helpers (render_inject, load_patch, THOR2VINE, ...) that paste the frozen patch onto the
target object in the current THOR frame.

The old behavioral (ALFRED manipulation) harness -- the VLMPlanner, the ACTION space,
run_episode, and its perception gate -- was removed once the MIM became navigation-only;
see nav_harness.py for the live nav loop. Run in the `embench` env.
"""
import os, sys
NSAP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ESCA (our modified fork) ships as a submodule in the replication package, so default to
# it. Override with $ESCA_ROOT to point at a separate working copy (e.g. on a dev machine).
ESCA = os.environ.get("ESCA_ROOT", f"{NSAP}/external/ESCA")
for p in (ESCA, NSAP, f"{NSAP}/src/LASER", f"{NSAP}/src/LASER/laser/models",
          f"{ESCA}/embodiedbench/evaluator", f"{NSAP}/rung2", f"{NSAP}/SceneGraphBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# OpenAI key: the planner needs it. Prefer $OPENAI_API_KEY; optionally fall back to an
# {ESCA}/.openai_key file (a dev convenience, absent in the package -> silently skipped).
if not os.environ.get("OPENAI_API_KEY"):
    import re
    _keyfile = f"{ESCA}/.openai_key"
    if os.path.isfile(_keyfile):
        _m = re.search(r"sk-[A-Za-z0-9_-]+", open(_keyfile).read())
        if _m:
            os.environ["OPENAI_API_KEY"] = _m.group(0)
# GroundingDINO checkpoint: download_model.py fetches it here (NSAP-local, NOT ESCA's copy).
os.environ.setdefault("NSAP_GD_CKPT", f"{NSAP}/checkpoints/groundingdino_swint_ogc.pth")
# VINE weights: pin to this repo's trained VINE (esca.py's own default is machine-specific).
os.environ.setdefault("NSAP_VINE_WEIGHTS", f"{NSAP}/model/vine/model.safetensors")
os.chdir(NSAP)

import numpy as np, torch
from PIL import Image
from functions.attack.patch import inject_patch
from functions.attack.anchor import Anchor                     # noqa: F401  (unpickle load_patch's anchor)
from functions.attack.train import AttackConfig                # noqa: F401  (unpickle load_patch's config)

# RELATIVE path under cwd (=NSAP): ESCA's planner splits the image path on '/' and
# os.path.join's it back to derive the bbox-viz path, which drops the leading slash on
# absolute paths. A relative path under an existing dir resolves correctly.
os.makedirs("output/rung2/obs", exist_ok=True)
TMP_IMG = "output/rung2/obs/obs.png"


def load_patch(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["w"], ck["anchor"], getattr(ck.get("config"), "coverage_ratio", 0.3)


# THOR objectType -> the SGClip category the patch attacks (classify_vocab's lowercase names).
THOR2VINE = {"Pot": "pot", "Microwave": "microwave", "Toaster": "toaster",
             "CoffeeMachine": "coffee machine", "Fridge": "fridge", "Kettle": "kettle",
             "Apple": "apple", "Bowl": "bowl"}


def _target_bbox(ev, target_id):
    """(bbox_dict, mask) of the target in the current frame, else (None, mask_or_None)."""
    mask = ev.instance_masks.get(target_id) if target_id else None
    if mask is None:
        return None, None
    ys, xs = np.where(mask)
    if not len(xs):
        return None, mask
    return {"x1": int(xs.min()), "y1": int(ys.min()),
            "x2": int(xs.max()), "y2": int(ys.max())}, mask


def _paste_on_target(frame, bbox, patch):
    """Adapter: unpack the (w, anchor, cov) patch tuple onto inject_patch. Returns (frame, flipped)."""
    if patch is None:
        return frame, False
    w, anchor, cov = patch
    return inject_patch(frame, bbox, w, anchor, cov)


def _whiten_outside_bbox(frame, bbox, alpha=1.0):
    """Blend everything OUTSIDE the target's bbox toward white (alpha=1 => hard white),
    leaving the bbox region (patch included) untouched. Removes background regions so GD +
    SGClip can't ground them AS the target -- kills the false-positive 'toaster' boxes on
    the counter/window that the patch can't reach -- while keeping the object at TRUE scale."""
    if bbox is None:
        return frame
    out = frame.astype(np.float32)
    keep = np.zeros(frame.shape[:2], dtype=bool)
    keep[bbox["y1"]:bbox["y2"] + 1, bbox["x1"]:bbox["x2"] + 1] = True
    white = np.full_like(out, 255.0)
    bg = ~keep
    out[bg] = out[bg] * (1.0 - alpha) + white[bg] * alpha
    return out.astype(np.uint8)


def render_inject(env, target_id, patch, save_copy=None, whiten=False):
    """Save the current THOR frame to TMP_IMG for the planner, with the patch injected on
    the target in full-frame coords. If whiten, the PLANNER INPUT has its background whitened
    (so VINE's SG only grounds the real target); the obs save_copy stays the full frame."""
    ev = env.event
    # patch EVERY frame the target is present, regardless of size (paste in full-frame coords).
    bbox, mask = _target_bbox(ev, target_id)
    frame, flipped_here = _paste_on_target(np.array(ev.frame), bbox, patch)
    fed = _whiten_outside_bbox(frame, bbox) if whiten else frame
    Image.fromarray(fed).save(TMP_IMG)            # planner / VINE input (bg whitened if enabled)
    if save_copy:
        Image.fromarray(frame).save(save_copy)    # obs view = full patched frame (never whitened)
    return TMP_IMG, flipped_here
