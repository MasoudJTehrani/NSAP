"""Make the simulator's pixels reproducible by memoising frames on the agent's pose.

Why this exists. AI2-THOR does not render the same scene the same way twice. Two resets
of one scene, five settle steps apart, differ on a median of ~108k of 360k pixels, and
re-rendering an unchanged state changes pixels too. Every model downstream is
deterministic, but GroundingDINO thresholds its scores
(`planner/navigation/esca.py:517`), so a box near the threshold flips in or out, every
later box index shifts, SGClip receives different crops, and the scene graph changes. We
measured 2 px of box movement turning into 146 label changes over 129 boxes.

The fix is to stop re-rendering. Navigation is discrete (`MoveAhead` moves 0.25 m on a 0.1
grid with fixed rotations), so the agent keeps returning to the same poses. The first time
a pose is seen we store its frame; every later visit, in any condition and any repetition,
replays those exact pixels. Two runs that take the same actions then see identical input,
and a patched run differs from a benign one only by the composited patch.

What is cached is the CLEAN frame. The patch is pasted on top afterwards, so the cache is
shared between conditions and never contains an attack.

Validity. A cached frame is only correct if the frame is a function of (object layout,
pose). The layout is therefore part of the cache path, not an assumption: if a run stages
the scene differently it gets a different cache directory rather than silently reusing
frames from another layout. Counting those directories also measures how reproducible the
staging is.

Enable with NSAP_FRAME_CACHE=/path/to/dir. Unset, nothing changes.
"""
import hashlib
import json
import os

import numpy as np
from PIL import Image

_DIR = None
_STATS = {"hit": 0, "miss": 0, "layouts": set()}


def enabled():
    global _DIR
    if _DIR is None:
        _DIR = os.environ.get("NSAP_FRAME_CACHE", "")
    return bool(_DIR)


def _agent(env):
    return env.env.last_event.metadata["agent"]


def pose_key(env):
    a = _agent(env)
    p, r = a["position"], a["rotation"]
    return (f"x{p['x']:.3f}_y{p['y']:.3f}_z{p['z']:.3f}"
            f"_ry{r['y']:.1f}_h{a.get('cameraHorizon', 0.0):.1f}"
            f"_s{int(bool(a.get('isStanding', True)))}")


def layout_sig(env):
    """Positions of every pickupable object. Two runs that staged the scene differently
    get different signatures, so their frames never mix."""
    objs = sorted(
        (o["objectId"], round(o["position"]["x"], 3), round(o["position"]["y"], 3),
         round(o["position"]["z"], 3))
        for o in env.env.last_event.metadata["objects"] if o.get("pickupable"))
    return hashlib.sha1(json.dumps(objs).encode()).hexdigest()[:12]


def _path(env):
    ep = env.episode_data or {}
    scene = str(ep.get("scene", "scene"))
    ep_id = str(ep.get("ep_id", "episode"))
    sig = layout_sig(env)
    _STATS["layouts"].add((scene, ep_id, sig))
    d = os.path.join(_DIR, scene, ep_id, sig)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, pose_key(env) + ".png")


def clean_frame(env, obs):
    """The clean frame for the agent's current pose: replayed if seen before, else
    rendered now and stored. Falls back to the live frame on any cache problem."""
    live = np.asarray(obs["head_rgb"]).astype(np.uint8)
    if not enabled():
        return live
    try:
        p = _path(env)
        if os.path.exists(p):
            cached = np.asarray(Image.open(p).convert("RGB")).astype(np.uint8)
            if cached.shape == live.shape:
                _STATS["hit"] += 1
                return cached
        Image.fromarray(live).save(p)
        _STATS["miss"] += 1
        return live
    except Exception as e:                      # never let caching break a run
        print(f"      [frame_cache] disabled for this frame: {repr(e)[:110]}", flush=True)
        return live


def stats():
    n = _STATS["hit"] + _STATS["miss"]
    return {"hits": _STATS["hit"], "misses": _STATS["miss"],
            "hit_rate": (_STATS["hit"] / n) if n else None,
            "distinct_layouts": len(_STATS["layouts"])}


def report():
    if not enabled():
        return
    s = stats()
    print(f"      [frame_cache] {s['hits']} replayed, {s['misses']} rendered"
          + (f", hit rate {100 * s['hit_rate']:.0f}%" if s["hit_rate"] is not None else "")
          + f", {s['distinct_layouts']} distinct staged layouts", flush=True)


def freeze_physics(env):
    """Stop AI2-THOR simulating physics for the rest of the episode.

    The simulator never stops stepping physics, so objects keep drifting: we measured up
    to 1.8 cm of movement between two resets after 5 settle steps, and 8.6 cm after 30, so
    waiting longer makes it worse. A drifting world means a different frame every run, and
    it also defeats the frame cache, whose key includes the object layout.

    Tied to the cache on purpose: a cached frame is only valid if the world it was rendered
    from cannot move. Set NSAP_PAUSE_PHYSICS=0 to override.
    """
    if not enabled() or os.environ.get("NSAP_PAUSE_PHYSICS", "1") == "0":
        return False
    try:
        env.env.step(action="PausePhysicsAutoSim")
        return True
    except Exception as e:
        print(f"      [frame_cache] could not pause physics: {repr(e)[:110]}", flush=True)
        return False
