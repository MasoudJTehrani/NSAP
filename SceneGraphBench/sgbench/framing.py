"""
Object-centric camera framing.

A robot perceives an object by navigating to it and looking at it — so the
benchmark frames each target object close and centered (with small jitter for
viewpoint variation), not from random room poses. This is what makes the rendered
views match deployment.
"""

import math
import random


def face_pose(obj, reachable, eye_offset=0.9, near=0.5, far=1.8):
    """A reachable agent position near `obj` plus the yaw/horizon to face it.

    Returns (position_dict, yaw_deg, horizon_deg) or None if nowhere reachable.
    yaw: 0 faces +z, clockwise. horizon: positive looks down.
    """
    op = obj['position']
    pool = [p for p in reachable if near <= math.hypot(op['x'] - p['x'], op['z'] - p['z']) <= far]
    pool = pool or reachable
    if not pool:
        return None
    best = min(pool, key=lambda p: math.hypot(op['x'] - p['x'], op['z'] - p['z']))
    dx, dz = op['x'] - best['x'], op['z'] - best['z']
    dist = math.hypot(dx, dz)
    yaw = math.degrees(math.atan2(dx, dz)) % 360
    horizon = math.degrees(math.atan2((best['y'] + eye_offset) - op['y'], max(dist, 1e-3)))
    return best, yaw, max(-30.0, min(60.0, horizon))


def jitter(yaw, horizon, rng, yaw_deg=12.0, horizon_deg=8.0):
    """Small random viewpoint perturbation around a facing pose (real-render EoT)."""
    return (yaw + rng.uniform(-yaw_deg, yaw_deg),
            max(-30.0, min(60.0, horizon + rng.uniform(-horizon_deg, horizon_deg))))
