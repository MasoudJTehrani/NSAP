"""
Rung2Env — a thin closed-loop AI2-THOR-5.0 environment for the rung-2
"man-in-the-middle" NAVIGATION test. Replaces EB-ALFRED's THOR-2.1.0 env (which
won't run on this OS) with a minimal EB-Navigation skill layer on modern THOR, plus
the patch-injection seam (the current-frame target mask) and a GT-distance metric.

Navigation-only: the agent walks/turns/looks toward a target using the 8 discrete
EB-Navigation skills, and success = GT distance(agent, target) <= threshold. The old
manipulation path (ALFRED skills, object isolation, metadata success checks, and the
teleport-to-face perception seam) was removed once the MIM became navigation-only.
"""
import os, sys
import numpy as np

NSAP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (NSAP, f"{NSAP}/SceneGraphBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from sgbench.framing import face_pose            # noqa: E402


class Rung2Env:
    def __init__(self, scene, resolution=600):
        from ai2thor.controller import Controller
        from ai2thor.platform import CloudRendering
        self.scene = scene
        self.c = Controller(platform=CloudRendering, scene=scene,
                            width=resolution, height=resolution,
                            renderInstanceSegmentation=True, gridSize=0.25)
        self.reach = self.c.step("GetReachablePositions").metadata.get("actionReturn", []) or []
        self.last_feedback = ""

    # ── scene access ─────────────────────────────────────────────────────────
    @property
    def event(self):
        return self.c.last_event

    def objects(self):
        return self.c.last_event.metadata["objects"]

    def obj_by_id(self, oid):
        return next((o for o in self.objects() if o["objectId"] == oid), None)

    def _agent_pos(self):
        return self.c.last_event.metadata["agent"]["position"]

    def _nearest(self, otype):
        """Nearest object of the given THOR type to the agent (target selection)."""
        ap = self._agent_pos()
        cands = [o for o in self.objects() if o["objectType"] == otype]
        if not cands:
            return None
        return min(cands, key=lambda o: (o["position"]["x"]-ap["x"])**2 + (o["position"]["z"]-ap["z"])**2)

    # ── navigation ───────────────────────────────────────────────────────────
    def navigate_to(self, obj):
        """Teleport to a standing pose that faces `obj` (reuses SceneGraphBench framing).
        Used only by run_init for pre-episode scene setup, never by the nav loop."""
        pose = face_pose(obj, self.reach)
        if pose is None:
            self.last_feedback = f"cannot reach {obj['objectType']}"
            return False
        pos, yaw, hor = pose
        ev = self.c.step(action="TeleportFull", position=pos, rotation=dict(x=0, y=yaw, z=0),
                         horizon=hor, standing=True)
        return ev.metadata["lastActionSuccess"]

    # ── init (set any scene state a task needs before the episode) ────────────
    def run_init(self, init_list):
        for it in init_list or []:
            o = self._nearest(it["object_type"])
            if o:
                self.navigate_to(o)
                self.c.step(action=it["action"], objectId=o["objectId"], forceAction=True)

    # ── EB-Navigation: real step-by-step movement + GT-distance metric ────────
    # Replicates EmbodiedBench's EB-Navigation (envs/eb_navigation/EBNavEnv.py):
    # 8 discrete move/rotate/look skills, success = GT distance(agent,target) <= 1 m.
    NAV_SKILLS = [
        "Move forward by 0.25", "Move backward by 0.25",
        "Move rightward by 0.25", "Move leftward by 0.25",
        "Rotate to the right by 90 degrees.", "Rotate to the left by 90 degrees.",
        "Tilt the camera upward by 30 degrees.", "Tilt the camera downward by 30 degrees.",
    ]

    def nav_move(self, action):
        """Execute one EB-Navigation movement skill (string from NAV_SKILLS or its index).
        Mirrors EBNavEnv.discrete_action_mapper. Returns (success, feedback)."""
        a = action.strip().lower() if isinstance(action, str) else \
            (self.NAV_SKILLS[action].lower() if isinstance(action, int)
             and 0 <= action < len(self.NAV_SKILLS) else "")
        step = self.c.step
        if "forward" in a:                         ev = step(action="MoveAhead", moveMagnitude=0.25)
        elif "backward" in a:                      ev = step(action="MoveBack",  moveMagnitude=0.25)
        elif "rightward" in a:                     ev = step(action="MoveRight", moveMagnitude=0.25)
        elif "leftward" in a:                      ev = step(action="MoveLeft",  moveMagnitude=0.25)
        elif "rotate" in a and "right" in a:       ev = step(action="RotateRight", degrees=90)
        elif "rotate" in a and "left" in a:        ev = step(action="RotateLeft",  degrees=90)
        elif ("up" in a) or ("upward" in a):       ev = step(action="LookUp",   degrees=30)
        elif ("down" in a) or ("downward" in a):   ev = step(action="LookDown", degrees=30)
        else:
            return False, f"unknown nav action: {action}"
        ok = ev.metadata["lastActionSuccess"]
        return ok, ("moved" if ok else (ev.metadata.get("errorMessage", "") or "blocked")[:80])

    def nav_distance(self, target_id):
        """GT Euclidean distance (x,z) from the agent to the target object (EB-Navigation)."""
        o = self.obj_by_id(target_id)
        if not o:
            return float("inf")
        ap, tp = self._agent_pos(), o["position"]
        return ((ap["x"] - tp["x"]) ** 2 + (ap["z"] - tp["z"]) ** 2) ** 0.5

    def nav_start(self, target_id, seed=0, success_dist=1.5, near_margin=0.75, far_margin=2.0,
                  face_jitter=35.0):
        """Face the agent toward the target (yaw +/- face_jitter) at a distance STRICTLY
        beyond the success radius -- in [success_dist+near_margin, success_dist+far_margin] --
        so it always has to actually move (it can NEVER spawn already inside the success
        radius and count as trivially done). The fallbacks also stay beyond the radius; only
        if the target is completely boxed in do we take the farthest reachable poses.
        Returns the start distance."""
        import random, math
        o = self.obj_by_id(target_id)
        if not o or not self.reach:
            return None
        tp = o["position"]
        def d(p):
            return ((p["x"] - tp["x"]) ** 2 + (p["z"] - tp["z"]) ** 2) ** 0.5
        lo, hi = success_dist + near_margin, success_dist + far_margin
        band = ([p for p in self.reach if lo <= d(p) <= hi]
                or [p for p in self.reach if d(p) >= lo]              # beyond radius, any distance
                or sorted(self.reach, key=d, reverse=True)[:5])       # target boxed in -> farthest
        rng = random.Random(seed)
        pos = rng.choice(band)
        # THOR yaw: 0 deg faces +z, 90 deg faces +x -> atan2(dx, dz) points at the target.
        dx, dz = tp["x"] - pos["x"], tp["z"] - pos["z"]
        yaw = (math.degrees(math.atan2(dx, dz)) + rng.uniform(-face_jitter, face_jitter)) % 360
        self.c.step(action="TeleportFull", position=pos,
                    rotation=dict(x=0, y=yaw, z=0), horizon=0, standing=True)
        return self.nav_distance(target_id)

    def teleport_pose(self, pose, target_id):
        """Teleport to an EXACT pose (EB-Navigation start: position + rotation + horizon),
        e.g. long_horizon starts facing AWAY from the target. Returns the start distance."""
        p = pose["position"]
        self.c.step(action="TeleportFull",
                    position=dict(x=p["x"], y=p["y"], z=p["z"]),
                    rotation=dict(x=0, y=pose.get("rotation", 0), z=0),
                    horizon=pose.get("horizon", 0), standing=True)
        return self.nav_distance(target_id)

    def render_frame(self):
        return np.array(self.c.last_event.frame)

    def stop(self):
        try:
            self.c.stop()
        except Exception:
            pass
