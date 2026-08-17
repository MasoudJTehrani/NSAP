"""
Rung2Env — a thin closed-loop AI2-THOR-5.0 task environment for the rung-2
"man-in-the-middle" test. Replaces EB-ALFRED's THOR-2.1.0 env (which won't run on
this OS) with a minimal skill layer on modern THOR, plus the patch-injection seam
and metadata-based success checks.

Skills (ALFRED-style language actions): "find a X", "pick up X", "open X",
"close X", "turn on X", "turn off X", "put down ... in the X", "slice X". Each
interaction auto-navigates to the nearest instance of the named type first, so the
behavioral divergence we measure comes from the planner's *choice* of actions
(which the flipped scene graph corrupts), not from THOR interactability mechanics.
"""
import os, sys
import numpy as np

NSAP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (NSAP, f"{NSAP}/SceneGraphBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
from sgbench.framing import face_pose, jitter            # noqa: E402

THOR_TYPES = ["Fridge", "Microwave", "Toaster", "CoffeeMachine", "Kettle", "Pot",
              "Apple", "Bowl", "StoveBurner", "CounterTop", "Cabinet", "Sink",
              "Bread", "Egg", "Potato", "Lettuce", "Tomato", "Plate", "Mug", "Pan"]


def _norm_type(word):
    """Map a free-text object word from a language action to a THOR objectType."""
    w = word.strip().lower().replace(" ", "")
    table = {t.lower(): t for t in THOR_TYPES}
    table.update({"coffeemachine": "CoffeeMachine", "coffee": "CoffeeMachine",
                  "stove": "StoveBurner", "counter": "CounterTop", "countertop": "CounterTop"})
    return table.get(w)


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

    def _nearest(self, otype, **flags):
        ap = self._agent_pos()
        cands = [o for o in self.objects() if o["objectType"] == otype]
        for k, v in flags.items():
            cands = [o for o in cands if o.get(k) == v]
        if not cands:
            return None
        return min(cands, key=lambda o: (o["position"]["x"]-ap["x"])**2 + (o["position"]["z"]-ap["z"])**2)

    # ── navigation ───────────────────────────────────────────────────────────
    def navigate_to(self, obj):
        """Teleport to a standing pose that faces `obj` (reuses SceneGraphBench framing)."""
        pose = face_pose(obj, self.reach)
        if pose is None:
            self.last_feedback = f"cannot reach {obj['objectType']}"
            return False
        pos, yaw, hor = pose
        ev = self.c.step(action="TeleportFull", position=pos, rotation=dict(x=0, y=yaw, z=0),
                         horizon=hor, standing=True)
        if ev.metadata["lastActionSuccess"]:
            self.last_nav = obj["objectId"]   # ALFRED 'find X then put' -> remember receptacle
        return ev.metadata["lastActionSuccess"]

    # ── one interaction step ─────────────────────────────────────────────────
    def execute(self, lang_action):
        """Run one ALFRED-style language action; auto-navigates to the target type.
        Returns (success: bool, feedback: str)."""
        if not lang_action or not isinstance(lang_action, str):
            return False, "empty action"
        a = lang_action.strip().lower()

        def _target_type(prefixes):
            for pre in prefixes:
                if a.startswith(pre):
                    return _norm_type(a[len(pre):])
            return None

        # held object (for put)
        held = [o for o in self.objects() if o.get("isPickedUp")]
        held = held[0] if held else None

        try:
            if a.startswith("find a") or a.startswith("find the") or a.startswith("go to"):
                t = _norm_type(a.split(maxsplit=2)[-1])
                o = self._nearest(t) if t else None
                if not o:
                    return False, f"no {t} in scene"
                ok = self.navigate_to(o)
                return ok, ("navigated to " + t) if ok else f"could not reach {t}"

            if a.startswith("pick up") or a.startswith("pickup"):
                t = _norm_type(a.replace("pick up", "").replace("pickup", "").replace("the", ""))
                o = self._nearest(t, pickupable=True) if t else None
                if not o:
                    return False, f"no pickupable {t}"
                self.navigate_to(o)
                ev = self.c.step(action="PickupObject", objectId=o["objectId"], forceAction=True)
                return ev.metadata["lastActionSuccess"], f"picked up {t}" if ev.metadata["lastActionSuccess"] else ev.metadata["errorMessage"][:80]

            if a.startswith("open") or a.startswith("close"):
                act = "OpenObject" if a.startswith("open") else "CloseObject"
                t = _norm_type(a.split(maxsplit=1)[-1].replace("the", ""))
                o = self._nearest(t, openable=True) if t else None
                if not o:
                    return False, f"no openable {t}"
                self.navigate_to(o)
                ev = self.c.step(action=act, objectId=o["objectId"], forceAction=True)
                return ev.metadata["lastActionSuccess"], f"{act} {t}"

            if a.startswith("turn on") or a.startswith("turn off"):
                act = "ToggleObjectOn" if a.startswith("turn on") else "ToggleObjectOff"
                t = _norm_type(a.replace("turn on", "").replace("turn off", "").replace("the", ""))
                o = self._nearest(t, toggleable=True) if t else None
                if not o:
                    return False, f"no toggleable {t}"
                self.navigate_to(o)
                ev = self.c.step(action=act, objectId=o["objectId"], forceAction=True)
                return ev.metadata["lastActionSuccess"], f"{act} {t}"

            if a.startswith("put") or a.startswith("drop"):
                if held is None:
                    return False, "nothing in hand"
                # explicit receptacle type? (ignore the phrase "in hand")
                t = None
                for sep in (" into ", " in the ", " on the ", " in ", " on "):
                    if sep in a:
                        tail = a.split(sep)[-1]
                        if "hand" in tail:
                            continue
                        cand = _norm_type(tail)
                        if cand:
                            t = cand; break
                recep = self._nearest(t, receptacle=True) if t else None
                # default: the receptacle we last navigated to (ALFRED 'find X then put down')
                if recep is None:
                    o = self.obj_by_id(getattr(self, "last_nav", None))
                    if o and o.get("receptacle"):
                        recep = o
                if recep is None:   # else nearest receptacle of any kind
                    ap = self._agent_pos()
                    recs = [o for o in self.objects() if o.get("receptacle")]
                    recep = min(recs, key=lambda o: (o["position"]["x"]-ap["x"])**2 + (o["position"]["z"]-ap["z"])**2) if recs else None
                if not recep:
                    return False, "no receptacle nearby"
                self.navigate_to(recep)
                ev = self.c.step(action="PutObject", objectId=recep["objectId"], forceAction=True)
                ok = ev.metadata["lastActionSuccess"]
                return ok, (f"put {held['objectType']} in {recep['objectType']}" if ok else ev.metadata["errorMessage"][:80])

            if a.startswith("slice"):
                t = _norm_type(a.split(maxsplit=1)[-1].replace("the", ""))
                o = self._nearest(t, sliceable=True) if t else None
                if not o:
                    return False, f"no sliceable {t}"
                self.navigate_to(o)
                ev = self.c.step(action="SliceObject", objectId=o["objectId"], forceAction=True)
                return ev.metadata["lastActionSuccess"], f"sliced {t}"

        except Exception as e:
            return False, f"exec error: {repr(e)[:80]}"
        return False, f"unparsed action: {lang_action}"

    # ── init (set scene state the attack needs) ──────────────────────────────
    def run_init(self, init_list):
        for it in init_list or []:
            o = self._nearest(it["object_type"])
            if o:
                self.navigate_to(o)
                self.c.step(action=it["action"], objectId=o["objectId"], forceAction=True)

    # ── MIM isolation test: strip confusable objects from the scene ───────────
    def isolate_target(self, target_id, keep_support=True):
        """Disable non-target PICKUPABLE/MOVEABLE objects — the confusable items and
        small appliances (potato/bowl/bread/coffee-machine...) that VINE misreads as
        the target category, producing spurious detections that survive the patch's
        flip. Structural objects (floor/walls/counters/cabinets) are LEFT in place:
        they ARE in objects(), and disabling them breaks THOR's floor/navmesh so the
        next TeleportFull fails ('too large a change') and the target drops off its
        support. Only the live MIM scene is affected; the dataset is untouched.
        Returns the number of objects disabled."""
        tgt  = self.obj_by_id(target_id)
        keep = {target_id}
        if keep_support and tgt:
            keep.update(tgt.get("parentReceptacles") or [])
        disabled = 0
        for o in list(self.objects()):
            if o["objectId"] in keep:
                continue
            if not (o.get("pickupable") or o.get("moveable")):
                continue   # structural — removing it breaks navigation; keep it
            ev = self.c.step(action="DisableObject", objectId=o["objectId"])
            if ev.metadata.get("lastActionSuccess"):
                disabled += 1
        return disabled

    # ── success checks ───────────────────────────────────────────────────────
    def check_success(self, spec):
        t = spec["type"]
        if t == "is_toggled":
            o = self._nearest(spec["object"])
            return bool(o and o.get("isToggled") == spec.get("value", True))
        if t == "is_open":
            o = self._nearest(spec["object"])
            return bool(o and o.get("isOpen") == spec.get("value", True))
        if t == "is_picked_up":
            return any(o["objectType"] == spec["object"] and o.get("isPickedUp") for o in self.objects())
        if t == "in_receptacle":
            recep = self._nearest(spec["receptacle"])
            if not recep:
                return False
            contained = recep.get("receptacleObjectIds") or []
            return any(spec["object"] in cid for cid in contained)
        return False

    # ── perception target: face object + return frame, mask, bbox ────────────
    def view_target(self, target_id, min_area=600, tries=8, seed=0, force_jitter=False):
        """Teleport to face the target object and return (frame_uint8, mask_bool, bbox)
        from a viewpoint where it's adequately visible. Returns (None,None,None) if not.
        force_jitter=True jitters even the first pose (for distinct viewpoints per seed)."""
        import random
        obj = self.obj_by_id(target_id)
        if obj is None:
            return None, None, None
        pose = face_pose(obj, self.reach)
        if pose is None:
            return None, None, None
        base_pos, base_yaw, base_hor = pose
        rng = random.Random(seed)
        for v in range(tries):
            yaw, hor = (base_yaw, base_hor) if (v == 0 and not force_jitter) else jitter(base_yaw, base_hor, rng)
            ev = self.c.step(action="TeleportFull", position=base_pos,
                             rotation=dict(x=0, y=yaw, z=0), horizon=hor, standing=True)
            m = ev.instance_masks.get(target_id)
            if m is not None and int(m.sum()) >= min_area:
                ys, xs = np.where(m)
                bbox = {"x1": int(xs.min()), "y1": int(ys.min()),
                        "x2": int(xs.max()), "y2": int(ys.max())}
                return np.array(ev.frame), m, bbox
        return None, None, None

    # ── EB-Navigation: real step-by-step movement + GT-distance metric ────────
    # Replicates EmbodiedBench's EB-Navigation (envs/eb_navigation/EBNavEnv.py):
    # 8 discrete move/rotate/look skills, success = GT distance(agent,target) <= 1 m.
    # We keep this alongside the ALFRED skills so the SAME env can do the nav MIM test.
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
