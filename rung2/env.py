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

NSAP = "/home/usi/NSAP"
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
    def view_target(self, target_id, min_area=600, tries=8, seed=0):
        """Teleport to face the target object and return (frame_uint8, mask_bool, bbox)
        from a viewpoint where it's adequately visible. Returns (None,None,None) if not."""
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
            yaw, hor = (base_yaw, base_hor) if v == 0 else jitter(base_yaw, base_hor, rng)
            ev = self.c.step(action="TeleportFull", position=base_pos,
                             rotation=dict(x=0, y=yaw, z=0), horizon=hor, standing=True)
            m = ev.instance_masks.get(target_id)
            if m is not None and int(m.sum()) >= min_area:
                ys, xs = np.where(m)
                bbox = {"x1": int(xs.min()), "y1": int(ys.min()),
                        "x2": int(xs.max()), "y2": int(ys.max())}
                return np.array(ev.frame), m, bbox
        return None, None, None

    def render_frame(self):
        return np.array(self.c.last_event.frame)

    def stop(self):
        try:
            self.c.stop()
        except Exception:
            pass
