"""Does pose-keyed frame caching remove the perception non-determinism?

Same measurement as before: reset the scene twice, stand in the same place, run the
detector and SGClip with a FIXED query, and see whether the scene graph matches. No MLLM
is involved, so this isolates exactly what the cache is meant to fix and costs nothing.

  python rung2/probes/frame_cache_check.py --mode off
  python rung2/probes/frame_cache_check.py --mode on --cache /tmp/fc
"""
import argparse, os, sys
sys.path.insert(0, "rung2"); os.environ.setdefault("NSAP_LASER_ATTR_REL", "1")
import harness  # noqa: F401
import numpy as np
from ai2thor.controller import Controller
from nav_harness import make_nav_planner

Q = ["pot", "counter top", "stove burner", "sink", "cabinet"]
VOCAB = ["pot", "counter top", "counter", "stove", "cabinet", "table", "sink",
         "shadow", "wall", "window", "mirror", "door", "glass surface", "other"]


class Shim:
    """Minimal stand-in for StagedNavEnv so frame_cache can read pose and layout."""
    def __init__(self, c, scene): self.env = c; self.episode_data = {"scene": scene, "ep_id": "probe"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["on", "off"], default="off")
    ap.add_argument("--cache", default="/tmp/nsap_frame_cache")
    ap.add_argument("--pause-physics", action="store_true",
                    help="freeze physics after reset; AI2-THOR keeps simulating otherwise "
                         "and objects drift centimetres between resets")
    ap.add_argument("--scenes", default="FloorPlan1,FloorPlan2,FloorPlan3,FloorPlan4,FloorPlan5,FloorPlan6,FloorPlan7,FloorPlan8")
    a = ap.parse_args()
    if a.mode == "on": os.environ["NSAP_FRAME_CACHE"] = a.cache
    else: os.environ.pop("NSAP_FRAME_CACHE", None)
    import frame_cache; frame_cache._DIR = None          # re-read the env var

    P = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o")
    c = Controller(scene="FloorPlan1", width=600, height=600)

    def observe(scene):
        c.reset(scene=scene)
        if a.pause_physics: c.step(action="PausePhysicsAutoSim")
        for _ in range(5): c.step(action="Pass")
        ev = c.step(action="Pass")
        return frame_cache.clean_frame(Shim(c, scene), {"head_rgb": ev.frame.copy()})

    def graph(img):
        labels, boxes = P.predict_gd(img, Q, box_threshold=0.1, text_threshold=0.1)
        names = []
        if len(boxes):
            P.predict_laser(image=img, boxes=boxes, target_names=["pot"], query_names=VOCAB,
                            query_attributes=[], query_relation_kws=["on"])
            pn = getattr(P, "_nsap_predicted_name", {}) or {}
            names = [(v[0][1] if v else None) for k, v in sorted(pn.items(), key=lambda t: int(t[0]))]
        return len(labels), names

    same_px = same_box = same_lab = tot = 0
    for s in a.scenes.split(","):
        A = observe(s); B = observe(s)
        px = int((np.abs(A.astype(int) - B.astype(int)).sum(2) > 0).sum())
        na, ma = graph(A); nb, mb = graph(B)
        tot += 1; same_px += (px == 0); same_box += (na == nb); same_lab += (na == nb and ma == mb)
        print(f"RESULT {s}: pixels differing={px:6d} | boxes {na} vs {nb} | labels identical={na==nb and ma==mb}")
    print(f"RESULT SUMMARY mode={a.mode} pause_physics={a.pause_physics}  scenes={tot}  identical frames={same_px}/{tot}  "
          f"identical box count={same_box}/{tot}  identical labels={same_lab}/{tot}")
    frame_cache.report()
    c.stop()


if __name__ == "__main__":
    main()
