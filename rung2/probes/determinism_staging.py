"""Do our staged scenes come back the same every time?

The staged relation tasks spawn two pots with `InitialRandomSpawn` at a fixed seed and
keep the first seed that puts one pot on a counter and one on a stove
(`rung2/staged_bench.py:50`). That is deterministic only if AI2-THOR's spawn, which settles
objects with physics, is deterministic. If it is not, the accept-or-retry loop amplifies the
difference: a seed accepted in one run can be rejected in the next, and the run then keeps a
different seed and an entirely different layout.

This matters because the step logs show the detector returning a different number of boxes
on 43% of step-0 pairs in the staged runs but only 4% in the benchmark runs, with a detector
we have separately verified to be bit-identical across calls. A scene that is not reproducible
would explain that.

  python rung2/probes/determinism_staging.py --repeats 5 --scenes FloorPlan1,FloorPlan2
"""
import argparse, collections, json, os, sys

sys.path.insert(0, "rung2")
import harness  # noqa: F401
from ai2thor.controller import Controller


def recep(o):
    return " ".join(o.get("parentReceptacles") or [])


def spawn_once(c, scene, spawn_seed=3):
    """Mirror of staged_bench.stage_relation: first seed that gives counter + stove."""
    for seed in range(spawn_seed, spawn_seed + 8):
        c.reset(scene=scene)
        ev = c.step(action="InitialRandomSpawn", randomSeed=seed, forceVisible=True,
                    numPlacementAttempts=12,
                    numDuplicatesOfType=[{"objectType": "Pot", "count": 2}])
        if not ev.metadata.get("lastActionSuccess"):
            continue
        pots = [o for o in c.last_event.metadata["objects"] if o["objectType"] == "Pot"]
        onc = [o for o in pots if "CounterTop" in recep(o)]
        ons = [o for o in pots if "StoveBurner" in recep(o)]
        if onc and ons:
            return seed, onc[0], ons[0], pots
    return None, None, None, []


def pos(o, nd=3):
    p = o["position"]
    return (round(p["x"], nd), round(p["y"], nd), round(p["z"], nd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--scenes", default="FloorPlan1,FloorPlan2,FloorPlan3")
    ap.add_argument("--spawn-seed", type=int, default=3)
    ap.add_argument("--out", default="output/probes/determinism_staging.json")
    a = ap.parse_args()

    c = Controller(scene="FloorPlan1", width=600, height=600)
    rows = []
    for scene in a.scenes.split(","):
        scene = scene.strip()
        seeds, layouts, counts = [], [], []
        for r in range(a.repeats):
            seed, onc, ons, pots = spawn_once(c, scene, a.spawn_seed)
            if seed is None:
                print(f"  {scene} repeat {r}: staging FAILED", flush=True)
                seeds.append(None); layouts.append(None); counts.append(0)
                continue
            seeds.append(seed)
            layouts.append((pos(onc), pos(ons)))
            counts.append(len(pots))
            print(f"  {scene} repeat {r}: seed={seed}  counter_pot={pos(onc)}  "
                  f"stove_pot={pos(ons)}  pots={len(pots)}", flush=True)
        same_seed = len({s for s in seeds if s is not None}) <= 1
        same_layout = len({l for l in layouts if l is not None}) <= 1
        rows.append(dict(scene=scene, seeds=seeds, same_seed=same_seed,
                         same_layout=same_layout,
                         distinct_layouts=len({l for l in layouts if l is not None})))
        print(f"  -> {scene}: same seed chosen={same_seed}  identical layout={same_layout}  "
              f"distinct layouts={rows[-1]['distinct_layouts']}\n", flush=True)
    c.stop()

    n = len(rows)
    summ = dict(repeats=a.repeats, scenes=n,
                scenes_same_seed=sum(r["same_seed"] for r in rows),
                scenes_same_layout=sum(r["same_layout"] for r in rows))
    print("================ STAGING ================")
    print(f"  scenes that chose the same spawn seed every time : {summ['scenes_same_seed']}/{n}")
    print(f"  scenes whose layout was identical every time     : {summ['scenes_same_layout']}/{n}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": summ, "per_scene": rows}, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
