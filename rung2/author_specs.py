"""Write the symbolic specification for every staged episode by hand, from the task.

Why. Pinning the MLLM's first draw (esca.py, NSAP_PIN_VOCAB) made the specification stable
but froze whatever that draw happened to be: 3 of 10 attribute episodes were frozen to a
spec that does not describe the task (`sofa`/`blue`, `sofa`/`brown`, and one with no
attribute at all), and all 9 relation episodes carried `counter top` as an *attribute*,
which the gate averages into every box's score (esca.py:699). Every episode of a head runs
the same instruction, so the correct spec is known and is written here instead.

Design, forced by the gate at top_k=1:
  - target_names holds the ONE object the task is about. A second name competes for the
    single graph slot, and relations are drawn only from the selected object
    (esca.py:741), so `counter top` in target_names would make (pot, on, counter top)
    impossible whenever the counter wins the slot.
  - gd_query_objects follows upstream's own formula for the spec (esca.py:833-838):
    target_names, plus for a relation its endpoints and the surface list.
  - query_objects is the task's objects plus upstream's common_objects (esca.py:206),
    de-duplicated, with no empty string.

  python rung2/author_specs.py output/vocab_cache_authored
"""
import json, os, sys

COMMON = ["wall", "door", "cabinet", "stove", "table", "shadow", "window", "mirror", "other"]
SURFACES = ["counter top", "stove burner", "cabinet", "shelf", "sink", "table", "dining table",
            "side table", "coffee table", "desk", "sofa", "bed", "floor"]

def dedupe(xs):
    return list(dict.fromkeys(x for x in xs if x.strip()))

SPECS = {
    # "Only if the television is turned on, walk right up to it ... else walk up to the sofa"
    "attr": dict(query_objects=dedupe(["television", "sofa"] + COMMON),
                 gd_query_objects=["television"],
                 target_names=["television"],
                 target_attributes=["displaying"],
                 relations=[]),
    # "find the pot that is on the counter top"
    "rel": dict(query_objects=dedupe(["pot", "counter top"] + COMMON),
                gd_query_objects=dedupe(["pot", "counter top"] + SURFACES),
                target_names=["pot"],
                target_attributes=[],
                relations=["on"]),
}

EPISODES = (["attr_FloorPlan%d" % n for n in range(201, 211)]
            + ["rel_FloorPlan%d" % n for n in (2, 3, 5, 6, 8, 9, 10, 12, 13)])

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "output/vocab_cache_authored"
    os.makedirs(out, exist_ok=True)
    for ep in EPISODES:
        json.dump(SPECS[ep.split("_")[0]], open(os.path.join(out, ep + ".json"), "w"), indent=1)
    print(f"wrote {len(EPISODES)} specs to {out}")

if __name__ == "__main__":
    main()
