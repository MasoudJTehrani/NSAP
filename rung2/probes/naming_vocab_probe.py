"""Is the pot lost because SGClip cannot name it, or because of the vocabulary it is given?

In deployment, ESCA hands SGClip the names gpt-4o proposed plus its own `common_objects`
list, which contains `shadow`, `wall`, `window`, `mirror`, `door` and `other`. Median
vocabulary size in our relation runs is 16. Over those runs SGClip names only 13% of the
detector's own `pot` boxes `pot`, and 36% of them `shadow`.

This probe keeps the frames and the detector boxes fixed and varies only the vocabulary,
so the two rates are directly comparable:

  small       the six names a task-specific vocabulary would hold
  deployed    a reconstruction of the vocabulary ESCA assembles, taken from the
              `query_names` logged in the relation runs

  python rung2/probes/naming_vocab_probe.py [--out output/probes/naming_vocab.json]
"""
import argparse, collections, json, os, sys

sys.path.insert(0, "rung2")
os.environ.setdefault("NSAP_LASER_ATTR_REL", "1")
import harness  # noqa: F401  (sets ESCA_ROOT and the import paths)
import numpy as np
from nav_harness import make_nav_planner
from functions.attack.sim_data import SimAttackDataset, explode_to_frames

# The six names a vocabulary written for this task would hold.
SMALL = ["pot", "counter top", "stove", "shadow", "wall", "table"]

# What ESCA actually assembles: the proposed names plus its common objects. Taken from the
# most frequent `query_names` entries logged across the relation runs (median size 16).
DEPLOYED = ["pot", "counter top", "counter", "stove", "cabinet", "table", "sink",
            "shadow", "wall", "window", "mirror", "door", "glass surface", "light",
            "other", "shelf"]

VOCABS = {"small": SMALL, "deployed": DEPLOYED}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/probes/naming_vocab.json")
    args = ap.parse_args()

    P = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o")
    ds = SimAttackDataset("dataset/sim", target_category="Pot", target_relation="on",
                          data_source="pot", split=None)
    items = [fr for i in range(len(ds)) for fr in explode_to_frames(ds[i])]
    print(f"frames: {len(items)}")

    counts = {k: collections.Counter() for k in VOCABS}
    confs = {k: [] for k in VOCABS}
    by_area = {k: collections.defaultdict(collections.Counter) for k in VOCABS}
    n_boxes = 0

    for it in items:
        fr = it["frames"][0]
        img = np.asarray(fr.detach().cpu() if hasattr(fr, "detach") else fr).astype(np.uint8)
        try:
            labels, boxes = P.predict_gd(img, ["pot", "counter top", "stove burner", "sink", "cabinet"],
                                         box_threshold=0.1, text_threshold=0.1)
        except Exception as e:
            print("skip gd:", type(e).__name__, str(e)[:80])
            continue
        if len(boxes) == 0:
            continue
        pot_idx = [i for i, lb in enumerate(labels) if str(lb).lower() == "pot"]
        if not pot_idx:
            continue
        n_boxes += len(pot_idx)
        for vname, vocab in VOCABS.items():
            try:
                P.predict_laser(image=img, boxes=boxes, target_names=["pot"], query_names=vocab,
                                query_attributes=[], query_relation_kws=["on"])
            except Exception as e:
                print("skip laser:", type(e).__name__, str(e)[:80])
                continue
            pn = getattr(P, "_nsap_predicted_name", {}) or {}
            for i in pot_idx:
                v = pn.get(i) or pn.get(str(i))
                if not v:
                    continue
                name = str(v[0][1]).lower()
                counts[vname][name] += 1
                if name == "pot":
                    confs[vname].append(float(v[0][0]))
                b = boxes[i].tolist() if hasattr(boxes[i], "tolist") else boxes[i]
                area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
                tag = ("tiny" if area < 2500 else "small" if area < 10000
                       else "medium" if area < 40000 else "large")
                by_area[vname][tag][name] += 1

    res = {"frames": len(items), "detector_pot_boxes": n_boxes, "vocabularies": VOCABS, "by_vocab": {}}
    for vname in VOCABS:
        c = counts[vname]
        tot = sum(c.values())
        res["by_vocab"][vname] = {
            "scored": tot,
            "named_pot": c["pot"],
            "named_pot_pct": round(100 * c["pot"] / tot, 1) if tot else None,
            "named_shadow": c["shadow"],
            "named_shadow_pct": round(100 * c["shadow"] / tot, 1) if tot else None,
            "mean_conf_when_pot": round(sum(confs[vname]) / len(confs[vname]), 3) if confs[vname] else None,
            "top_names": c.most_common(8),
            "by_area": {k: {"n": sum(v.values()), "named_pot": v["pot"]} for k, v in by_area[vname].items()},
        }
        print(f"\n[{vname}] vocabulary of {len(VOCABS[vname])} names, {tot} boxes scored")
        print(f"  named pot   : {c['pot']}/{tot} = {100 * c['pot'] / max(tot, 1):.0f}%")
        print(f"  named shadow: {c['shadow']}/{tot} = {100 * c['shadow'] / max(tot, 1):.0f}%")
        print(f"  top names   : {c.most_common(6)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
