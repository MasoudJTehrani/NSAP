"""Is SGClip deterministic, and how close to its decision boundaries does it run?

Two questions, because they have different fixes.

  repeat  Given the SAME image and the SAME frozen boxes, does SGClip return the same
          label every time? This is the question the step logs could not answer, because
          the logger records box labels and counts but not box coordinates, so a detector
          box that moved a few pixels was being blamed on SGClip.

  margin  How far apart are the top-1 and top-2 names? A head that is deterministic but
          decides by a hair is still unstable in deployment, because any upstream wobble
          in the box changes the crop and crosses the boundary. This is the mechanism we
          suspect behind `pot` being named `shadow` in 36% of deployed detector boxes.

  jitter  Move each frozen box by a few pixels, the way a non-deterministic detector
          would, and measure how often the label changes. This connects the two.

  python rung2/probes/determinism_sgclip.py --repeats 20 --jitter 2
"""
import argparse, collections, json, os, statistics, sys

sys.path.insert(0, "rung2")
os.environ.setdefault("NSAP_LASER_ATTR_REL", "1")
import harness  # noqa: F401
import numpy as np
import torch
from nav_harness import make_nav_planner
from functions.attack.sim_data import SimAttackDataset, explode_to_frames

QUERY = ["pot", "counter top", "stove burner", "sink", "cabinet"]
# the vocabulary ESCA assembles: proposed names plus its common_objects list (esca.py:206)
VOCAB = ["pot", "counter top", "counter", "stove", "cabinet", "table", "sink",
         "shadow", "wall", "window", "mirror", "door", "glass surface", "other"]


def top_per_box(P):
    d = getattr(P, "_nsap_predicted_name", {}) or {}
    out = {}
    for k, v in d.items():
        if not v:
            continue
        name1, p1 = str(v[0][1]).lower(), float(v[0][0])
        p2 = float(v[1][0]) if len(v) > 1 else 0.0
        out[str(k)] = (name1, p1, p1 - p2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--jitter", type=int, default=2, help="box jitter in pixels, 0 to skip")
    ap.add_argument("--sim-data", default="dataset/sim")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out", default="output/probes/determinism_sgclip.json")
    a = ap.parse_args()

    if a.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print("  could not force deterministic algorithms:", repr(e)[:90])

    P = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o")
    ds = SimAttackDataset(a.sim_data, target_category="Pot", target_relation="on",
                          data_source="pot", split=None)
    items = [fr for i in range(len(ds)) for fr in explode_to_frames(ds[i])][:a.frames]
    print(f"  frames: {len(items)}, repeats: {a.repeats}, jitter: {a.jitter}px\n", flush=True)

    rows, margins, flip_repeat, flip_jitter, n_boxes_tot = [], [], 0, 0, 0
    for fi, it in enumerate(items):
        fr = it["frames"][0]
        img = np.asarray(fr.detach().cpu() if hasattr(fr, "detach") else fr).astype(np.uint8)
        labels, boxes = P.predict_gd(img, QUERY, box_threshold=0.1, text_threshold=0.1)
        if len(boxes) == 0:
            continue
        frozen = np.asarray(boxes, dtype=float).copy()   # frozen for every repeat below

        # --- repeat: identical image, identical boxes
        seen = collections.defaultdict(set)
        for _ in range(a.repeats):
            P.predict_laser(image=img, boxes=frozen, target_names=["pot"], query_names=VOCAB,
                            query_attributes=[], query_relation_kws=["on"])
            for k, (name, p1, m) in top_per_box(P).items():
                seen[k].add(name)
                margins.append(m)
        unstable = sum(1 for v in seen.values() if len(v) > 1)
        flip_repeat += unstable
        n_boxes_tot += len(seen)

        # --- jitter: move each box a few pixels, as a wobbling detector would
        jit_unstable = 0
        if a.jitter:
            P.predict_laser(image=img, boxes=frozen, target_names=["pot"], query_names=VOCAB,
                            query_attributes=[], query_relation_kws=["on"])
            base = {k: v[0] for k, v in top_per_box(P).items()}
            rng = np.random.default_rng(0)
            H, W = img.shape[:2]
            for _ in range(5):
                jb = frozen + rng.integers(-a.jitter, a.jitter + 1, frozen.shape)
                # keep every box inside the frame and at least 4px on a side, or CLIP's
                # resize divides by zero on a degenerate crop
                jb[:, 0] = np.clip(jb[:, 0], 0, W - 5); jb[:, 1] = np.clip(jb[:, 1], 0, H - 5)
                jb[:, 2] = np.clip(jb[:, 2], jb[:, 0] + 4, W); jb[:, 3] = np.clip(jb[:, 3], jb[:, 1] + 4, H)
                P.predict_laser(image=img, boxes=jb.astype(float), target_names=["pot"],
                                query_names=VOCAB, query_attributes=[], query_relation_kws=["on"])
                for k, (name, _p, _m) in top_per_box(P).items():
                    if k in base and name != base[k]:
                        jit_unstable += 1
            flip_jitter += jit_unstable

        rows.append(dict(frame=fi, boxes=len(seen), unstable_on_repeat=unstable,
                         label_changes_under_jitter=jit_unstable))
        print(f"  frame {fi}: {len(seen)} boxes | changed label on an identical repeat: "
              f"{unstable} | label changes under {a.jitter}px jitter: {jit_unstable}", flush=True)

    tight = sum(1 for m in margins if m < 0.01)
    summ = dict(repeats=a.repeats, jitter_px=a.jitter, deterministic_mode=a.deterministic,
                boxes=n_boxes_tot, boxes_unstable_on_repeat=flip_repeat,
                label_changes_under_jitter=flip_jitter,
                margin_median=round(statistics.median(margins), 4) if margins else None,
                margin_p10=round(float(np.percentile(margins, 10)), 4) if margins else None,
                margin_under_0p01=f"{tight}/{len(margins)}" if margins else None)
    print("\n================ SGCLIP ================")
    print(f"  boxes whose label changed on an identical repeat : {flip_repeat}/{n_boxes_tot}")
    print(f"  label changes under {a.jitter}px box jitter               : {flip_jitter}")
    if margins:
        print(f"  top-1 minus top-2 probability, median            : {summ['margin_median']}")
        print(f"  ... 10th percentile                              : {summ['margin_p10']}")
        print(f"  ... decisions made on a margin under 0.01        : {summ['margin_under_0p01']}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": summ, "per_frame": rows}, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
