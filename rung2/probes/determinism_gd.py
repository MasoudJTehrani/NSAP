"""Is GroundingDINO deterministic, and if not, where does the instability enter?

The step logs say the detector disagrees with itself on 46% of identical step-0 frames
given an identical query. A forward pass under no_grad should not do that, so this probe
separates two different things:

  raw      are the model's output tensors bit-identical across repeated calls?
  decided  are the boxes that survive `max(logits) > box_threshold` the same?

These can differ. `planner/navigation/esca.py:517` thresholds the logits, so a box whose
score sits within floating-point noise of the threshold flips in and out between runs even
when the tensors differ only in the last bits. That is amplification, not model
non-determinism, and it is fixed differently.

  python rung2/probes/determinism_gd.py --repeats 20
  python rung2/probes/determinism_gd.py --repeats 20 --deterministic   # force determinism
"""
import argparse, collections, json, os, sys

sys.path.insert(0, "rung2")
os.environ.setdefault("NSAP_LASER_ATTR_REL", "1")
import harness  # noqa: F401  (sets ESCA_ROOT, sys.path, env vars, chdir)
import numpy as np
import torch
from nav_harness import make_nav_planner
from functions.attack.sim_data import SimAttackDataset, explode_to_frames

QUERY = ["pot", "counter top", "stove burner", "sink", "cabinet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--frames", type=int, default=8, help="how many distinct frames to test")
    ap.add_argument("--sim-data", default="dataset/sim")
    ap.add_argument("--box-threshold", type=float, default=0.1)
    ap.add_argument("--text-threshold", type=float, default=0.1)
    ap.add_argument("--deterministic", action="store_true",
                    help="disable TF32 and cuDNN autotuning and force deterministic kernels")
    ap.add_argument("--out", default="output/probes/determinism_gd.json")
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
    print(f"  cudnn.benchmark={torch.backends.cudnn.benchmark} "
          f"tf32_matmul={torch.backends.cuda.matmul.allow_tf32}", flush=True)

    P = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o")
    import embodiedbench.planner.navigation.esca as esca_mod

    ds = SimAttackDataset(a.sim_data, target_category="Pot", target_relation="on",
                          data_source="pot", split=None)
    items = [fr for i in range(len(ds)) for fr in explode_to_frames(ds[i])][:a.frames]
    print(f"  frames: {len(items)}, repeats: {a.repeats}\n", flush=True)

    rows = []
    for fi, it in enumerate(items):
        fr = it["frames"][0]
        img = np.asarray(fr.detach().cpu() if hasattr(fr, "detach") else fr).astype(np.uint8)

        raw_sigs, decided = [], []
        near_thresh = None
        for _ in range(a.repeats):
            # (1) the raw forward pass, exactly as predict_gd runs it
            cap = esca_mod.preprocess_caption(". ".join(QUERY))
            proc = P.grounding_model.preprocess_image(image_bgr=img).to(P.grounding_model.device)
            with torch.no_grad():
                out = P.grounding_model.model(proc.unsqueeze(0), captions=cap)
            lg = out["pred_logits"].cpu().sigmoid()[0]
            bx = out["pred_boxes"].cpu()[0]
            raw_sigs.append((lg.numpy().tobytes(), bx.numpy().tobytes()))
            # how many query slots sit within 1e-4 of the decision threshold
            m = lg.max(dim=1)[0]
            if near_thresh is None:
                near_thresh = int(((m - a.box_threshold).abs() < 1e-4).sum())
            # (2) the decision the pipeline actually takes
            labels, boxes = P.predict_gd(img, QUERY, box_threshold=a.box_threshold,
                                         text_threshold=a.text_threshold)
            b = np.asarray(boxes, dtype=float) if len(boxes) else np.zeros((0, 4))
            decided.append((tuple(labels), b.round(3).tobytes(), len(labels)))

        raw_same = len(set(raw_sigs)) == 1
        lab_same = len({d[0] for d in decided}) == 1
        box_same = len({d[1] for d in decided}) == 1
        counts = collections.Counter(d[2] for d in decided)
        rows.append(dict(frame=fi, raw_identical=raw_same, labels_identical=lab_same,
                         boxes_identical=box_same, box_counts=dict(counts),
                         distinct_raw=len(set(raw_sigs)), near_threshold_slots=near_thresh))
        print(f"  frame {fi}: raw tensors identical={raw_same}  labels identical={lab_same}  "
              f"boxes identical={box_same}  box counts seen={dict(counts)}", flush=True)

    n = len(rows)
    summ = dict(
        repeats=a.repeats, frames=n, deterministic_mode=a.deterministic,
        frames_raw_identical=sum(r["raw_identical"] for r in rows),
        frames_labels_identical=sum(r["labels_identical"] for r in rows),
        frames_boxes_identical=sum(r["boxes_identical"] for r in rows),
        frames_box_count_varies=sum(len(r["box_counts"]) > 1 for r in rows),
    )
    print("\n================ GROUNDING DINO ================")
    print(f"  raw output tensors identical across {a.repeats} calls : "
          f"{summ['frames_raw_identical']}/{n} frames")
    print(f"  surviving boxes identical                            : "
          f"{summ['frames_boxes_identical']}/{n} frames")
    print(f"  labels identical                                     : "
          f"{summ['frames_labels_identical']}/{n} frames")
    print(f"  number of boxes varied between calls                 : "
          f"{summ['frames_box_count_varies']}/{n} frames")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": summ, "per_frame": rows}, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
