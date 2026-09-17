#!/usr/bin/env python
"""RQ: how visually inconspicuous is the chameleon patch?

Measures LPIPS (learned perceptual image patch similarity) between each CLEAN frame and
the same frame with a patch pasted on the target object. Lower means more similar.

Two patch types on identical frames, anchors and coverage, so the comparison is controlled:
  chameleon  Stage 2, object-initialised with the content loss (the stealth claim)
  universal  Stage 1, the loud patch it warm-starts from (the control)

Reports WHOLE-IMAGE LPIPS as the headline, and patch-region LPIPS alongside. The whole-image
value is diluted, because the patch covers a small fraction of the frame and LPIPS pools
spatially, so it should be read as a RELATIVE comparison between the two patch types rather
than as an absolute imperceptibility claim.
"""
import argparse, glob, json, os, sys, collections
NSAP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NSAP); os.chdir(NSAP)
import numpy as np
import torch
import lpips

from functions.attack.train import AttackConfig                             # noqa: E402
from functions.attack.patch import AdversarialPatch                         # noqa: E402
from functions.attack.sim_data import SimAttackDataset, explode_to_frames   # noqa: E402
from functions.eval._common import _patch_object_across_frames, _cat_ok     # noqa: E402
from functions.attack.data import largest_object_id                         # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
COVERAGE = None   # set by --coverage; forces both patch types to the same pasted size
_NET = {"m": None}


def net():
    if _NET["m"] is None:
        _NET["m"] = lpips.LPIPS(net="alex").to(DEV)     # AlexNet backbone, the LPIPS default
    return _NET["m"]


def to_t(img):
    """HWC 0-255 -> 1x3xHxW in [-1,1], which is what LPIPS expects."""
    a = img.detach().cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
    a = np.clip(a, 0, 255).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(DEV)


def lpips_of(a, b):
    with torch.no_grad():
        return float(net()(to_t(a), to_t(b)).item())


def crop(img, box, pad=8):
    a = img.detach().cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
    h, w = a.shape[:2]
    x1 = max(0, int(box["x1"]) - pad); y1 = max(0, int(box["y1"]) - pad)
    x2 = min(w, int(box["x2"]) + pad); y2 = min(h, int(box["y2"]) + pad)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return a[y1:y2, x1:x2]


def run(patch_path, sid, sim_data, kind, limit=12):
    ck = torch.load(patch_path, map_location=DEV, weights_only=False)
    cfg: AttackConfig = ck["config"]; anchor = ck["anchor"]; cfg.sim_data_dir = sim_data
    # MATCH THE PASTED SIZE. compute_patch_size scales the patch to coverage_ratio * bbox
    # per SIDE, so the universals' 0.3 against the chameleons' 0.5 is 2.78x the AREA.
    # Comparing visual similarity across different patch sizes measures size, not camouflage.
    if COVERAGE is not None:
        cfg.coverage_ratio = COVERAGE
    ph, pw = ck["w"].shape[1], ck["w"].shape[2]
    p = AdversarialPatch(ph, pw, torch.device(DEV)); p.w.data.copy_(ck["w"]); p.eval()
    mode = cfg.attack_mode
    ds = SimAttackDataset(
        sim_data,
        target_category=cfg.target_category if mode != "relation" else None,
        target_attribute=cfg.target_attribute if mode == "attribute" else None,
        target_relation=cfg.target_relation if mode == "relation" else None,
        split=(None if sid else "test"), max_clips=getattr(cfg, "max_clips", 0),
        only_scenarios=[sid] if sid else None, data_source=getattr(cfg, "data_source", ""))
    items = [fr for it in [ds[i] for i in range(len(ds))] for fr in explode_to_frames(it)][:limit]
    full, region = [], []
    for it in items:
        gt = it.get("gt_categories") or {}
        cand = set(it["masks"].get(0, {}).keys())
        if cfg.target_category and gt:
            keep = {o for o in cand if _cat_ok(gt.get(o, ""), cfg.target_category)}
            cand = keep or cand
        oid = largest_object_id(it["masks"], cand)
        if oid is None:
            continue
        pf = _patch_object_across_frames(it["frames"], it["bboxes"], p, anchor, cfg,
                                         torch.device(DEV), oid)
        clean, patched = it["frames"][0], pf[0]
        full.append(lpips_of(clean, patched))
        bb = (it["bboxes"].get(0, {}) or {}).get(oid)
        if isinstance(bb, dict):
            ca, cb = crop(clean, bb), crop(patched, bb)
            if ca is not None and cb is not None:
                region.append(lpips_of(ca, cb))
    # Keep the PER-VIEWPOINT values, not just the mean. Camouflage is viewpoint dependent,
    # and the spread is itself a result. Selecting the best frames and reporting only those
    # would be selecting on the reported metric, so we report the distribution instead.
    f = sorted(full)
    return {"patch": patch_path, "kind": kind, "mode": mode, "scenario": sid or "test-split",
            "n": len(full), "per_frame": full,
            "lpips_full": float(np.mean(full)) if full else None,
            "lpips_full_std": float(np.std(full)) if full else None,
            "lpips_best": float(f[0]) if f else None,
            "lpips_p25": float(np.percentile(f, 25)) if f else None,
            "lpips_median": float(np.median(f)) if f else None,
            "lpips_worst": float(f[-1]) if f else None,
            "lpips_region": float(np.mean(region)) if region else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-data", default="dataset/sim")
    ap.add_argument("--out", default="output/stealth_eval.json")
    ap.add_argument("--coverage", type=float, default=0.5,
                    help="force both patch types to this coverage so the comparison is size-matched")
    a = ap.parse_args()
    global COVERAGE
    COVERAGE = a.coverage
    print(f"  coverage forced to {COVERAGE} for BOTH patch types (size-matched)\n", flush=True)

    jobs = []
    for p in sorted(glob.glob("output/chameleon_patches/*/*/*/*_final.pt")):
        sid = p.split("/")[3]
        sd = ("SceneGraphBench/data/kitchen_ebnav" if sid.startswith("ebnav_scenario") else a.sim_data)
        jobs.append((p, sid.replace("ebnav_", ""), sd, "chameleon"))
    for p in sorted(glob.glob("output/universal_patches/*/*_final.pt")):
        jobs.append((p, "", a.sim_data, "universal"))

    rows = []
    for path, sid, sd, kind in jobs:
        try:
            r = run(path, sid, sd, kind)
        except Exception as e:
            print(f"  !! {kind} {sid}: {repr(e)[:120]}", flush=True); continue
        rows.append(r)
        lf = "n/a" if r["lpips_full"] is None else f"{r['lpips_full']:.4f}"
        lr = "n/a" if r["lpips_region"] is None else f"{r['lpips_region']:.4f}"
        print(f"  {kind:10s} {r['mode']:10s} {str(sid)[:20]:20s} n={r['n']:>2}  "
              f"LPIPS full={lf}  region={lr}", flush=True)

    print("\n================ STEALTH (LPIPS, lower = more similar) ================")
    for mode in ("category", "attribute", "relation"):
        for kind in ("universal", "chameleon"):
            sel = [r for r in rows if r["mode"] == mode and r["kind"] == kind and r["lpips_full"] is not None]
            if not sel:
                continue
            f = float(np.mean([r["lpips_full"] for r in sel]))
            g = [r["lpips_region"] for r in sel if r["lpips_region"] is not None]
            rg = float(np.mean(g)) if g else float("nan")
            allf = [x for r in sel for x in (r.get("per_frame") or [])]
            allf.sort()
            b = allf[0] if allf else float("nan")
            p25 = float(np.percentile(allf, 25)) if allf else float("nan")
            med = float(np.median(allf)) if allf else float("nan")
            wst = allf[-1] if allf else float("nan")
            print(f"  {mode:10s} {kind:10s} patches={len(sel):>2} frames={len(allf):>3}  "
                  f"mean={f:.4f}  best={b:.4f}  p25={p25:.4f}  median={med:.4f}  worst={wst:.4f}")
        ch = [r for r in rows if r["mode"] == mode and r["kind"] == "chameleon" and r["lpips_full"] is not None]
        un = [r for r in rows if r["mode"] == mode and r["kind"] == "universal" and r["lpips_full"] is not None]
        if ch and un:
            c, u = np.mean([r["lpips_full"] for r in ch]), np.mean([r["lpips_full"] for r in un])
            cf = sorted(x for r in ch for x in (r.get("per_frame") or []))
            uf = sorted(x for r in un for x in (r.get("per_frame") or []))
            def pct(a, b):
                return (b - a) / b * 100 if b else float("nan")
            print(f"             -> chameleon closer by {pct(c, u):.0f}% on the MEAN, "
                  f"{pct(np.percentile(cf,25), np.percentile(uf,25)):.0f}% at the best quartile, "
                  f"{pct(cf[0], uf[0]):.0f}% at the single best viewpoint")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
