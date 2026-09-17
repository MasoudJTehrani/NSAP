#!/usr/bin/env python
"""Per-head OUTCOME analysis of the attack, plus a SAM2-mask variant.

Two things the existing evaluation does not do.

1. It records THAT a fact changed, not WHAT it became. Our attack is untargeted: the
   objective only pushes the grounded fact past its decision boundary and never
   prescribes a replacement. So what it turns into is an empirical question, and the
   answer differs by head:
       category   relabelled to X, or the object leaves the graph
       attribute  only negation is available (the head is a two-way contrast)
       relation   deleted, redirected to a different object, or the predicate changes
   A redirect matters more than a deletion: the graph stays well-formed and plausible,
   so nothing downstream looks broken.

2. It scores ground-truth instance masks. LASER's own demo feeds SGClip masks produced
   by SAM2 from GroundingDINO boxes. `--masks sam2` reproduces that, so we can show
   whether the oracle masks flatter the attack.

Existing code is untouched; this is a parallel path.

  python rung2/outcome_eval.py --patch <p.pt> --only-scenario <sid> [--masks gt|sam2] [--viz]
  python rung2/outcome_eval.py --all            # every chameleon + universal, both mask modes
"""
import argparse, glob, json, os, sys, collections
NSAP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NSAP)
os.chdir(NSAP)
import numpy as np
import torch
from PIL import Image, ImageDraw

from functions.attack.train import AttackConfig                              # noqa: E402
from functions.attack.patch import AdversarialPatch                          # noqa: E402
from functions.attack.sim_data import SimAttackDataset, explode_to_frames    # noqa: E402
from functions.eval._common import _patch_object_across_frames, _cat_ok      # noqa: E402
from functions.eval.scene_graph import predict_scene_graph                   # noqa: E402
from functions.attack.data import largest_object_id                          # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_SAM = {"p": None}


def sam_predictor():
    if _SAM["p"] is None:
        from functions.download_model import SAM2_CHECKPOINT, SAM2_CONFIG
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _SAM["p"] = SAM2ImagePredictor(build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=DEV))
        print("  SAM2 loaded", flush=True)
    return _SAM["p"]


def to_uint8(frame):
    f = frame.detach().cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
    return np.ascontiguousarray(np.clip(f, 0, 255).astype(np.uint8))


def sam2_masks(frame, bboxes_for_frame, gt_masks):
    """Replace each ground-truth mask with SAM2's, prompted by that object's box.
    This is the LASER demo's region source. Falls back to the GT mask on failure."""
    pred = sam_predictor()
    img = to_uint8(frame)
    out = {}
    try:
        pred.set_image(img)
    except Exception as e:
        print(f"   SAM2 set_image failed: {repr(e)[:80]}"); return gt_masks
    for oid, bb in (bboxes_for_frame or {}).items():
        if oid not in gt_masks:
            continue
        try:
            box = np.array([bb["x1"], bb["y1"], bb["x2"], bb["y2"]], dtype=float)
            m, scores, _ = pred.predict(box=box, multimask_output=True)
            best = m[int(np.argmax(scores))].astype(bool)
            if best.sum() < 20:                       # degenerate; keep the GT mask
                out[oid] = gt_masks[oid]
            else:
                ref = gt_masks[oid]
                t = torch.from_numpy(best)
                out[oid] = t.to(ref.device) if torch.is_tensor(ref) else t
        except Exception:
            out[oid] = gt_masks[oid]
    for oid in gt_masks:
        out.setdefault(oid, gt_masks[oid])
    return out


def outcomes(sg_c, sg_p, oid, mode, cfg):
    """What happened to the attacked fact on this object? Returns (kind, detail)."""
    if mode == "category":
        c, p = sg_c.categories.get(oid), sg_p.categories.get(oid)
        if p is None:
            return "vanished", None
        if c == p:
            return "unchanged", None
        return "relabelled", f"{c} -> {p}"
    if mode == "attribute":
        c, p = sg_c.attributes.get(oid), sg_p.attributes.get(oid)
        if p is None:
            return "vanished", None
        if c == p:
            return "unchanged", None
        return "negated", f"{c} -> {p}"
    # relation: look at every clean edge whose subject is oid
    tgt = (getattr(cfg, "target_relation", "") or "").lower()
    res = []
    for (s, o), rel in sg_c.relations.items():
        if s != oid or rel.lower() != tgt:
            continue
        pr = sg_p.relations.get((s, o))
        if pr is None:
            moved = [oo for (ss, oo), rr in sg_p.relations.items()
                     if ss == oid and rr.lower() == tgt and oo != o]
            res.append(("redirected", f"{o} -> {moved[0]}") if moved else ("deleted", str(o)))
        elif pr.lower() != tgt:
            res.append(("predicate_changed", f"{rel} -> {pr}"))
        else:
            res.append(("unchanged", None))
    return res or [("no_target_edge", None)]


def clean_has_fact(sg, oid, mode, cfg):
    """Did the CLEAN model ground the targeted fact on this object? This is the flip-rate
    denominator, matching run_attack.py --eval: only facts that exist can be flipped."""
    if mode == "category":
        return _cat_ok(sg.categories.get(oid, ""), cfg.target_category)
    if mode == "attribute":
        return (sg.attributes.get(oid) or "").lower() == (getattr(cfg, "target_attribute", "") or "").lower()
    tgt = (getattr(cfg, "target_relation", "") or "").lower()
    return any(s_ == oid and r.lower() == tgt for (s_, o_), r in sg.relations.items())


def flip_and_isolation(sg_c, sg_p, oid, mode, cfg):
    """Returns (flipped, buckets). Buckets count changes to THIS object, by head,
    exactly as the existing evaluation does."""
    flipped = clean_has_fact(sg_c, oid, mode, cfg) and not clean_has_fact(sg_p, oid, mode, cfg)
    b = {"category": 0, "attribute": 0, "relation": 0, "disappeared": 0}
    if sg_p.categories.get(oid) is None:
        b["disappeared"] = 1
    else:
        if sg_c.categories.get(oid) != sg_p.categories.get(oid):
            b["category"] = 1
        if sg_c.attributes.get(oid) != sg_p.attributes.get(oid):
            b["attribute"] = 1
    subj = {(s_, o_) for (s_, o_) in sg_c.relations if s_ == oid} | \
           {(s_, o_) for (s_, o_) in sg_p.relations if s_ == oid}
    b["relation"] = sum(1 for pr in subj if sg_c.relations.get(pr) != sg_p.relations.get(pr))
    return flipped, b


def draw(frame, sg, oid, masks, path, title):
    img = Image.fromarray(to_uint8(frame)).convert("RGB")
    d = ImageDraw.Draw(img)
    m = masks.get(oid)
    if m is not None:
        a = (m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)).squeeze()
        ys, xs = np.nonzero(a)
        if len(xs):
            d.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                        outline=(255, 40, 40), width=3)
    lines = [title,
             f"cat: {sg.categories.get(oid)}",
             f"attr: {sg.attributes.get(oid)}"]
    rels = [f"({s},{r},{o})" for (s, o), r in sg.relations.items() if s == oid][:3]
    lines += ["rel: " + (", ".join(rels) if rels else "-")]
    d.rectangle([0, 0, img.width, 16 * len(lines) + 6], fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((4, 3 + 16 * i), t[:90], fill=(255, 255, 255))
    img.save(path)


def run_one(vine, patch_path, sid, sim_data, mask_mode, viz_dir, split):
    ck = torch.load(patch_path, map_location=DEV, weights_only=False)
    cfg: AttackConfig = ck["config"]; anchor = ck["anchor"]
    cfg.sim_data_dir = sim_data
    ph, pw = ck["w"].shape[1], ck["w"].shape[2]
    patch = AdversarialPatch(ph, pw, torch.device(DEV))
    patch.w.data.copy_(ck["w"]); patch.eval()
    mode = cfg.attack_mode
    _uk = ([cfg.target_attribute, f"not {cfg.target_attribute}"]
           if mode == "attribute" and getattr(cfg, "target_attribute", "") else None)
    _bk = ([cfg.target_relation, f"not {cfg.target_relation}"]
           if mode == "relation" and getattr(cfg, "target_relation", "") else None)

    ds = SimAttackDataset(
        sim_data,
        target_category=cfg.target_category if mode != "relation" else None,
        target_attribute=cfg.target_attribute if mode == "attribute" else None,
        target_relation=cfg.target_relation if mode == "relation" else None,
        split=(None if sid else split), max_clips=getattr(cfg, "max_clips", 0),
        only_scenarios=[sid] if sid else None,
        data_source=getattr(cfg, "data_source", ""))
    items = [fr for it in [ds[i] for i in range(len(ds))] for fr in explode_to_frames(it)]

    tally = collections.Counter(); details = collections.Counter(); n = 0
    n_clean = n_flip = 0; buckets = collections.Counter()
    for k, it in enumerate(items):
        masks, bboxes = it["masks"], it["bboxes"]
        if mask_mode == "sam2":
            masks = {f: sam2_masks(it["frames"][f] if f < len(it["frames"]) else it["frames"][0],
                                   bboxes.get(f, {}), mm) for f, mm in masks.items()}
        gt = it.get("gt_categories") or {}
        cand = set(masks.get(0, {}).keys())
        if cfg.target_category and gt:
            keep = {o for o in cand if _cat_ok(gt.get(o, ""), cfg.target_category)}
            cand = keep or cand
        oid = largest_object_id(masks, cand)
        if oid is None:
            continue
        sg_c = predict_scene_graph(vine, it["frames"], masks, bboxes, it["gpt_spec"],
                                   torch.device(DEV), unary_kws_override=_uk, binary_kws_override=_bk)
        pf = _patch_object_across_frames(it["frames"], bboxes, patch, anchor, cfg,
                                         torch.device(DEV), oid)
        sg_p = predict_scene_graph(vine, pf, masks, bboxes, it["gpt_spec"],
                                   torch.device(DEV), unary_kws_override=_uk, binary_kws_override=_bk)
        n += 1
        if clean_has_fact(sg_c, oid, mode, cfg):
            n_clean += 1
            fl, bk = flip_and_isolation(sg_c, sg_p, oid, mode, cfg)
            n_flip += int(fl)
            for kk, vv in bk.items():
                buckets[kk] += vv
        res = outcomes(sg_c, sg_p, oid, mode, cfg)
        for kind, det in (res if isinstance(res, list) else [res]):
            tally[kind] += 1
            if det: details[f"{kind}: {det}"] += 1
        if viz_dir and k < 4:
            os.makedirs(viz_dir, exist_ok=True)
            draw(it["frames"][0], sg_c, oid, masks.get(0, {}),
                 os.path.join(viz_dir, f"f{k:02d}_clean.png"), f"CLEAN [{mask_mode}]")
            draw(pf[0], sg_p, oid, masks.get(0, {}),
                 os.path.join(viz_dir, f"f{k:02d}_patched.png"), f"PATCHED [{mask_mode}]")
    on = buckets[mode] + (buckets["disappeared"] if mode == "category" else 0)
    tot = sum(buckets.values())
    return {"patch": patch_path, "scenario": sid or f"split={split}", "mode": mode,
            "masks": mask_mode, "n_objects": n,
            "n_clean_grounded": n_clean, "n_flipped": n_flip,
            "flip_rate": (n_flip / n_clean) if n_clean else None,
            "iso_buckets": dict(buckets),
            "isolation": (on / tot) if tot else None,
            "outcomes": dict(tally), "details": dict(details.most_common(12))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default=""); ap.add_argument("--only-scenario", default="")
    ap.add_argument("--sim-data", default="dataset/sim")
    ap.add_argument("--masks", default="gt", choices=["gt", "sam2"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="output/outcome_eval.json")
    a = ap.parse_args()

    from run_attack import load_vine
    vine = load_vine(DEV)

    jobs = []
    if a.all:
        for p in sorted(glob.glob("output/chameleon_patches/*/*/*/*_final.pt")):
            sid = p.split("/")[3]
            sd = ("SceneGraphBench/data/kitchen_ebnav" if sid.startswith("ebnav_scenario")
                  else a.sim_data)
            jobs.append((p, sid.replace("ebnav_", ""), sd, "chameleon"))
        for p in sorted(glob.glob("output/universal_patches/*/*_final.pt")):
            jobs.append((p, "", a.sim_data, "universal"))
    else:
        jobs.append((a.patch, a.only_scenario, a.sim_data, "single"))

    rows = []
    for mask_mode in (["gt", "sam2"] if a.all else [a.masks]):
        for path, sid, sd, kind in jobs:
            vd = (os.path.join("output/outcome_viz", mask_mode, kind,
                               sid or os.path.basename(os.path.dirname(path)))
                  if (a.viz or a.all) else "")
            try:
                r = run_one(vine, path, sid, sd, mask_mode, vd, a.split)
            except Exception as e:
                print(f"  !! {kind} {sid}: {repr(e)[:130]}", flush=True); continue
            r["kind"] = kind
            rows.append(r)
            fr = "n/a" if r["flip_rate"] is None else f"{r['flip_rate']:.0%}"
            iso = "n/a" if r["isolation"] is None else f"{r['isolation']:.0%}"
            print(f"  [{mask_mode:4s}] {kind:10s} {r['mode']:10s} {str(sid)[:20]:20s} "
                  f"clean={r['n_clean_grounded']:>3} flip={r['n_flipped']:>3} ({fr:>4}) "
                  f"iso={iso:>4}  {r['outcomes']}", flush=True)

    print("\n================ PER-HEAD OUTCOMES ================")
    for mask_mode in sorted({r["masks"] for r in rows}):
        for kind in sorted({r["kind"] for r in rows}):
            for mode in ("category", "attribute", "relation"):
                sel = [r for r in rows if r["masks"] == mask_mode and r["kind"] == kind
                       and r["mode"] == mode]
                if not sel:
                    continue
                agg = collections.Counter()
                for r in sel:
                    agg.update(r["outcomes"])
                tot = sum(agg.values())
                parts = ", ".join(f"{k} {v} ({v/tot:.0%})" for k, v in agg.most_common())
                nc = sum(r["n_clean_grounded"] for r in sel)
                nf = sum(r["n_flipped"] for r in sel)
                bk = collections.Counter()
                for r in sel:
                    bk.update(r.get("iso_buckets") or {})
                on = bk[mode] + (bk["disappeared"] if mode == "category" else 0)
                bt = sum(bk.values())
                print(f"  [{mask_mode:4s}] {kind:10s} {mode:10s} "
                      f"clean-grounded={nc:>4}  flip={nf}/{nc} = {(nf/nc) if nc else 0:>4.0%}  "
                      f"isolation={on}/{bt} = {(on/bt) if bt else 0:>4.0%}")
                print(f"       outcomes (n={tot}): {parts}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print("visualisations under output/outcome_viz/<masks>/<kind>/<scenario>/")


if __name__ == "__main__":
    main()
