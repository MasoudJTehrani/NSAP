#!/usr/bin/env python
"""Deployment-faithful model-level evaluation.

The existing evaluation (`run_attack.py --eval`) scores the patch through
`VINE.predict` with the PRECISE AI2-THOR instance mask. The deployed agent never has
that: GroundingDINO returns boxes, and `esca.bbox_to_mask` turns each box into a filled
RECTANGLE. So the existing number is measured under a region deployment cannot produce.

This script keeps the precise mask for PLACING the patch, which is legitimate (the
attacker really can stick a patch on the object), and then scores the result through the
DEPLOYMENT path: predict_gd -> predict_laser -> the aggregation gate. It reports the two
numbers that matter and that the old evaluation conflates:

  clean grounding rate : how often the deployed stack grounds the target fact at all
  flip rate            : of those, how often the patch removes or changes it

`run_attack.py` and `functions/eval/*` are NOT modified. This is a parallel path.

  python rung2/deploy_eval.py --patch <patch.pt> --only-scenario <sid> [--sim-data dataset/sim]
  python rung2/deploy_eval.py --all-chameleons          # every trained patch, summarised
"""
import argparse
import collections
import glob
import json
import os
import sys

NSAP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NSAP)
sys.path.insert(0, os.path.join(NSAP, "rung2"))
os.environ.setdefault("NSAP_AGGR_THRES", "0.4")
os.environ["NSAP_LASER_ATTR_REL"] = "1"      # else attributes/relations never surface

import torch                                                    # noqa: E402
from nav_harness import make_nav_planner                        # noqa: E402
from functions.attack.train import AttackConfig                  # noqa: E402
from functions.attack.patch import AdversarialPatch              # noqa: E402
from functions.attack.sim_data import SimAttackDataset, explode_to_frames   # noqa: E402
from functions.eval._common import _patch_object_across_frames, _cat_ok     # noqa: E402
from functions.attack.data import largest_object_id              # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_COMPETITORS = int(os.environ.get("NSAP_N_COMPETITORS", "8"))   # detector query width
# Control condition. Replaces the trained patch with random pixels of the SAME shape, at the
# SAME anchor on the SAME object. Whatever churn this produces is the pipeline's response to
# any perturbation; the attack's real effect is its own numbers minus these.
RANDOM_PATCH = os.environ.get("NSAP_RANDOM_PATCH", "0") == "1"
# Evaluate over a MULTI-OBJECT graph, as the SGG attack literature does (top-15 objects and
# triples). ESCA deploys with top_k=1; that is faithful for the system-level runs but leaves
# no untargeted remainder, so isolation cannot be measured. Widening it here is a property of
# the EVALUATION, not of the deployed agent.
os.environ.setdefault("NSAP_TOP_K", "15")


def _np(frame):
    """sim_data yields float32 HWC tensors in 0-255; the detector wants uint8 numpy RGB,
    which is exactly what the live agent hands it."""
    import numpy as np
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    return np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- deployment scoring
def deployment_sg(planner, frame, target_name, query_names, attr=None, rel=None):
    """Run the agent's own perception path on one frame and return its scene graph.

    Mirrors what `esca.predict_laser` does in the live loop: detector boxes, rectangular
    region per box, contrastive grounding, aggregation gate.

    CASE MATTERS. predict_laser keeps a box only when `name in target_names`, and the
    predicted names are lowercase. Passing the config's capitalised target ("Pot",
    "Television") scores every box 0 and silently grounds nothing. The live agent is
    unaffected because the MLLM proposes lowercase concepts. Lowercase here to match it.
    """
    target_name = str(target_name).lower()
    query_names = [str(q).lower() for q in query_names]
    frame = _np(frame)
    # Concept list size matters more than it looks, because ESCA's aggregation gate keeps
    # only top_k=1 object (its own default, planner/navigation/esca.py:177). Querying just
    # the target makes it win the single slot unconditionally and hides all collateral;
    # querying everything starves it. The live agent asks about a few task-relevant objects,
    # so we query the target plus N_COMPETITORS others.
    gd_query = list(dict.fromkeys([target_name] + list(query_names)))[:1 + N_COMPETITORS]
    labels, boxes = planner.predict_gd(frame, gd_query,
                                       box_threshold=0.1, text_threshold=0.1)
    if boxes is None or len(boxes) == 0:
        return {"objects": [], "attrs": [], "rels": [], "boxes": []}
    lab, _bx, rels, attrs = planner.predict_laser(
        image=frame, boxes=boxes, target_names=gd_query,
        query_names=list(query_names) + ["other"],
        query_attributes=[attr] if attr else [],
        query_relation_kws=[rel] if rel else [])
    bx = []
    if _bx is not None:
        try:
            bx = [list(map(float, b)) for b in (_bx.tolist() if hasattr(_bx, "tolist") else _bx)]
        except Exception:
            bx = []
    return {"objects": lab or [], "attrs": attrs or [], "rels": rels or [], "boxes": bx}


def sg_facts(sg):
    """Flatten a deployment scene graph into comparable facts, bucketed by head."""
    objs, attrs, rels = set(), set(), set()
    names = []
    for x in sg["objects"]:
        n = x[1] if isinstance(x, (list, tuple)) and len(x) == 2 else x
        names.append(str(n).lower()); objs.add(str(n).lower())
    for i, lst in enumerate(sg["attrs"] or []):
        owner = names[i] if i < len(names) else f"obj{i}"
        for a in (lst or []):
            attrs.add((owner, str(a).lower()))
    for r in sg["rels"] or []:
        if len(r) == 3:
            rels.add((str(r[0]).lower(), str(r[1]).lower(), str(r[2]).lower()))
    return objs, attrs, rels


def isolation_buckets(sg_clean, sg_pat, target_cat):
    """Split every scene-graph change by HEAD and by whether it concerns the TARGET object.

    The original eval computes isolation over the patched object only. Counting changes
    anywhere in the graph as on-target is wrong: an attribute flipping on a different
    object, or an edge moving between two unrelated things, is collateral, not the attack.
    """
    co, ca, cr = sg_facts(sg_clean)
    po, pa, pr = sg_facts(sg_pat)

    def is_t(name):
        return _cat_ok(str(name), target_cat)

    obj = co ^ po
    att = ca ^ pa
    rel = cr ^ pr
    return {
        "target_category":  sum(1 for n in obj if is_t(n)),
        "other_category":   sum(1 for n in obj if not is_t(n)),
        "target_attribute": sum(1 for (o, _a) in att if is_t(o)),
        "other_attribute":  sum(1 for (o, _a) in att if not is_t(o)),
        "target_relation":  sum(1 for (sname, _r, _o) in rel if is_t(sname)),
        "other_relation":   sum(1 for (sname, _r, _o) in rel if not is_t(sname)),
    }


def target_pairs(sg, target_cat, rel):
    """The specific (subject, object) name pairs that hold the target relation.

    The original eval tracks a pair and asks whether THAT pair still holds the relation,
    so a redirect (pot,on,counter) -> (pot,on,stove) counts as a flip. Asking only whether
    any (pot,on,X) survives is a weaker test and is not comparable to it.
    """
    out = set()
    for r in sg["rels"] or []:
        if len(r) == 3 and str(r[1]).lower() == (rel or "").lower() \
           and _cat_ok(str(r[0]), target_cat):
            out.add((str(r[0]).lower(), str(r[2]).lower()))
    return out


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def fact_present_at(sg, gt_box, mode, target_cat, attr, rel, min_iou=0.3):
    """Is the targeted fact present ON THE SAME OBJECT INSTANCE?

    With top_k=15 the graph holds many objects, and several boxes can be labelled with the
    target category. Asking "does ANY television still have [displaying]" then reports no
    flip whenever a spurious second television carries it, which drove the attribute flip
    rate to 5%. Match the graph node to the ground-truth target box and test THAT node.
    """
    boxes = sg.get("boxes") or []
    names = []
    for x in sg["objects"]:
        n = x[1] if isinstance(x, (list, tuple)) and len(x) == 2 else x
        names.append(str(n).lower())
    best, bi = min_iou, -1
    for i, b in enumerate(boxes):
        if i >= len(names):
            break
        v = _iou(gt_box, b)
        if v > best:
            best, bi = v, i
    if bi < 0:
        return False                     # the target instance is not in the graph at all
    tname = names[bi]
    if mode == "category":
        return _cat_ok(tname, target_cat)
    if mode == "attribute":
        if not _cat_ok(tname, target_cat):
            return False
        lst = sg["attrs"][bi] if sg.get("attrs") and bi < len(sg["attrs"]) else []
        return (attr or "").lower() in {str(x).lower() for x in (lst or [])}
    if mode == "relation":
        for r in sg["rels"] or []:
            if len(r) == 3 and str(r[1]).lower() == (rel or "").lower() and str(r[0]).lower() == tname:
                return True
        return False
    return False


def fact_present(sg, mode, target_cat, attr, rel):
    """Is the attacked fact in this scene graph?"""
    names = {str(n).lower() for (_p, n) in
             [x if isinstance(x, (list, tuple)) and len(x) == 2 else (0, x) for x in sg["objects"]]}
    if mode == "category":
        return any(_cat_ok(n, target_cat) for n in names)
    if mode == "attribute":
        if not any(_cat_ok(n, target_cat) for n in names):
            return False
        flat = set()
        for lst in sg["attrs"]:
            flat |= {str(x).lower() for x in (lst or [])}
        return (attr or "").lower() in flat
    if mode == "relation":
        for r in sg["rels"]:
            if len(r) == 3 and str(r[1]).lower() == (rel or "").lower() \
               and _cat_ok(str(r[0]), target_cat):
                return True
        return False
    return False


# ---------------------------------------------------------------- one patch
def eval_patch(planner, patch_path, only_scenario, sim_data):
    ckpt = torch.load(patch_path, map_location=DEV, weights_only=False)
    config: AttackConfig = ckpt["config"]
    anchor = ckpt["anchor"]
    config.sim_data_dir = sim_data
    ph, pw = ckpt["w"].shape[1], ckpt["w"].shape[2]
    patch = AdversarialPatch(ph, pw, torch.device(DEV))
    if RANDOM_PATCH:
        # AdversarialPatch uses tanh parameterisation: pixels = 127.5*(tanh(w)+1).
        # Sample w wide enough to span the full pixel range rather than sitting at mid-grey.
        patch.w.data.uniform_(-2.5, 2.5)
    else:
        patch.w.data.copy_(ckpt["w"])
    patch.eval()

    mode = config.attack_mode
    cat = config.target_category
    attr = getattr(config, "target_attribute", "") or None
    rel = getattr(config, "target_relation", "") or None
    if mode == "relation" and not cat:
        cat = "pot"

    ds = SimAttackDataset(
        sim_data,
        target_category=cat if mode != "relation" else None,
        target_attribute=attr if mode == "attribute" else None,
        target_relation=rel if mode == "relation" else None,
        split=None, max_clips=getattr(config, "max_clips", 0),
        only_scenarios=[only_scenario] if only_scenario else None,
        data_source=getattr(config, "data_source", ""))
    items = [ds[i] for i in range(len(ds))]
    items = [fr for it in items for fr in explode_to_frames(it)]

    n_frames = n_clean = n_flip = 0
    buckets = collections.Counter()
    pair_total = pair_flipped = 0     # relation, pair-level (comparable to the original eval)
    for it in items:
        if it is None:
            continue
        frames, masks, bboxes = it["frames"], it["masks"], it["bboxes"]
        gt = it.get("gt_categories") or {}
        obj_ids = set(masks.get(0, {}).keys())
        if cat and gt:
            keep = {o for o in obj_ids if _cat_ok(gt.get(o, ""), cat)}
            obj_ids = keep or obj_ids
        oid = largest_object_id(masks, obj_ids)
        if oid is None:
            continue
        qnames = sorted({str(v).lower() for v in gt.values()})[:8] or [cat]
        n_frames += 1

        gb = (bboxes.get(0, {}) or {}).get(oid)
        gt_box = [gb["x1"], gb["y1"], gb["x2"], gb["y2"]] if isinstance(gb, dict) else None
        if gt_box is None:
            continue
        sg_clean = deployment_sg(planner, frames[0], cat, qnames, attr, rel)
        if not fact_present_at(sg_clean, gt_box, mode, cat, attr, rel):
            continue                        # deployment never grounded it -> nothing to flip
        n_clean += 1

        patched = _patch_object_across_frames(frames, bboxes, patch, anchor, config,
                                              torch.device(DEV), oid)
        sg_pat = deployment_sg(planner, patched[0], cat, qnames, attr, rel)
        if not fact_present_at(sg_pat, gt_box, mode, cat, attr, rel):
            n_flip += 1
        for k, v in isolation_buckets(sg_clean, sg_pat, cat).items():
            buckets[k] += v
        if mode == "relation":
            cp, pp = target_pairs(sg_clean, cat, rel), target_pairs(sg_pat, cat, rel)
            pair_total += len(cp)
            pair_flipped += len(cp - pp)        # that pair no longer holds the relation
    return {"mode": mode, "target": cat, "attr": attr, "rel": rel,
            "scenario": only_scenario, "n_frames": n_frames,
            "n_clean_grounded": n_clean, "n_flipped": n_flip,
            "clean_grounding_rate": (n_clean / n_frames) if n_frames else None,
            "flip_rate": (n_flip / n_clean) if n_clean else None,
            "changes": dict(buckets),
            "pair_total": pair_total, "pair_flipped": pair_flipped,
            "pair_flip_rate": (pair_flipped / pair_total) if pair_total else None,
            "isolation": ((buckets["target_" + mode] / sum(buckets.values()))
                          if sum(buckets.values()) else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="")
    ap.add_argument("--only-scenario", default="")
    ap.add_argument("--sim-data", default="dataset/sim")
    ap.add_argument("--all-chameleons", action="store_true")
    ap.add_argument("--out", default="output/deploy_eval.json")
    a = ap.parse_args()
    os.chdir(NSAP)

    jobs = []
    if a.all_chameleons:
        for p in sorted(glob.glob("output/chameleon_patches/*/*/*/*_final.pt")):
            sid = p.split("/")[3]
            # ebnav_scenario_XXXXX patches were trained on SceneGraphBench/data/kitchen_ebnav,
            # where the scenario is named scenario_XXXXX. Without this they resolve to zero frames.
            if sid.startswith("ebnav_scenario"):
                jobs.append((p, sid.replace("ebnav_", ""), "SceneGraphBench/data/kitchen_ebnav"))
            else:
                jobs.append((p, sid, a.sim_data))
    elif a.patch:
        jobs.append((a.patch, a.only_scenario, a.sim_data))
    else:
        ap.error("give --patch or --all-chameleons")

    planner = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o")
    print(f"  use_laser_attr_rel = {getattr(planner,'use_laser_attr_rel',None)}\n", flush=True)

    rows = []
    for path, sid, sdata in jobs:
        try:
            r = eval_patch(planner, path, sid, sdata)
        except Exception as e:
            print(f"  !! {sid}: {repr(e)[:140]}", flush=True); continue
        r["patch"] = path
        rows.append(r)
        cg = "n/a" if r["clean_grounding_rate"] is None else f"{r['clean_grounding_rate']:.0%}"
        fr = "n/a" if r["flip_rate"] is None else f"{r['flip_rate']:.0%}"
        iso = "n/a" if r["isolation"] is None else f"{r['isolation']:.0%}"
        pf = "" if r.get("pair_flip_rate") is None else \
             f"  pair-flip={r['pair_flipped']}/{r['pair_total']} = {r['pair_flip_rate']:.0%}"
        print(f"  {r['mode']:10s} {sid:22s} frames={r['n_frames']:>3} "
              f"clean-grounded={r['n_clean_grounded']:>3} ({cg:>4})  flip={fr:>4} "
              f"({r['n_flipped']}/{r['n_clean_grounded']})  iso={iso:>4}{pf}", flush=True)

    print("\n================ DEPLOYMENT-FAITHFUL MODEL-LEVEL EVAL ================")
    print("(patch placed with the precise mask; scored through predict_gd -> predict_laser)")
    by, bk = {}, {}
    for r in rows:
        b = by.setdefault(r["mode"], [0, 0, 0])
        b[0] += r["n_frames"]; b[1] += r["n_clean_grounded"]; b[2] += r["n_flipped"]
        b.append(0) if False else None
        acc = bk.setdefault(r["mode"], collections.Counter())
        by.setdefault("_pairs_" + r["mode"], [0, 0])
        by["_pairs_" + r["mode"]][0] += r.get("pair_total", 0) or 0
        by["_pairs_" + r["mode"]][1] += r.get("pair_flipped", 0) or 0
        for k, v in (r.get("changes") or {}).items():
            acc[k] += v
    for m in [k for k in by if not k.startswith("_pairs_")]:
        nf, nc, nfl = by[m]
        acc = bk[m]; tot = sum(acc.values())
        pt, pfl = by.get("_pairs_" + m, [0, 0])
        if pt:
            print(f"  {m:10s} PAIR-LEVEL flip (comparable to the oracle eval): "
                  f"{pfl}/{pt} = {pfl/pt:.0%}")
        on = acc["target_" + m]
        print(f"  {m:10s} frames={nf:>4}  clean-grounded={nc:>4} ({nc/max(nf,1):.0%})  "
              f"flip={nfl}/{nc} = {(nfl/nc) if nc else 0:.0%}  "
              f"isolation={on}/{tot} = {(on/tot) if tot else 0:.0%}")
        print(f"      ON-TARGET  cat={acc['target_category']} attr={acc['target_attribute']} rel={acc['target_relation']}")
        print(f"      COLLATERAL cat={acc['other_category']} attr={acc['other_attribute']} rel={acc['other_relation']}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
