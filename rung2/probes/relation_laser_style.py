"""Relation head, tested the way the LASER demo uses it.

Our earlier probes scored EVERY object pair through rectangular regions, and the head
looked blind. The LASER demo does two things differently:
  - masks come from SAM2, i.e. precise object silhouettes, not filled boxes
  - relations are scored on SPECIFIED pairs (`related_object = [(5, 2)]`), not all pairs

This reproduces that setting using the simulator's exact instance masks (standing in for
SAM2) and asking about one chosen pair. Ground truth is AI2-THOR `parentReceptacles`.

Two questions:
  A. CLEAN: with the pot genuinely on the counter, does (pot,on,counter) beat (pot,on,stove)?
  B. ATTACK: does the trained patch change that, and does a random patch do the same?
"""
import os, sys, collections, json
sys.path.insert(0, "/home/usi/NSAP"); sys.path.insert(0, "/home/usi/NSAP/rung2")
os.environ.setdefault("NSAP_AGGR_THRES", "0.1")     # ESCA's own default, per the author
os.environ["NSAP_LASER_ATTR_REL"] = "1"
import numpy as np, torch
from nav_harness import make_nav_planner
from functions.attack.sim_data import SimAttackDataset, explode_to_frames
from functions.attack.patch import AdversarialPatch
from functions.eval._common import _patch_object_across_frames
from functions.attack.train import AttackConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"
REL = ["on", "not on"]

def np_frame(f):
    if torch.is_tensor(f): f = f.detach().cpu().numpy()
    return np.ascontiguousarray(np.clip(f, 0, 255).astype(np.uint8))

def mask_to_box(m):
    m = m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
    m = m.squeeze()
    ys, xs = np.nonzero(m)
    if len(xs) == 0: return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def score_pair(pm, frame, masks, names, pair_idx):
    """Score REL for one ordered pair, using PRECISE masks, as the demo does."""
    img = np_frame(frame)
    bm = [np.asarray(m.detach().cpu() if torch.is_tensor(m) else m).squeeze()[..., None].astype(bool)
          for m in masks]
    boxes = [mask_to_box(m) for m in masks]
    if any(b is None for b in boxes): return None
    oids = [[0, 0, i] for i in range(len(bm))]
    i, j = pair_idx
    try:
        _c, _u, binp, _d = pm(
            batched_video_ids=[0], batched_videos=[img], batched_masks=bm,
            batched_bboxes=torch.tensor(boxes), batched_names=[names],
            batched_object_ids=oids, batched_unary_kws=[[]], batched_binary_kws=[REL],
            batched_obj_pairs=[(0, 0, (i, j))], batched_video_splits=[0],
            batched_binary_predicates=[None])
    except Exception as e:
        print("   model call failed:", repr(e)[:120]); return None
    out = {}
    if binp and len(binp) > 0:
        for (fid, pr, rel), p in binp[0].items():
            out[rel] = p.item() if hasattr(p, "item") else float(p)
    return out

def main():
    p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o")
    pm = p.predicate_model
    print(f"  aggr_thres={p.aggr_thres}  attr_rel={p.use_laser_attr_rel}\n", flush=True)

    stats = collections.Counter(); rows = []
    for sid in [f"pot_scenario_{i:05d}" for i in range(1, 10)]:
        pt = (f"/home/usi/NSAP/output/chameleon_patches/relation_on_Pot/{sid}/"
              f"relation_on_Pot/relation_on_Pot_final.pt")
        if not os.path.isfile(pt): continue
        ck = torch.load(pt, map_location=DEV, weights_only=False)
        cfg: AttackConfig = ck["config"]; anchor = ck["anchor"]; cfg.sim_data_dir = "dataset/sim"
        ph, pw = ck["w"].shape[1], ck["w"].shape[2]
        trained = AdversarialPatch(ph, pw, torch.device(DEV)); trained.w.data.copy_(ck["w"]); trained.eval()
        rand = AdversarialPatch(ph, pw, torch.device(DEV)); rand.w.data.uniform_(-2.5, 2.5); rand.eval()

        ds = SimAttackDataset("dataset/sim", target_relation="on", split=None, only_scenarios=[sid])
        items = [fr for it in [ds[i] for i in range(len(ds))] for fr in explode_to_frames(it)]
        for it in items:
            gt = it.get("gt_categories") or {}
            m0 = it["masks"].get(0, {})
            pot = next((o for o in m0 if str(gt.get(o, "")).lower() == "pot"), None)
            rec = next((o for o in m0 if "counter" in str(gt.get(o, "")).lower()), None)
            if pot is None or rec is None: continue
            oid_list = [pot, rec]
            names = [str(gt.get(o, "object")).lower() for o in oid_list]
            masks = [m0[o] for o in oid_list]

            clean = score_pair(pm, it["frames"][0], masks, names, (0, 1))
            if not clean or "on" not in clean: continue
            stats["pairs"] += 1
            clean_on = clean["on"] > clean.get("not on", 1e9)
            stats["clean_says_on"] += clean_on
            if not clean_on: continue
            stats["attackable"] += 1
            for tag, pch in (("trained", trained), ("random", rand)):
                pf = _patch_object_across_frames(it["frames"], it["bboxes"], pch, anchor, cfg,
                                                 torch.device(DEV), pot)
                sc = score_pair(pm, pf[0], masks, names, (0, 1))
                if sc and "on" in sc:
                    flipped = not (sc["on"] > sc.get("not on", 1e9))
                    stats[f"{tag}_flip"] += flipped
                    if len(rows) < 10 and tag == "trained":
                        rows.append((sid, round(clean["on"],3), round(clean.get("not on",0),3),
                                     round(sc["on"],3), round(sc.get("not on",0),3), flipped))
        print(f"{sid}: pairs={stats['pairs']} clean_on={stats['clean_says_on']}", flush=True)

    n = max(stats["pairs"], 1); a = max(stats["attackable"], 1)
    print("\n===== RELATION, LASER-STYLE (precise masks, specified pair) =====")
    print(f"  pairs scored (pot, counter)          : {stats['pairs']}")
    print(f"  clean says 'on' (correct, it IS on)  : {stats['clean_says_on']} ({stats['clean_says_on']/n:.0%})")
    print(f"  of those, TRAINED patch flips        : {stats['trained_flip']}/{stats['attackable']} = {stats['trained_flip']/a:.0%}")
    print(f"  of those, RANDOM patch flips         : {stats['random_flip']}/{stats['attackable']} = {stats['random_flip']/a:.0%}")
    print("\n  examples (clean on/not-on -> patched on/not-on, flipped):")
    for r in rows: print(f"    {r[0]:20s} {r[1]:.3f}/{r[2]:.3f} -> {r[3]:.3f}/{r[4]:.3f}  {r[5]}")
    json.dump(dict(stats), open("output/probes/rel_laser_style.json","w"), indent=2)

if __name__ == "__main__":
    main()
