#!/usr/bin/env python
"""
Offline verification for the ESCA rung-2 head re-enable.

Two things to confirm without any API / AI2-THOR / GroundingDINO / SAM2:
  1. The trained VINE weights load into the predicate model with the RIGHT CLIP
     backbone (patch32, not ESCA's hardcoded patch16).  -> checked by loading.
  2. The attribute (unary) + relation (binary) heads return SENSIBLE grounded
     output on real SceneGraphBench frames (so the re-enabled heads actually carry
     the signal the chameleon patches attack).

We run the heads through NSAP's VineModel.predict (the forward that works in this
env; the raw PredicateModel.forward that ESCA calls runs in ESCA's pinned env).
Same weights, same three heads, same vocab as the edited predict_laser.

Run: conda run -n NSAP python scripts/verify_esca_heads.py
"""
import os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src/LASER"))

from model.vine.vine_config import VineConfig
from model.vine.vine_model import VineModel
from functions.eval.scene_graph import predict_scene_graph
from functions.attack.sim_data import SimAttackDataset

DATA = "SceneGraphBench/data/kitchen_pilot"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = VineConfig(use_hf_repo=False,
                    local_dir=os.path.join(ROOT, "model", "vine"),
                    local_filename="model.safetensors", device=str(device))
vine = VineModel(config).to(device)
vine.eval()
for p in vine.parameters():
    p.requires_grad_(False)


def run_frame(scenario_id):
    ds = SimAttackDataset(DATA, only_scenarios=[scenario_id])
    item = ds[0]
    sgj = ds._load_sg(ds.scenarios[0])
    tid, tcat = sgj["target"]["id"], sgj["target"]["category"]

    sg = predict_scene_graph(vine, item["frames"], item["masks"], item["bboxes"],
                             item["gpt_spec"], device)

    # GT for the selected target (use frame 0 as a reference)
    fr0 = sgj["frames"][0]
    gt_attr = next((o.get("attributes") for o in fr0["objects"] if o["id"] == tid), None)
    gt_rel = [(tid, r["predicate"], r["object"]) for r in fr0["relations"] if r["subject"] == tid]
    tgt_rels = [(i, r, j) for (i, j), r in sg.relations.items() if i == tid]

    print(f"\n=== {scenario_id}  target='{tcat}' (id {tid}) ===")
    print(f"  VINE category : {sg.categories.get(tid)}  (conf {sg.category_conf.get(tid, 0):.2f})")
    print(f"  VINE attribute: {sg.attributes.get(tid)}     | GT attributes: {gt_attr}")
    print(f"  VINE relations(subj=target): {tgt_rels[:6]} | GT: {gt_rel[:6]}")


run_frame("scenario_00037")   # open fridge  -> expect attribute ~ 'open'
run_frame("scenario_00003")   # pot on X     -> expect a relation 'on'
print("\n[done] Sensible category/attribute/relation above => re-enabled heads carry the attack signal.")
