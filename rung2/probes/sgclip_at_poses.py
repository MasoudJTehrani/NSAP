"""Decisive check on the vision-blind poses.

gpt-4o reads these frames as "off" because the television is edge-on. The attack needs
SGClip to still ground the television AND its `displaying` attribute from the same frame:
that is what makes the symbol the sole source of the fact. If SGClip is blind too, the
benign condition collapses and there is nothing for the patch to delete.

Runs the DEPLOYMENT path (predict_gd -> predict_laser) on each selected frame.
"""
import os, sys, json
sys.path.insert(0, "rung2")
os.environ.setdefault("NSAP_AGGR_THRES", "0.4")
import numpy as np
from PIL import Image
from nav_harness import make_nav_planner

SHOTS = "output/probes/tvshots_sel"
VOCAB = ['television', 'sofa', 'chair', 'table', 'lamp', 'painting', 'window',
         'cabinet', 'shadow', 'wall', 'floor', 'other']

p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o")
print("planner up\n", flush=True)
rows = []
for f in sorted(os.listdir(SHOTS)):
    if not f.endswith("_selected.png"):
        continue
    scene = f.split("_")[0]
    img = np.array(Image.open(os.path.join(SHOTS, f)).convert("RGB"))
    labels, boxes = p.predict_gd(img, ['television'], box_threshold=0.1, text_threshold=0.1)
    nb = 0 if boxes is None else len(boxes)
    if nb == 0:
        print(f"{scene}: GroundingDINO found NO television box -> dead in every condition")
        rows.append((scene, 0, [], []))
        continue
    tl, *rest = p.predict_laser(image=img, boxes=boxes, target_names=['television'],
                                query_names=VOCAB, query_attributes=['displaying'],
                                query_relation_kws=[])
    topk = getattr(p, "_nsap_predicted_name", None)
    print(f"{scene}: GD boxes={nb}  grounded_target_labels={tl}")
    print(f"    SGClip top-k per box: {str(topk)[:300]}")
    print(f"    extra returns: {str(rest)[:220]}\n", flush=True)
    rows.append((scene, nb, tl, str(rest)[:200]))
print("SUMMARY")
for scene, nb, tl, _ in rows:
    ok = bool(tl)
    print(f"  {scene:14s} GD boxes={nb:>2}  television grounded: {'YES' if ok else 'NO'}")
