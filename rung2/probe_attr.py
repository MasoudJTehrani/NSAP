"""Probe: does VINE's unary head DISCRIMINATE 'open' vs 'closed' per object category?
High separation (AUC) => that object's 'open' attribute is attackable."""
import os, sys, json, glob
sys.path.insert(0, "rung2")
import harness, cv2, numpy as np
from nav_harness import make_nav_planner
from embodiedbench.planner.alfred.esca import bbox_to_mask
import collections

D = "SceneGraphBench/data/openables_probe"
p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o-mini")

def open_score(img, bbox):
    h, w = img.shape[:2]
    m = [bbox_to_mask(bbox, h, w)]
    _, unary, _, _ = p.predicate_model(batched_video_ids=[0], batched_videos=[img], batched_masks=m,
        batched_bboxes=[bbox], batched_names=[["object"]], batched_object_ids=[[0, 0, 0]],
        batched_unary_kws=[["open", "not open"]], batched_binary_kws=[[]], batched_obj_pairs=[],
        batched_video_splits=[0], batched_binary_predicates=[None])
    d = {kw: v.item() for (fid, oid, kw), v in unary[0].items()}
    return d.get("open")

data = collections.defaultdict(lambda: {"open": [], "closed": []})
for s in sorted(glob.glob(f"{D}/scenarios/scenario_*")):
    sg = json.load(open(os.path.join(s, "scene_graph.json"))); tid = sg["target"]["id"]; cat = sg["target"]["category"]
    for fr in sg["frames"]:
        o = next((o for o in fr["objects"] if o["id"] == tid and "bbox" in o), None)
        if not o: continue
        a = [x.lower() for x in o.get("attributes", [])]
        st = "open" if "open" in a else ("closed" if ("closed" in a) else None)
        if st is None: continue
        img = cv2.cvtColor(cv2.imread(os.path.join(s, fr["frame"])), cv2.COLOR_BGR2RGB)
        sc = open_score(img, o["bbox"])
        if sc is not None: data[cat][st].append(sc)

def auc(pos, neg):
    if not pos or not neg: return None
    return sum((x > y) + 0.5*(x == y) for x in pos for y in neg) / (len(pos)*len(neg))

print("\n=== VINE 'open' discrimination per category (open-score) ===")
print(f"  {'category':12}{'n_open':>7}{'n_closed':>9}{'mean_open':>11}{'mean_closed':>13}{'AUC':>7}  attackable?")
for cat in sorted(data):
    o, c = data[cat]["open"], data[cat]["closed"]
    mo = np.mean(o) if o else 0; mc = np.mean(c) if c else 0; a = auc(o, c)
    verdict = "YES" if (a and a >= 0.70) else ("weak" if (a and a >= 0.6) else "no")
    print(f"  {cat:12}{len(o):>7}{len(c):>9}{mo:>11.2f}{mc:>13.2f}{(a if a else 0):>7.2f}  {verdict}")
