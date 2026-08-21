"""Planner-faithful box-open flip meter. Replicates esca.predict_laser's unary grounding
EXACTLY (bbox mask + actual bbox + query_names) so the number reflects the MIM benchmark.
For each GT-open box frame: ground open/not-open benign, paste the patch on the bbox, re-ground.
Reports, over frames VINE benignly reads OPEN, the fraction the patch flips to CLOSED."""
import os, sys, glob, json, argparse
sys.path.insert(0, "rung2"); os.environ["NSAP_LASER_ATTR_REL"] = "1"
import harness, cv2, numpy as np                                  # noqa: E402
from nav_harness import make_nav_planner                          # noqa: E402
from embodiedbench.planner.alfred.esca import bbox_to_mask        # noqa: E402
import nsap_inject                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="SceneGraphBench/data/attribute_box")
ap.add_argument("--patch", required=True)
ap.add_argument("--attr", default="open")
ap.add_argument("--qnames", default="box,shelf,cabinet,counter top,sofa")
args = ap.parse_args()

p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o-mini")
patch = nsap_inject.load_patch(args.patch)
QN = args.qnames.split(","); kws = [args.attr, f"not {args.attr}"]

def open_gt(img, box):
    """EXACT planner unary call (esca.py:627-640): actual bbox + bbox-mask + query_names."""
    h, w = img.shape[:2]; m = [bbox_to_mask(box, h, w)]
    _, u, _, _ = p.predicate_model(batched_video_ids=[0], batched_videos=[img], batched_masks=m,
        batched_bboxes=[box], batched_names=[QN], batched_object_ids=[[0, 0, 0]],
        batched_unary_kws=[kws], batched_binary_kws=[[]], batched_obj_pairs=[],
        batched_video_splits=[0], batched_binary_predicates=[None])
    d = {kw: v.item() for (f, o, kw), v in u[0].items()}
    return d[args.attr] > d[f"not {args.attr}"]

n = bopen = flip = 0
for s in sorted(glob.glob(os.path.join(args.data, "scenarios", "scenario_*"))):
    sg = json.load(open(os.path.join(s, "scene_graph.json"))); tid = sg["target"]["id"]
    for fr in sg["frames"]:
        o = next((o for o in fr["objects"] if o["id"] == tid and "bbox" in o
                  and any(args.attr in a.lower() for a in o.get("attributes", []))), None)
        if not o:
            continue
        img = np.array(cv2.cvtColor(cv2.imread(os.path.join(s, fr["frame"])), cv2.COLOR_BGR2RGB))
        n += 1
        if not open_gt(img, o["bbox"]):
            continue
        bopen += 1
        patched, _ = nsap_inject.paste_on_bbox(img.copy(), o["bbox"], patch)
        if not open_gt(patched, o["bbox"]):
            flip += 1

print(f"\n=== BOX '{args.attr}' PLANNER-forward flip  ({os.path.basename(os.path.dirname(os.path.dirname(args.patch)))}) ===")
print(f"frames (GT-open):     {n}")
print(f"benign reads OPEN:    {bopen}/{n} = {bopen/max(n,1):.0%}")
print(f"flipped to CLOSED:    {flip}/{bopen} = {flip/max(bopen,1):.0%}   <-- MIM-relevant attack strength")
