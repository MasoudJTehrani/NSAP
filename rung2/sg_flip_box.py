"""Box ATTRIBUTE SG-flip audit. Reuses the generated attribute_box frames (GT-open box +
bbox), so no THOR re-staging. For each GT-open frame: ground benign open/not-open, then paste
the trained patch on the box bbox and re-ground. The attack works when, on frames VINE
benignly reads OPEN, the patch flips it to CLOSED. Reports the benign open-read rate and the
of-those flip rate (the number that matters)."""
import os, sys, glob, json, argparse
sys.path.insert(0, "rung2")
os.environ["NSAP_LASER_ATTR_REL"] = "1"
import harness, numpy as np, cv2                                  # noqa: E402
from nav_harness import make_nav_planner                          # noqa: E402
import nsap_inject                                                # noqa: E402
from embodiedbench.planner.alfred.esca import bbox_to_mask        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="SceneGraphBench/data/attribute_box")
ap.add_argument("--patch", default="output/chexp_attr/uni_attribute_box_open/attribute_box_open/attribute_box_open_final.pt")
ap.add_argument("--attr", default="open")
args = ap.parse_args()

p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o-mini")
patch = nsap_inject.load_patch(args.patch)
kws = [args.attr, f"not {args.attr}"]

def read_open(img, box):
    h, w = img.shape[:2]; m = [bbox_to_mask(box, h, w)]
    _, u, _, _ = p.predicate_model(batched_video_ids=[0], batched_videos=[img], batched_masks=m,
        batched_bboxes=[box], batched_names=[["object"]], batched_object_ids=[[0, 0, 0]],
        batched_unary_kws=[kws], batched_binary_kws=[[]], batched_obj_pairs=[],
        batched_video_splits=[0], batched_binary_predicates=[None])
    d = {kw: v.item() for (f, o, kw), v in u[0].items()}
    return d[args.attr], d[f"not {args.attr}"]

n = bopen = flip = pboth = 0
for s in sorted(glob.glob(os.path.join(args.data, "scenarios", "scenario_*"))):
    sg = json.load(open(os.path.join(s, "scene_graph.json"))); tid = sg["target"]["id"]
    for fr in sg["frames"]:
        o = next((o for o in fr["objects"] if o["id"] == tid and "bbox" in o
                  and any(args.attr in a.lower() for a in o.get("attributes", []))), None)
        if not o:
            continue
        img = np.array(cv2.cvtColor(cv2.imread(os.path.join(s, fr["frame"])), cv2.COLOR_BGR2RGB))
        b_o, b_no = read_open(img, o["bbox"]); n += 1
        if b_o <= b_no:
            continue                      # benign already reads closed -> nothing to suppress
        bopen += 1
        patched, _ = nsap_inject.paste_on_bbox(img.copy(), o["bbox"], patch)
        p_o, p_no = read_open(patched, o["bbox"])
        if p_o <= p_no:
            flip += 1                     # open -> closed  (attack succeeded on this frame)

print(f"\n=== BOX '{args.attr}' SG-flip audit  ({os.path.basename(args.patch)}) ===")
print(f"frames (GT-open):            {n}")
print(f"benign reads OPEN:           {bopen}/{n} = {bopen/max(n,1):.0%}")
print(f"of-those flipped to CLOSED:  {flip}/{bopen} = {flip/max(bopen,1):.0%}   <-- attack strength")
