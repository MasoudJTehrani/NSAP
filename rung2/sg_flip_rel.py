"""Relation ('on') SG-flip audit. Reuses the generated relation_pot frames (GT pot-on-counter
+ bbox). For each GT-on frame: ground benign relations for the pot, then paste the patch on the
pot bbox and re-ground. The attack works when the patch DELETES the (pot, on, counter) relation.
Reports, over frames where VINE benignly grounds the relation, the of-those delete rate."""
import os, sys, glob, json, argparse
sys.path.insert(0, "rung2")
os.environ["NSAP_LASER_ATTR_REL"] = "1"
import harness, numpy as np, cv2                                  # noqa: E402
from nav_harness import make_nav_planner                          # noqa: E402
import nsap_inject                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="SceneGraphBench/data/relation_pot")
ap.add_argument("--patch", default="output/chexp_rel/uni_relation_on_pot/relation_on_pot/relation_on_pot_final.pt")
ap.add_argument("--pred", default="on")
ap.add_argument("--qobjs", default="pot,stove burner,cabinet,counter top,sink")
args = ap.parse_args()

p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o-mini")
patch = nsap_inject.load_patch(args.patch)
QO = args.qobjs.split(",")

def rels_for(img):
    labels, boxes = p.predict_gd(img, QO, box_threshold=0.15, text_threshold=0.1)
    _, _, rels, *_ = p.predict_laser(image=img, boxes=boxes, target_names=["pot"], query_names=QO,
                                 query_attributes=[], query_relation_kws=[args.pred])
    return [r for r in rels if r[1] == args.pred and "pot" in r[0]]

n = bhas = flip = 0
for s in sorted(glob.glob(os.path.join(args.data, "scenarios", "scenario_*"))):
    sg = json.load(open(os.path.join(s, "scene_graph.json"))); tid = sg["target"]["id"]
    for fr in sg["frames"]:
        o = next((o for o in fr["objects"] if o["id"] == tid and "bbox" in o), None)
        if not o:
            continue
        img = np.array(cv2.cvtColor(cv2.imread(os.path.join(s, fr["frame"])), cv2.COLOR_BGR2RGB))
        n += 1
        if not rels_for(img):
            continue                       # benign didn't ground the relation -> nothing to delete
        bhas += 1
        patched, _ = nsap_inject.paste_on_bbox(img.copy(), o["bbox"], patch)
        if not rels_for(patched):
            flip += 1                       # relation deleted (attack succeeded)

print(f"\n=== RELATION '{args.pred}' SG-flip audit  ({os.path.basename(args.patch)}) ===")
print(f"frames (GT-on):              {n}")
print(f"benign grounds relation:     {bhas}/{n} = {bhas/max(n,1):.0%}")
print(f"of-those DELETED by patch:   {flip}/{bhas} = {flip/max(bhas,1):.0%}   <-- attack strength")
