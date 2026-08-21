"""Planner-faithful model-level eval for the CURRENT attribute (TV) and relation (pot) patches.
The run_attack --eval forward is stale for these heads (multi-way vocab, no 2-way contrast), so
we measure through the SAME forward the agent deploys: the attacked head via its 2-way contrast,
and the CATEGORY head (the main collateral) via its argmax. Reports, over frames the clean model
grounds correctly: attacked-head flip, category-collateral rate, and isolation = attacked/(attacked+collateral)."""
import os, sys, glob, json
sys.path.insert(0, "rung2"); os.environ["NSAP_LASER_ATTR_REL"] = "1"
import harness, cv2, numpy as np                                  # noqa: E402
from nav_harness import make_nav_planner                          # noqa: E402
from embodiedbench.planner.alfred.esca import bbox_to_mask        # noqa: E402
import nsap_inject                                                # noqa: E402

p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o-mini")

def cate_argmax(img, box, names):
    h, w = img.shape[:2]; m = [bbox_to_mask(box, h, w)]
    c, _, _, _ = p.predicate_model(batched_video_ids=[0], batched_videos=[img], batched_masks=m,
        batched_bboxes=[box], batched_names=[names], batched_object_ids=[[0, 0, 0]],
        batched_unary_kws=[[]], batched_binary_kws=[[]], batched_obj_pairs=[], batched_video_splits=[0],
        batched_binary_predicates=[None])
    d = {name: v.item() for (oid, name), v in c[0].items()}
    return max(d, key=d.get)

def unary_pos(img, box, attr):
    h, w = img.shape[:2]; m = [bbox_to_mask(box, h, w)]
    _, u, _, _ = p.predicate_model(batched_video_ids=[0], batched_videos=[img], batched_masks=m,
        batched_bboxes=[box], batched_names=[["television", "cabinet", "sofa", "object"]],
        batched_object_ids=[[0, 0, 0]], batched_unary_kws=[[attr, f"not {attr}"]], batched_binary_kws=[[]],
        batched_obj_pairs=[], batched_video_splits=[0], batched_binary_predicates=[None])
    d = {k: v.item() for (f, o, k), v in u[0].items()}
    return d[attr] > d[f"not {attr}"]

def rels_on(img, qo):
    labels, boxes = p.predict_gd(img, qo, box_threshold=0.15, text_threshold=0.1)
    _, _, r, *_ = p.predict_laser(image=img, boxes=boxes, target_names=["pot"], query_names=qo,
                              query_attributes=[], query_relation_kws=["on"])
    return [x for x in r if x[1] == "on" and "pot" in x[0]]

def report(tag, n, atk_base, atk_flip, cat_base, cat_flip):
    iso = atk_flip / max(atk_flip + cat_flip, 1)
    print(f"\n=== {tag} ===")
    print(f"  frames (clean grounds target): {atk_base}/{n}")
    print(f"  ATTACKED head flip:   {atk_flip}/{atk_base} = {atk_flip/max(atk_base,1):.0%}")
    print(f"  COLLATERAL category:  {cat_flip}/{cat_base} = {cat_flip/max(cat_base,1):.0%}   (lower = better isolation)")
    print(f"  ISOLATION = attacked/(attacked+collateral) = {atk_flip}/{atk_flip+cat_flip} = {iso:.3f}")

# ---------- ATTRIBUTE: television 'displaying' ----------
patch = nsap_inject.load_patch("output/chexp_attr/uni_attribute_tv_displaying/attribute_television_displaying/attribute_television_displaying_final.pt")
n = ab = af = cb = cf = 0
for s in sorted(glob.glob("SceneGraphBench/data/attribute_tv/scenarios/scenario_*")):
    sg = json.load(open(os.path.join(s, "scene_graph.json"))); tid = sg["target"]["id"]
    for fr in sg["frames"]:
        o = next((o for o in fr["objects"] if o["id"] == tid and "bbox" in o), None)
        if not o:
            continue
        img = np.array(cv2.cvtColor(cv2.imread(os.path.join(s, fr["frame"])), cv2.COLOR_BGR2RGB))
        n += 1
        b_d = unary_pos(img, o["bbox"], "displaying"); b_c = cate_argmax(img, o["bbox"], ["television", "cabinet", "sofa", "shelf", "object"])
        patched, _ = nsap_inject.paste_on_bbox(img.copy(), o["bbox"], patch)
        p_d = unary_pos(patched, o["bbox"], "displaying"); p_c = cate_argmax(patched, o["bbox"], ["television", "cabinet", "sofa", "shelf", "object"])
        if b_d: ab += 1;  af += int(not p_d)
        if b_c == "television": cb += 1; cf += int(p_c != "television")
report("ATTRIBUTE  television 'displaying'  (chexp_attr/uni_attribute_tv_displaying)", n, ab, af, cb, cf)

# ---------- RELATION: pot 'on' counter ----------
patch = nsap_inject.load_patch("output/chexp_rel/uni_relation_on_pot_fix/relation_on_pot/relation_on_pot_final.pt")
QO = ["pot", "stove burner", "cabinet", "counter top", "sink"]
n = ab = af = cb = cf = 0
for s in sorted(glob.glob("SceneGraphBench/data/relation_pot/scenarios/scenario_*")):
    sg = json.load(open(os.path.join(s, "scene_graph.json"))); tid = sg["target"]["id"]
    for fr in sg["frames"][:6]:            # cap for speed (relation grounding is heavy)
        o = next((o for o in fr["objects"] if o["id"] == tid and "bbox" in o), None)
        if not o:
            continue
        img = np.array(cv2.cvtColor(cv2.imread(os.path.join(s, fr["frame"])), cv2.COLOR_BGR2RGB))
        n += 1
        b_r = len(rels_on(img, QO)) > 0; b_c = cate_argmax(img, o["bbox"], ["pot", "stove burner", "counter top", "sink", "object"])
        patched, _ = nsap_inject.paste_on_bbox(img.copy(), o["bbox"], patch)
        p_r = len(rels_on(patched, QO)) > 0; p_c = cate_argmax(patched, o["bbox"], ["pot", "stove burner", "counter top", "sink", "object"])
        if b_r: ab += 1;  af += int(not p_r)
        if b_c == "pot": cb += 1; cf += int(p_c != "pot")
report("RELATION  (pot, on, counter)  (chexp_rel/uni_relation_on_pot_fix)", n, ab, af, cb, cf)
