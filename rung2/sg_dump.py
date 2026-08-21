"""Model-level per-instance visualization for the relation & attribute attacks. For a few
instances it grounds the target object's scene-graph facts (category + the attacked
attribute/relation) CLEAN vs PATCHED through the deployed forward, prints the before/after,
and saves the clean and patched frames so you can eyeball what the patch does."""
import os, sys, glob, json
sys.path.insert(0, "rung2"); os.environ["NSAP_LASER_ATTR_REL"] = "1"
import harness, cv2, numpy as np                                  # noqa: E402
from nav_harness import make_nav_planner                          # noqa: E402
from embodiedbench.planner.alfred.esca import bbox_to_mask        # noqa: E402
import nsap_inject                                                # noqa: E402

p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o-mini")
OUT = "output/sg_dump"; os.makedirs(OUT, exist_ok=True)

def cate(img, box, names):
    h, w = img.shape[:2]
    c, _, _, _ = p.predicate_model(batched_video_ids=[0], batched_videos=[img], batched_masks=[bbox_to_mask(box, h, w)],
        batched_bboxes=[box], batched_names=[names], batched_object_ids=[[0, 0, 0]], batched_unary_kws=[[]],
        batched_binary_kws=[[]], batched_obj_pairs=[], batched_video_splits=[0], batched_binary_predicates=[None])
    d = {n: v.item() for (o, n), v in c[0].items()}; k = max(d, key=d.get); return k, d[k]

def unary(img, box, attr):
    h, w = img.shape[:2]
    _, u, _, _ = p.predicate_model(batched_video_ids=[0], batched_videos=[img], batched_masks=[bbox_to_mask(box, h, w)],
        batched_bboxes=[box], batched_names=[["television", "cabinet", "sofa", "object"]], batched_object_ids=[[0, 0, 0]],
        batched_unary_kws=[[attr, f"not {attr}"]], batched_binary_kws=[[]], batched_obj_pairs=[], batched_video_splits=[0],
        batched_binary_predicates=[None])
    d = {k: v.item() for (f, o, k), v in u[0].items()}; return d[attr], d[f"not {attr}"]

def rels(img, qo):
    labels, boxes = p.predict_gd(img, qo, box_threshold=0.15, text_threshold=0.1)
    _, _, r, *_ = p.predict_laser(image=img, boxes=boxes, target_names=["pot"], query_names=qo, query_attributes=[], query_relation_kws=["on"])
    return [tuple(x) for x in r if x[1] == "on" and "pot" in x[0]]

def save(img_rgb, path):
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

print("\n" + "#" * 70 + "\n#  RELATION attack  (pot, on, counter)\n" + "#" * 70)
patch = nsap_inject.load_patch("output/chexp_rel/uni_relation_on_pot_fix/relation_on_pot/relation_on_pot_final.pt")
QO = ["pot", "stove burner", "cabinet", "counter top", "sink"]
shown = 0
for s in sorted(glob.glob("SceneGraphBench/data/relation_pot/scenarios/scenario_*")):
    sg = json.load(open(os.path.join(s, "scene_graph.json"))); tid = sg["target"]["id"]; scene = sg["scene"]
    for fr in sg["frames"]:
        o = next((o for o in fr["objects"] if o["id"] == tid and "bbox" in o), None)
        if not o:
            continue
        img = np.array(cv2.cvtColor(cv2.imread(os.path.join(s, fr["frame"])), cv2.COLOR_BGR2RGB))
        cr = rels(img, QO)
        if not cr:
            continue                       # only show instances the clean model grounds the relation
        cc, ccp = cate(img, o["bbox"], QO + ["object"])
        patched, _ = nsap_inject.paste_on_bbox(img.copy(), o["bbox"], patch)
        pr = rels(patched, QO); pc, pcp = cate(patched, o["bbox"], QO + ["object"])
        tag = f"rel_{os.path.basename(s)}"
        save(img, f"{OUT}/{tag}_clean.png"); save(patched, f"{OUT}/{tag}_patched.png")
        print(f"\n{scene}  ({tag})   frames: {OUT}/{tag}_[clean|patched].png")
        print(f"  CLEAN   scene graph:  category=({cc} {ccp:.2f})   relations={cr}")
        print(f"  PATCHED scene graph:  category=({pc} {pcp:.2f})   relations={pr}"
              f"   {'<-- relation DELETED' if cr and not pr else ''}"
              f"{'  <-- category CHANGED' if pc != cc else ''}")
        shown += 1; break
    if shown >= 4:
        break

print("\n" + "#" * 70 + "\n#  ATTRIBUTE attack  television 'displaying'\n" + "#" * 70)
patch = nsap_inject.load_patch("output/chexp_attr/uni_attribute_tv_displaying/attribute_television_displaying/attribute_television_displaying_final.pt")
TN = ["television", "cabinet", "sofa", "shelf", "object"]
shown = 0
for s in sorted(glob.glob("SceneGraphBench/data/attribute_tv/scenarios/scenario_*")):
    sg = json.load(open(os.path.join(s, "scene_graph.json"))); tid = sg["target"]["id"]; scene = sg["scene"]
    fr = sg["frames"][len(sg["frames"]) // 2]
    o = next((o for o in fr["objects"] if o["id"] == tid and "bbox" in o), None)
    if not o:
        continue
    img = np.array(cv2.cvtColor(cv2.imread(os.path.join(s, fr["frame"])), cv2.COLOR_BGR2RGB))
    cd, cnd = unary(img, o["bbox"], "displaying"); cc, ccp = cate(img, o["bbox"], TN)
    patched, _ = nsap_inject.paste_on_bbox(img.copy(), o["bbox"], patch)
    pd, pnd = unary(patched, o["bbox"], "displaying"); pc, pcp = cate(patched, o["bbox"], TN)
    tag = f"attr_{os.path.basename(s)}"
    save(img, f"{OUT}/{tag}_clean.png"); save(patched, f"{OUT}/{tag}_patched.png")
    print(f"\n{scene}  ({tag})   frames: {OUT}/{tag}_[clean|patched].png")
    print(f"  CLEAN   scene graph:  category=({cc} {ccp:.2f})   displaying: {cd:.2f} vs not {cnd:.2f} -> {'ON' if cd>cnd else 'off'}")
    print(f"  PATCHED scene graph:  category=({pc} {pcp:.2f})   displaying: {pd:.2f} vs not {pnd:.2f} -> {'ON' if pd>pnd else 'off'}"
          f"   {'<-- attribute FLIPPED on->off' if (cd>cnd and pd<=pnd) else ''}"
          f"{'  <-- category CHANGED' if pc != cc else ''}")
    shown += 1
    if shown >= 4:
        break
print(f"\nframes saved under {OUT}/")
