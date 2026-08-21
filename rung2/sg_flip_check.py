"""SG-flip sanity check: does the patch flip VINE's grounded relation/attribute?
Stages a 2-object scene, faces the target, grounds benign vs patched frame."""
import os, sys, json, math
sys.path.insert(0, "rung2")
os.environ["NSAP_LASER_ATTR_REL"] = "1"
import harness, numpy as np, cv2
from nav_harness import make_nav_planner
import nsap_inject
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

p = make_nav_planner(scene_graph_text=True, language_only=False, model_name="gpt-4o-mini")
c = Controller(platform=CloudRendering, width=600, height=600, renderInstanceSegmentation=True)
def recep(o): return [r.split("|")[0] for r in (o.get("parentReceptacles") or [])]

def face(tp):  # teleport close, facing the target
    reach = c.step("GetReachablePositions").metadata["actionReturn"]
    d = lambda q: math.hypot(q["x"]-tp["x"], q["z"]-tp["z"])
    band = [q for q in reach if 0.5 <= d(q) <= 1.0] or sorted(reach, key=d)[:5]
    pos = min(band, key=d)
    yaw = math.degrees(math.atan2(tp["x"]-pos["x"], tp["z"]-pos["z"])) % 360
    c.step(action="TeleportFull", position=pos, rotation=dict(x=0,y=yaw,z=0), horizon=15, standing=True)

def ground_rel(img):
    labels, boxes = p.predict_gd(img, ["pot","stove burner","cabinet","counter top","sink"], box_threshold=0.15, text_threshold=0.1)
    _,_,rels = p.predict_laser(image=img, boxes=boxes, target_names=["pot"],
        query_names=["pot","stove burner","cabinet","counter top","sink"], query_attributes=[], query_relation_kws=["on"])
    return rels

def open_score(img, box):
    from embodiedbench.planner.alfred.esca import bbox_to_mask
    h,w=img.shape[:2]; m=[bbox_to_mask(box,h,w)]
    _,u,_,_ = p.predicate_model(batched_video_ids=[0],batched_videos=[img],batched_masks=m,batched_bboxes=[box],
        batched_names=[["object"]],batched_object_ids=[[0,0,0]],batched_unary_kws=[["open","not open"]],
        batched_binary_kws=[[]],batched_obj_pairs=[],batched_video_splits=[0],batched_binary_predicates=[None])
    d={kw:v.item() for (f,o,kw),v in u[0].items()}; return d.get("open"), d.get("not open")

# ---------- RELATION: 2-pot, patch the pot on the counter ----------
print("\n==================== RELATION (pot on counter) ====================")
patch = nsap_inject.load_patch("output/chexp_rel/uni_relation_on_pot/relation_on_pot/relation_on_pot_final.pt")
c.reset(scene="FloorPlan1")
c.step(action="InitialRandomSpawn", randomSeed=3, forceVisible=True, numPlacementAttempts=12,
       numDuplicatesOfType=[{"objectType":"Pot","count":2}])
pots=[o for o in c.last_event.metadata["objects"] if o["objectType"]=="Pot"]
tgt=next(o for o in pots if "CounterTop" in recep(o))
face(tgt["position"])
dets=c.last_event.instance_detections2D
box=dets.get(tgt["objectId"])
img=cv2.cvtColor(c.last_event.frame, cv2.COLOR_RGB2RGB) if False else np.array(c.last_event.frame)
print("benign  relations:", ground_rel(img))
if box is not None:
    patched,_=nsap_inject.paste_on_bbox(np.array(c.last_event.frame), box, patch)
    print("patched relations:", ground_rel(patched))
else:
    print("target not visible for injection")

# ---------- ATTRIBUTE: 2-laptop, patch the open one ----------
print("\n==================== ATTRIBUTE (open laptop) ====================")
patchA = nsap_inject.load_patch("output/chexp_attr/uni_attribute_laptop_open/attribute_laptop_open/attribute_laptop_open_final.pt")
c.reset(scene="FloorPlan201")
c.step(action="InitialRandomSpawn", randomSeed=3, forceVisible=True, numPlacementAttempts=12,
       numDuplicatesOfType=[{"objectType":"Laptop","count":2}])
laps=[o for o in c.last_event.metadata["objects"] if o["objectType"]=="Laptop"]
c.step(action="OpenObject", objectId=laps[0]["objectId"], forceAction=True)
c.step(action="CloseObject", objectId=laps[1]["objectId"], forceAction=True)
tgt=next(o for o in c.last_event.metadata["objects"] if o["objectId"]==laps[0]["objectId"])
face(tgt["position"])
box=c.last_event.instance_detections2D.get(tgt["objectId"])
if box is not None:
    o,no=open_score(np.array(c.last_event.frame), box)
    print(f"benign  open-score: open={o:.2f} not-open={no:.2f}  -> VINE reads {'OPEN' if o>no else 'closed'}")
    patched,_=nsap_inject.paste_on_bbox(np.array(c.last_event.frame), box, patchA)
    o2,no2=open_score(patched, box)
    print(f"patched open-score: open={o2:.2f} not-open={no2:.2f}  -> VINE reads {'OPEN' if o2>no2 else 'closed'}")
else:
    print("target laptop not visible")
c.stop()
