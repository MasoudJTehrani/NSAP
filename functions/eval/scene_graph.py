"""
Run VINE on a video clip (clean or patched) and return a structured scene graph.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SceneGraph:
    """Structured scene graph for one video clip."""
    data_id: str
    # {obj_id: top-1 category string}
    categories: Dict[int, str]          = field(default_factory=dict)
    # {obj_id: softmax confidence of the top-1 category}
    category_conf: Dict[int, float]     = field(default_factory=dict)
    # {obj_id: top-1 attribute string}
    attributes: Dict[int, str]          = field(default_factory=dict)
    # {(subj_id, obj_id): top-1 relation string}
    relations: Dict[Tuple[int,int], str] = field(default_factory=dict)


def select_eval_frame(masks, n_frames, require_obj=None):
    """The single frame predict_scene_graph scores: the one with the largest total
    mask area (objects clearest), tie-broken by lowest frame id. A pure function of
    masks, so the clean and patched passes always agree.

    If require_obj is given, restrict to frames whose masks contain that object id
    (used by visualize so the displayed frame actually holds the target). Returns
    None if there is no valid frame.
    """
    valid = [fid for fid in masks
             if fid < n_frames and (require_obj is None or require_obj in masks[fid])]
    if not valid:
        return None
    return max(valid, key=lambda fid: (sum(int(m.sum()) for m in masks[fid].values()), -fid))


def predict_scene_graph(
    vine_model,
    frames: List[torch.Tensor],
    masks: Dict[int, Dict[int, torch.Tensor]],
    bboxes: Dict[int, Dict[int, dict]],
    gpt_spec: dict,
    device: torch.device,
    return_top_k: int = 1,
    single_frame: bool = True,
    unary_kws_override=None,
    binary_kws_override=None,
) -> SceneGraph:
    """
    Run VINE's predict() method and return a SceneGraph.

    This uses VineModel.predict() (the standard no-grad inference path) — not the
    differentiable path. Only used for evaluation, not training.

    Args:
        vine_model:    loaded VineModel
        frames:        List of [H,W,3] float tensors (RGB)
        masks:         {frame_id: {obj_id: Tensor[H,W] bool}}
        bboxes:        {frame_id: {obj_id: {x1,y1,x2,y2}}}
        gpt_spec:      gpt_spec dict for this clip (provides keyword lists)
        device:        torch device
        return_top_k:  number of top predictions to aggregate (default 1)
        single_frame:  evaluate on ONE frame (default True), to match how ESCA
                       queries VINE — one frame at a time. See note below.

    Returns:
        SceneGraph with per-object top-1 predictions
    """
    cat_kws    = gpt_spec.get('consts',      [])
    # Eval parity: attribute/relation attacks ground the TARGET predicate through a
    # 2-way [k, "not k"] contrast (matching train.py + ESCA), passed as an override,
    # instead of the multi-way ATTRIBUTE_WORDS / RELATION_WORDS vocab.
    unary_kws  = unary_kws_override  if unary_kws_override  is not None else gpt_spec.get('unary_kws',  [])
    binary_kws = binary_kws_override if binary_kws_override is not None else gpt_spec.get('binary_kws', [])

    # ── Lock evaluation to a single frame (match ESCA's inference) ────────────
    # ESCA queries VINE one frame at a time (predict_laser wraps a single image
    # as a length-1 "video"), and VINE itself does no cross-frame reasoning.
    # Evaluating on the whole clip otherwise misrepresents deployment: the
    # category head silently keeps only the LAST frame's result (its key drops
    # frame_id and overwrites), while attribute/relation get majority-voted
    # across frames. Pick the single frame with the largest total mask area
    # (objects clearest), tie-broken by lowest frame_id. The choice depends only
    # on masks, so the clean and patched passes always select the SAME frame.
    if single_frame and len(masks) > 0:
        sel = select_eval_frame(masks, len(frames))
        if sel is not None:
            frames = [frames[sel]]
            masks  = {0: masks[sel]}
            bboxes = {0: bboxes.get(sel, {})}

    # Stack frames into a tensor [N, H, W, 3]
    video_tensor = torch.stack(frames).to(device)
    H_frame, W_frame = video_tensor.shape[1], video_tensor.shape[2]

    # Convert bboxes to list format expected by VineModel
    bboxes_list: Dict[int, Dict[int, List]] = {}
    for fid, obj_bboxes in bboxes.items():
        bboxes_list[fid] = {}
        for oid, bb in obj_bboxes.items():
            bboxes_list[fid][oid] = [bb['x1'], bb['y1'], bb['x2'], bb['y2']]

    # Convert masks to tensor format, resizing to match frame if needed
    masks_tensor: Dict[int, Dict[int, torch.Tensor]] = {}
    for fid, obj_masks in masks.items():
        masks_tensor[fid] = {}
        for oid, m in obj_masks.items():
            m = m.to(device)
            if m.shape[0] != H_frame or m.shape[1] != W_frame:
                m = F.interpolate(
                    m.float().unsqueeze(0).unsqueeze(0),
                    size=(H_frame, W_frame), mode='nearest',
                ).squeeze(0).squeeze(0).bool()
            masks_tensor[fid][oid] = m

    # VINE scores relations ONLY for the pairs passed in `object_pairs` — its
    # forward skips binary entirely when that list is empty. So a relation eval
    # MUST supply pairs, or sg.relations stays empty and every relation attack
    # reads 0 targets (while training, which scores each pair explicitly, finds
    # plenty). Pass all ordered object pairs; the forward filters to
    # overlapping-bbox pairs. Skip the work when there are no binary keywords.
    if binary_kws:
        oids = sorted({oid for fobj in masks_tensor.values() for oid in fobj})
        object_pairs = [(i, j) for i in oids for j in oids if i != j]
    else:
        object_pairs = []

    results = vine_model.predict(
        video_frames=video_tensor,
        masks=masks_tensor,
        bboxes=bboxes_list,
        categorical_keywords=cat_kws,
        unary_keywords=unary_kws,
        binary_keywords=binary_kws,
        object_pairs=object_pairs,
        return_top_k=return_top_k,
    )

    sg = SceneGraph(data_id="")

    # Aggregate predictions across frames: take majority-vote top-1.
    # Under single_frame=True there is exactly one frame, so the vote is trivial.
    cat_votes:  Dict[int, Dict[str, int]] = {}
    attr_votes: Dict[int, Dict[str, int]] = {}
    rel_votes:  Dict[Tuple[int,int], Dict[str, int]] = {}

    cat_conf: Dict[int, float] = {}
    for obj_id, pred_list in results.get('categorical_predictions', {}).items():
        if pred_list:
            prob, cat = pred_list[0][0], pred_list[0][1]   # (prob, label)
            cat_votes.setdefault(obj_id, {})
            cat_votes[obj_id][cat] = cat_votes[obj_id].get(cat, 0) + 1
            cat_conf[obj_id] = max(cat_conf.get(obj_id, 0.0), float(prob))

    for (frame_id, obj_id), pred_list in results.get('unary_predictions', {}).items():
        if pred_list:
            attr = pred_list[0][1]
            attr_votes.setdefault(obj_id, {})
            attr_votes[obj_id][attr] = attr_votes[obj_id].get(attr, 0) + 1

    for (frame_id, obj_pair), pred_list in results.get('binary_predictions', {}).items():
        if pred_list:
            rel = pred_list[0][1]
            rel_votes.setdefault(obj_pair, {})
            rel_votes[obj_pair][rel] = rel_votes[obj_pair].get(rel, 0) + 1

    sg.categories    = {oid: max(votes, key=votes.get) for oid, votes in cat_votes.items()}
    sg.category_conf = cat_conf
    sg.attributes    = {oid: max(votes, key=votes.get) for oid, votes in attr_votes.items()}
    sg.relations     = {pair: max(votes, key=votes.get) for pair, votes in rel_votes.items()}

    # Attribute-attack eval parity: VineModel.predict grounds the unary head on the
    # FULL frame, but train.py and ESCA (predict_laser) ground attributes on a RAW
    # bbox crop of the object -- that crop IS what ESCA feeds VINE to build the scene
    # graph. When a 2-way attribute contrast is requested (attribute attack), regrind
    # each object's attribute through that same crop forward, so the eval measures the
    # patch on the exact input it was trained and deployed against (box_flip confirms
    # ~97% through this forward, vs 0% on the full-frame one).
    if unary_kws_override is not None and len(video_tensor) > 0:
        from functions.attack.preprocess import crop_image_contain_bboxes_torch, clip_scores
        frame_hwc = video_tensor[0]
        crop_attr: Dict[int, str] = {}
        with torch.no_grad():
            for oid, bb in bboxes_list.get(0, {}).items():
                crop = crop_image_contain_bboxes_torch(frame_hwc, [bb])
                sc = clip_scores(vine_model, vine_model.clip_unary_model, crop,
                                 unary_kws_override, device, 224)
                crop_attr[oid] = unary_kws_override[int(sc.argmax().item())]
        sg.attributes = crop_attr

    # Relation-attack eval parity: VineModel.predict's binary forward undercounts the
    # attack (22% here vs 56% through the deployment forward). Re-ground relations the
    # way train.py + ESCA/predict_laser do -- extract subject+object, crop to their
    # union, score the binary head with the 2-way [rel, "not rel"] contrast -- so a
    # deleted/redirected (pot, on, X) edge is actually seen. Only nearby (overlapping-ish)
    # pairs are scored, matching predict_laser's pair filter and keeping it cheap.
    if binary_kws_override is not None and len(video_tensor) > 0:
        from functions.attack.preprocess import (extract_object_subject_torch,
                                                 crop_image_contain_bboxes_torch, clip_scores)
        def _near(a, b):
            gap = 0.15 * max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1)
            return not (a[0] - gap > b[2] or b[0] - gap > a[2] or a[1] - gap > b[3] or b[1] - gap > a[3])
        frame_hwc = video_tensor[0]
        m0 = masks_tensor.get(0, {}); bb0 = bboxes_list.get(0, {})
        oids = sorted(set(m0) & set(bb0))
        crop_rel: Dict[Tuple[int, int], str] = {}
        with torch.no_grad():
            for i in oids:
                for j in oids:
                    if i == j or not _near(bb0[i], bb0[j]):
                        continue
                    proc = extract_object_subject_torch(
                        frame_hwc, m0[i], m0[j],
                        alpha=vine_model.config.alpha, white_alpha=vine_model.config.white_alpha)
                    crop = crop_image_contain_bboxes_torch(proc, [bb0[i], bb0[j]])
                    sc = clip_scores(vine_model, vine_model.clip_binary_model, crop,
                                     binary_kws_override, device, 224)
                    crop_rel[(i, j)] = binary_kws_override[0 if int(sc.argmax().item()) == 0 else 1]
        sg.relations = crop_rel

    return sg
