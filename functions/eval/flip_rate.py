"""
Compute per-object prediction flip rate — the headline attack metric.

Flip rate = fraction of TARGET objects whose top-1 prediction changed after
patching.  Only the objects/relations that match the attack target are counted;
unrelated objects in the same clip are ignored.

  - category/attribute attack: target objects = those predicted as target_category
    in the clean scene graph
  - relation attack: target pairs = those whose clean relation matches target_relation
    (optionally filtered by subject category = target_category)
"""

import torch
from typing import List, Dict, Optional


def compute_flip_rate(
    vine_model,
    clean_frames: List[torch.Tensor],
    patched_frames: List[torch.Tensor],
    masks: Dict[int, Dict[int, torch.Tensor]],
    bboxes: Dict[int, Dict[int, dict]],
    gpt_spec: dict,
    device: torch.device,
    attack_mode: str,
    target_category: str = '',
    target_relation: str = '',
) -> dict:
    """
    Compute the prediction flip rate restricted to the target object(s).

    Args:
        vine_model:       loaded VineModel
        clean_frames:     List of original [H,W,3] float tensors
        patched_frames:   List of patched [H,W,3] float tensors (same length)
        masks:            {frame_id: {obj_id: Tensor[H,W]}}
        bboxes:           {frame_id: {obj_id: {x1,y1,x2,y2}}}
        gpt_spec:         clip keyword lists for this clip
        device:           torch device
        attack_mode:      'category' | 'attribute' | 'relation'
        target_category:  category string being attacked (e.g. 'camera')
        target_relation:  relation string being attacked (e.g. 'wearing')

    Returns:
        dict with keys:
            flip_rate           overall fraction of target objects/relations that flipped
            n_total             number of target objects/relations evaluated
            n_flipped           number that flipped
            per_object_flips    list[bool]
    """
    from functions.eval.scene_graph import predict_scene_graph

    sg_clean   = predict_scene_graph(vine_model, clean_frames,   masks, bboxes, gpt_spec, device)
    sg_patched = predict_scene_graph(vine_model, patched_frames, masks, bboxes, gpt_spec, device)

    def _match(label: str, target: str) -> bool:
        return target.lower() in label.lower()

    flips: List[bool] = []

    if attack_mode in ('category', 'attribute'):
        # Identify target objects by their clean-SG category prediction
        if target_category:
            target_obj_ids = {
                obj_id for obj_id, cat in sg_clean.categories.items()
                if _match(cat, target_category)
            }
        else:
            target_obj_ids = set(sg_clean.categories.keys())

        if attack_mode == 'category':
            clean_preds   = sg_clean.categories
            patched_preds = sg_patched.categories
        else:
            clean_preds   = sg_clean.attributes
            patched_preds = sg_patched.attributes

        for obj_id in target_obj_ids:
            c_pred = clean_preds.get(obj_id)
            if c_pred is None:
                continue
            p_pred = patched_preds.get(obj_id)
            # Count as flip if prediction changed OR target object disappeared
            flips.append(c_pred != p_pred if p_pred is not None else True)

    elif attack_mode == 'relation':
        if target_relation:
            target_pairs = {
                pair for pair, rel in sg_clean.relations.items()
                if _match(rel, target_relation)
            }
        else:
            target_pairs = set(sg_clean.relations.keys())

        # Optionally filter by subject category
        if target_category:
            target_pairs = {
                pair for pair in target_pairs
                if _match(sg_clean.categories.get(pair[0], ''), target_category)
            }

        for pair in target_pairs:
            c_pred = sg_clean.relations.get(pair)
            if c_pred is None:
                continue
            p_pred = sg_patched.relations.get(pair)
            flips.append(c_pred != p_pred if p_pred is not None else True)

    else:
        raise ValueError(f"Unknown attack_mode: {attack_mode}")

    n_total   = len(flips)
    n_flipped = sum(flips)
    flip_rate = n_flipped / n_total if n_total > 0 else 0.0

    return {
        'flip_rate':        flip_rate,
        'n_total':          n_total,
        'n_flipped':        n_flipped,
        'per_object_flips': flips,
    }


def compute_flip_rate_across_clips(
    vine_model,
    clip_items: list,
    patch,
    anchor,
    config,
    device: torch.device,
) -> dict:
    """
    Evaluate flip rate over a list of test clips.

    Returns:
        dict with aggregated flip_rate and per-clip results
    """
    from functions.attack.patch import paste_patch, compute_patch_size

    all_results = []

    for item in clip_items:
        if item is None:
            continue

        clean_frames = item['frames']
        masks        = item['masks']
        bboxes       = item['bboxes']
        gpt_spec     = item['gpt_spec']

        patched_frames = []
        for frame_id, frame in enumerate(clean_frames):
            frame_bboxes = bboxes.get(frame_id, {})
            patched      = frame.clone()
            for obj_id, bbox in frame_bboxes.items():
                ph, pw  = compute_patch_size(bbox, config.coverage_ratio)
                patched = paste_patch(
                    patched.to(device), patch.pixels_hwc,
                    anchor.cx, anchor.cy, ph, pw,
                )
            patched_frames.append(patched.cpu())

        result = compute_flip_rate(
            vine_model, clean_frames, patched_frames,
            masks, bboxes, gpt_spec, device,
            attack_mode      = config.attack_mode,
            target_category  = config.target_category,
            target_relation  = config.target_relation,
        )
        result['data_id'] = item['data_id']
        all_results.append(result)

    if not all_results:
        return {'flip_rate': 0.0, 'per_clip': []}

    total_flip  = sum(r['n_flipped'] for r in all_results)
    total_pairs = sum(r['n_total']   for r in all_results)
    return {
        'flip_rate': total_flip / total_pairs if total_pairs > 0 else 0.0,
        'n_total':   total_pairs,
        'n_flipped': total_flip,
        'per_clip':  all_results,
    }
