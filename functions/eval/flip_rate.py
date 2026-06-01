"""
Compute per-object prediction flip rate — the headline attack metric.

Flip rate = fraction of TARGET objects whose top-1 prediction changed after
patching.  Only the objects/relations that match the attack target are counted;
unrelated objects in the same clip are ignored.

For each target object the patch is applied only to THAT object across every
frame where it appears.  The full multi-frame scene graph is then compared
against the clean scene graph for that specific object.  This mirrors the
real-world scenario where a single person wears the physical patch.

  - category/attribute attack: target objects = those predicted as target_category
    (and target_attribute for attribute mode) in the clean scene graph
  - relation attack: target pairs = those whose clean relation matches target_relation
    (optionally filtered by subject category = target_category)
"""

import torch
from typing import List, Dict, Optional


# ── shared helpers ────────────────────────────────────────────────────────────

def _patch_object_across_frames(
    clean_frames: list,
    bboxes: dict,
    patch,
    anchor,
    config,
    device: torch.device,
    obj_id: int,
) -> list:
    """
    Return a copy of clean_frames with the patch applied only to obj_id
    in every frame where that object appears.  Frames without obj_id are
    returned unchanged.
    """
    from functions.attack.patch import paste_patch, compute_patch_size
    from functions.attack.anchor import anchor_center

    patched = []
    for frame_id, frame in enumerate(clean_frames):
        frame_bboxes = bboxes.get(frame_id, {})
        if obj_id in frame_bboxes:
            bbox   = frame_bboxes[obj_id]
            ph, pw = compute_patch_size(bbox, config.coverage_ratio)
            cx, cy = anchor_center(anchor, bbox)
            f = paste_patch(
                frame.clone().to(device), patch.pixels_hwc, cx, cy, ph, pw,
            ).cpu()
        else:
            f = frame
        patched.append(f)
    return patched


# ── per-clip helper (kept for external use) ───────────────────────────────────

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
    target_attribute: str = '',
) -> dict:
    """
    Compute the prediction flip rate given pre-computed clean and patched frames.

    NOTE: the caller is responsible for patching the correct objects only.
    For batch evaluation use compute_flip_rate_across_clips() instead.

    Returns:
        dict with keys: flip_rate, n_total, n_flipped, per_object_flips
    """
    from functions.eval.scene_graph import predict_scene_graph

    sg_clean   = predict_scene_graph(vine_model, clean_frames,   masks, bboxes, gpt_spec, device)
    sg_patched = predict_scene_graph(vine_model, patched_frames, masks, bboxes, gpt_spec, device)

    def _match(label: str, target: str) -> bool:
        return target.lower() in label.lower()

    flips: List[bool] = []

    if attack_mode in ('category', 'attribute'):
        if target_category:
            target_obj_ids = {
                obj_id for obj_id, cat in sg_clean.categories.items()
                if _match(cat, target_category)
            }
        else:
            target_obj_ids = set(sg_clean.categories.keys())

        if attack_mode == 'attribute' and target_attribute:
            target_obj_ids = {
                obj_id for obj_id in target_obj_ids
                if _match(sg_clean.attributes.get(obj_id, ''), target_attribute)
            }

        clean_preds   = sg_clean.categories   if attack_mode == 'category' else sg_clean.attributes
        patched_preds = sg_patched.categories if attack_mode == 'category' else sg_patched.attributes

        for obj_id in target_obj_ids:
            c_pred = clean_preds.get(obj_id)
            if c_pred is None:
                continue
            p_pred = patched_preds.get(obj_id)
            flips.append(c_pred != p_pred if p_pred is not None else True)

    elif attack_mode == 'relation':
        if target_relation:
            target_pairs = {
                pair for pair, rel in sg_clean.relations.items()
                if _match(rel, target_relation)
            }
        else:
            target_pairs = set(sg_clean.relations.keys())

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
    return {
        'flip_rate':        n_flipped / n_total if n_total > 0 else 0.0,
        'n_total':          n_total,
        'n_flipped':        n_flipped,
        'per_object_flips': flips,
    }


# ── main evaluation entry point ───────────────────────────────────────────────

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

    For each target object the patch is applied only to THAT object across
    every frame where it appears (other objects are left clean).  The full
    multi-frame scene graph is then compared against the clean scene graph.

    Returns:
        dict with aggregated flip_rate, n_total, n_flipped, and per_clip results
    """
    from functions.eval.scene_graph import predict_scene_graph

    all_results = []

    for item in clip_items:
        if item is None:
            continue

        clean_frames = item['frames']
        masks        = item['masks']
        bboxes       = item['bboxes']
        gpt_spec     = item['gpt_spec']

        sg_clean = predict_scene_graph(vine_model, clean_frames, masks, bboxes, gpt_spec, device)
        flips: List[bool] = []

        if config.attack_mode in ('category', 'attribute'):
            target_obj_ids = {
                oid for oid, cat in sg_clean.categories.items()
                if config.target_category.lower() in cat.lower()
            } if config.target_category else set(sg_clean.categories.keys())

            if config.attack_mode == 'attribute' and getattr(config, 'target_attribute', ''):
                target_obj_ids = {
                    oid for oid in target_obj_ids
                    if config.target_attribute.lower() in (sg_clean.attributes.get(oid) or '').lower()
                }

            clean_preds = (sg_clean.categories if config.attack_mode == 'category'
                           else sg_clean.attributes)

            for obj_id in target_obj_ids:
                c_pred = clean_preds.get(obj_id)
                if c_pred is None:
                    continue

                patched_frames = _patch_object_across_frames(
                    clean_frames, bboxes, patch, anchor, config, device, obj_id,
                )
                sg_patched = predict_scene_graph(
                    vine_model, patched_frames, masks, bboxes, gpt_spec, device,
                )

                patched_preds = (sg_patched.categories if config.attack_mode == 'category'
                                 else sg_patched.attributes)
                p_pred = patched_preds.get(obj_id)
                flips.append(c_pred != p_pred if p_pred is not None else True)

        elif config.attack_mode == 'relation':
            target_pairs = {
                pair for pair, rel in sg_clean.relations.items()
                if config.target_relation.lower() in rel.lower()
            } if config.target_relation else set(sg_clean.relations.keys())

            if config.target_category:
                target_pairs = {
                    p for p in target_pairs
                    if config.target_category.lower() in sg_clean.categories.get(p[0], '').lower()
                }

            for pair in target_pairs:
                subj_id = pair[0]
                c_pred  = sg_clean.relations.get(pair)
                if c_pred is None:
                    continue

                patched_frames = _patch_object_across_frames(
                    clean_frames, bboxes, patch, anchor, config, device, subj_id,
                )
                sg_patched = predict_scene_graph(
                    vine_model, patched_frames, masks, bboxes, gpt_spec, device,
                )
                p_pred = sg_patched.relations.get(pair)
                flips.append(c_pred != p_pred if p_pred is not None else True)

        result = {
            'data_id':          item['data_id'],
            'n_flipped':        sum(flips),
            'n_total':          len(flips),
            'per_object_flips': flips,
            'flip_rate':        sum(flips) / len(flips) if flips else 0.0,
        }
        all_results.append(result)

    if not all_results:
        return {'flip_rate': 0.0, 'n_total': 0, 'n_flipped': 0, 'per_clip': []}

    total_flip  = sum(r['n_flipped'] for r in all_results)
    total_objs  = sum(r['n_total']   for r in all_results)
    clips_with  = sum(1 for r in all_results if r['n_total'] > 0)
    return {
        'flip_rate':         total_flip / total_objs if total_objs > 0 else 0.0,
        'n_total':           total_objs,
        'n_flipped':         total_flip,
        'n_clips_with_targets': clips_with,
        'n_clips_total':     len(all_results),
        'per_clip':          all_results,
    }
