"""
Compute per-frame prediction flip rate — the headline attack metric.

Flip rate = fraction of frames where the top-1 prediction changed after patching.
Reported separately for category, attribute, and relation heads.
"""

import torch
from typing import List, Dict, Tuple, Optional


def compute_flip_rate(
    vine_model,
    clean_frames: List[torch.Tensor],
    patched_frames: List[torch.Tensor],
    masks: Dict[int, Dict[int, torch.Tensor]],
    bboxes: Dict[int, Dict[int, dict]],
    gpt_spec: dict,
    device: torch.device,
    attack_mode: str,
) -> dict:
    """
    Compute the per-frame prediction flip rate for the attacked head.

    Args:
        vine_model:      loaded VineModel
        clean_frames:    List of original [H,W,3] float tensors
        patched_frames:  List of patched [H,W,3] float tensors (same length)
        masks:           {frame_id: {obj_id: Tensor[H,W]}}
        bboxes:          {frame_id: {obj_id: {x1,y1,x2,y2}}}
        gpt_spec:        clip keyword lists for this clip
        device:          torch device
        attack_mode:     'category' | 'attribute' | 'relation'

    Returns:
        dict with keys:
            'flip_rate':        overall fraction of frame-object pairs that flipped
            'n_total':          total number of frame-object pairs evaluated
            'n_flipped':        number of pairs that flipped
            'per_frame_flips':  list[bool] — True if any object in that frame flipped
    """
    from functions.eval.scene_graph import predict_scene_graph

    sg_clean   = predict_scene_graph(vine_model, clean_frames,   masks, bboxes, gpt_spec, device)
    sg_patched = predict_scene_graph(vine_model, patched_frames, masks, bboxes, gpt_spec, device)

    if attack_mode == 'category':
        clean_preds   = sg_clean.categories
        patched_preds = sg_patched.categories
        keys          = set(clean_preds) | set(patched_preds)
    elif attack_mode == 'attribute':
        clean_preds   = sg_clean.attributes
        patched_preds = sg_patched.attributes
        keys          = set(clean_preds) | set(patched_preds)
    elif attack_mode == 'relation':
        clean_preds   = sg_clean.relations
        patched_preds = sg_patched.relations
        keys          = set(clean_preds) | set(patched_preds)
    else:
        raise ValueError(f"Unknown attack_mode: {attack_mode}")

    flips = []
    for k in keys:
        c_pred = clean_preds.get(k)
        p_pred = patched_preds.get(k)
        if c_pred is not None and p_pred is not None:
            flips.append(c_pred != p_pred)

    n_total   = len(flips)
    n_flipped = sum(flips)
    flip_rate = n_flipped / n_total if n_total > 0 else 0.0

    return {
        'flip_rate':       flip_rate,
        'n_total':         n_total,
        'n_flipped':       n_flipped,
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

    Args:
        vine_model:  loaded VineModel
        clip_items:  list of dataset items (from EscaAttackDataset)
        patch:       trained AdversarialPatch
        anchor:      Anchor(cx, cy, label)
        config:      AttackConfig
        device:      torch device

    Returns:
        dict with aggregated flip_rate and per-clip results
    """
    from functions.attack.patch import paste_patch, compute_patch_size

    all_results = []

    for item in clip_items:
        if item is None:
            continue

        clean_frames   = item['frames']
        masks          = item['masks']
        bboxes         = item['bboxes']
        gpt_spec       = item['gpt_spec']

        patched_frames = []
        for frame_id, frame in enumerate(clean_frames):
            frame_bboxes = bboxes.get(frame_id, {})
            patched      = frame.clone()
            for obj_id, bbox in frame_bboxes.items():
                ph, pw   = compute_patch_size(bbox, config.coverage_ratio)
                patched  = paste_patch(
                    patched.to(device), patch.pixels_hwc,
                    anchor.cx, anchor.cy, ph, pw,
                )
            patched_frames.append(patched.cpu())

        result = compute_flip_rate(
            vine_model, clean_frames, patched_frames,
            masks, bboxes, gpt_spec, device, config.attack_mode,
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
