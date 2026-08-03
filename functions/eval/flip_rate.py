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

from functions.eval._common import _cat_ok, _patch_object_across_frames


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
                if _cat_ok(cat, config.target_category)
            } if config.target_category else set(sg_clean.categories.keys())

            # Sim: also require GROUND-TRUTH category == target, so the target is a
            # REAL toaster — not an object VINE misreads as one (box on wrong object).
            _gt = item.get('gt_categories')
            if _gt is not None and config.target_category:
                target_obj_ids = {oid for oid in target_obj_ids
                                  if _cat_ok(_gt.get(oid, ''), config.target_category)}

            if config.attack_mode == 'attribute' and getattr(config, 'target_attribute', ''):
                target_obj_ids = {
                    oid for oid in target_obj_ids
                    if config.target_attribute.lower() in (sg_clean.attributes.get(oid) or '').lower()
                }

            # Confidence gate: drop objects VINE only weakly calls the target
            # category (e.g. a table at 0.3). Matches training's min_target_conf.
            _min_conf = getattr(config, 'min_target_conf', 0.0)
            if _min_conf > 0:
                target_obj_ids = {
                    oid for oid in target_obj_ids
                    if sg_clean.category_conf.get(oid, 0.0) >= _min_conf
                }

            # Targeting policy: attack only the largest target object per clip
            # (masks are over-segmented; ignore fragments). Matches training.
            if config.target_category:
                from functions.attack.data import largest_object_id
                best = largest_object_id(masks, target_obj_ids)
                target_obj_ids = {best} if best is not None else set()

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
                    if _cat_ok(sg_clean.categories.get(p[0], ''), config.target_category)
                }
                _gt = item.get('gt_categories')
                if _gt is not None:
                    target_pairs = {p for p in target_pairs
                                    if _cat_ok(_gt.get(p[0], ''), config.target_category)}

            _min_conf = getattr(config, 'min_target_conf', 0.0)
            if _min_conf > 0:
                target_pairs = {
                    p for p in target_pairs
                    if sg_clean.category_conf.get(p[0], 0.0) >= _min_conf
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
