"""
Scene graph edit distance: proves attack isolation.

All metrics are computed on the TARGET object(s) only:
  - category/attribute attack: objects whose clean-SG category matches target_category
    (and whose clean-SG attribute matches target_attribute for attribute mode)
  - relation attack: (subj, obj) pairs whose clean-SG relation matches target_relation

For each target object the patch is applied only to THAT object across every
frame where it appears; then the full multi-frame scene graph is re-predicted
and compared against the clean graph.  This mirrors the real-world scenario
where a single person wears the physical patch.

Results are POOLED across frames as raw counts, not per-clip means: with a
generated dataset every sample is one frame, so per-clip means only dilute the
signal. Isolation = fraction of all target-object changes that hit the attacked head.
"""

from typing import Dict, Tuple
from dataclasses import dataclass, field
from functions.eval.scene_graph import SceneGraph
from functions.eval._common import _cat_ok, _patch_object_across_frames


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EditDistanceResult:
    """Per-frame change counts on the target object(s). A plain data holder;
    all aggregation (pooled counts, isolation) is done in evaluate_isolation."""
    n_target_objects: int = 0     # distinct target objects patched (relation: subjects)
    n_target_relations: int = 0   # target relation edges (relation mode only)

    category_changes: int = 0
    attribute_changes: int = 0
    relation_changes: int = 0
    target_disappeared: int = 0


# ── main evaluation entry point ───────────────────────────────────────────────

def evaluate_isolation(
    vine_model,
    clip_items: list,
    patch,
    anchor,
    config,
    device,
) -> dict:
    """
    Compute target-object edit distance and isolation over test clips.

    For each target object the patch is applied only to THAT object across
    every frame where it appears (other objects left clean).  Changes on all
    three heads (category, attribute, relation) are measured for that object,
    giving a true picture of attack focus vs collateral damage.

    Returns dict of POOLED raw counts:
        total_target_objects / total_target_relations   int
        total_{category,attribute,relation}_changes      int
        total_disappeared / total_changes                int
        pooled_isolation_score                           float | None
        per_clip                                         list[EditDistanceResult]
    """
    from functions.eval.scene_graph import predict_scene_graph

    all_edits = []

    # Eval parity: ground the attacked attribute/relation via the 2-way [k, "not k"]
    # contrast (same forward as train.py + ESCA), not the multi-way vocab.
    _uk = ([config.target_attribute, f"not {config.target_attribute}"]
           if config.attack_mode == 'attribute' and getattr(config, 'target_attribute', '') else None)
    _bk = ([config.target_relation, f"not {config.target_relation}"]
           if config.attack_mode == 'relation' and getattr(config, 'target_relation', '') else None)

    for item in clip_items:
        if item is None:
            continue

        clean_frames = item['frames']
        masks        = item['masks']
        bboxes       = item['bboxes']
        gpt_spec     = item['gpt_spec']

        sg_clean = predict_scene_graph(vine_model, clean_frames, masks, bboxes, gpt_spec, device,
                                       unary_kws_override=_uk, binary_kws_override=_bk)
        clip_result = EditDistanceResult()

        if config.attack_mode in ('category', 'attribute'):
            target_obj_ids = {
                oid for oid, cat in sg_clean.categories.items()
                if _cat_ok(cat, config.target_category)
            } if config.target_category else set(sg_clean.categories.keys())

            # Sim: also require GROUND-TRUTH category == target (real object only).
            _gt = item.get('gt_categories')
            if _gt is not None and config.target_category:
                target_obj_ids = {oid for oid in target_obj_ids
                                  if _cat_ok(_gt.get(oid, ''), config.target_category)}

            if config.attack_mode == 'attribute' and getattr(config, 'target_attribute', ''):
                # exact-positive (2-way grounding yields exactly "k" or "not k").
                target_obj_ids = {
                    oid for oid in target_obj_ids
                    if (sg_clean.attributes.get(oid) or '').lower() == config.target_attribute.lower()
                }

            _min_conf = getattr(config, 'min_target_conf', 0.0)
            if _min_conf > 0:
                target_obj_ids = {
                    oid for oid in target_obj_ids
                    if sg_clean.category_conf.get(oid, 0.0) >= _min_conf
                }

            # Targeting policy: largest target object per clip (matches flip_rate
            # and training). Without this, isolation counts fragments too.
            if config.target_category:
                from functions.attack.data import largest_object_id
                best = largest_object_id(masks, target_obj_ids)
                target_obj_ids = {best} if best is not None else set()

            clip_result.n_target_objects = len(target_obj_ids)

            for obj_id in target_obj_ids:
                patched_frames = _patch_object_across_frames(
                    clean_frames, bboxes, patch, anchor, config, device, obj_id,
                )
                sg_patched = predict_scene_graph(
                    vine_model, patched_frames, masks, bboxes, gpt_spec, device,
                    unary_kws_override=_uk, binary_kws_override=_bk,
                )

                if obj_id not in sg_patched.categories:
                    clip_result.target_disappeared += 1
                    continue

                clean_cat  = sg_clean.categories.get(obj_id)
                patch_cat  = sg_patched.categories.get(obj_id)
                if clean_cat and patch_cat and clean_cat != patch_cat:
                    clip_result.category_changes += 1

                clean_attr = sg_clean.attributes.get(obj_id)
                patch_attr = sg_patched.attributes.get(obj_id)
                if clean_attr and patch_attr and clean_attr != patch_attr:
                    clip_result.attribute_changes += 1

        elif config.attack_mode == 'relation':
            target_pairs = {
                pair for pair, rel in sg_clean.relations.items()
                if rel.lower() == config.target_relation.lower()   # exact-positive (2-way)
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

            target_subj_ids = {pair[0] for pair in target_pairs}
            clip_result.n_target_relations = len(target_pairs)
            clip_result.n_target_objects   = len(target_subj_ids)

            # Per unique subject: patch it, then check relation + category + attribute
            for subj_id in target_subj_ids:
                patched_frames = _patch_object_across_frames(
                    clean_frames, bboxes, patch, anchor, config, device, subj_id,
                )
                sg_patched = predict_scene_graph(
                    vine_model, patched_frames, masks, bboxes, gpt_spec, device,
                    unary_kws_override=_uk, binary_kws_override=_bk,
                )

                if subj_id not in sg_patched.categories:
                    clip_result.target_disappeared += 1
                    continue

                clean_cat = sg_clean.categories.get(subj_id)
                patch_cat = sg_patched.categories.get(subj_id)
                if clean_cat and patch_cat and clean_cat != patch_cat:
                    clip_result.category_changes += 1

                clean_attr = sg_clean.attributes.get(subj_id)
                patch_attr = sg_patched.attributes.get(subj_id)
                if clean_attr and patch_attr and clean_attr != patch_attr:
                    clip_result.attribute_changes += 1

                for pair in target_pairs:
                    if pair[0] != subj_id:
                        continue
                    clean_rel = sg_clean.relations.get(pair)
                    patch_rel = sg_patched.relations.get(pair)
                    if clean_rel and patch_rel and clean_rel != patch_rel:
                        clip_result.relation_changes += 1

        all_edits.append(clip_result)

    if not all_edits:
        return {}

    n = len(all_edits)

    # Raw totals (pooled across clips) — interpretable counts, not per-clip means.
    tot_obj     = sum(e.n_target_objects   for e in all_edits)
    tot_rel     = sum(e.n_target_relations for e in all_edits)
    tot_cat_ch  = sum(e.category_changes   for e in all_edits)
    tot_attr_ch = sum(e.attribute_changes  for e in all_edits)
    tot_rel_ch  = sum(e.relation_changes   for e in all_edits)
    tot_disp    = sum(e.target_disappeared for e in all_edits)
    tot_changes = tot_cat_ch + tot_attr_ch + tot_rel_ch + tot_disp

    if config.attack_mode == 'category':
        attacked_total = tot_cat_ch + tot_disp
    elif config.attack_mode == 'attribute':
        attacked_total = tot_attr_ch
    elif config.attack_mode == 'relation':
        attacked_total = tot_rel_ch
    else:
        attacked_total = 0
    # Pooled isolation: of ALL changes on target objects, the fraction that hit
    # the attacked head. None when there were NO changes at all (0 target objects,
    # or the attack flipped nothing), isolation is undefined then, NOT 100%.
    pooled_isolation = (attacked_total / tot_changes) if tot_changes else None

    return {
        # raw counts (pooled across frames)
        'n_clips':                  n,
        'n_clips_with_targets':     sum(1 for e in all_edits if e.n_target_objects > 0),
        'total_target_objects':     tot_obj,
        'total_target_relations':   tot_rel,
        'total_category_changes':   tot_cat_ch,
        'total_attribute_changes':  tot_attr_ch,
        'total_relation_changes':   tot_rel_ch,
        'total_disappeared':        tot_disp,
        'total_changes':            tot_changes,
        'pooled_isolation_score':   pooled_isolation,
        'per_clip':                 all_edits,
    }
