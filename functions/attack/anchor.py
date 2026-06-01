"""
SAM-guided adversarial anchor selection (Phase 0).

The anchor is stored as a (dx, dy) pixel offset relative to the TARGET OBJECT'S
bbox center.  At paste time use anchor_center(anchor, bbox) which applies the
offset and then CLAMPS the result to within the bbox — this ensures the patch
always lands on the target object even when the eval bbox is smaller than the
training bbox used to derive the offset.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import List, Tuple, NamedTuple
import copy


class Anchor(NamedTuple):
    dx: int      # x-offset from bbox center in pixels (positive = right)
    dy: int      # y-offset from bbox center in pixels (positive = down)
    label: str   # informal description (e.g. "sam-anchor-0")


def _bbox_center(bbox: dict) -> Tuple[int, int]:
    return (bbox['x1'] + bbox['x2']) // 2, (bbox['y1'] + bbox['y2']) // 2


def _mask_centroid(mask_np: np.ndarray) -> Tuple[int, int]:
    """Return (cx, cy) centroid of a binary mask."""
    ys, xs = np.where(mask_np)
    if len(xs) == 0:
        return 0, 0
    return int(xs.mean()), int(ys.mean())


def _get_sam_anchors(
    image_rgb: np.ndarray,
    bbox: dict,
    sam_predictor,
) -> List[Tuple[int, int]]:
    """
    Use SAM2 with a box prompt to obtain sub-region candidates within bbox.
    Returns a list of ABSOLUTE (cx, cy) centroids — callers convert to relative.
    Falls back to the bbox center if SAM is unavailable.
    """
    if sam_predictor is None:
        cx, cy = _bbox_center(bbox)
        return [(cx, cy)]

    box_np = np.array([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']], dtype=float)

    try:
        sam_predictor.set_image(image_rgb)
        masks, _, _ = sam_predictor.predict(
            box=box_np,
            multimask_output=True,
        )
        anchors = [_mask_centroid(m) for m in masks]
        # Include the geometric bbox center as a baseline
        anchors.append(_bbox_center(bbox))
        return anchors
    except Exception as e:
        print(f"SAM prediction failed ({e}), falling back to bbox center")
        return [_bbox_center(bbox)]


def _abs_to_rel(cx: int, cy: int, bbox: dict) -> Tuple[int, int]:
    """Convert an absolute centroid to a (dx, dy) pixel offset from the bbox center."""
    bcx, bcy = _bbox_center(bbox)
    return cx - bcx, cy - bcy


def anchor_center(anchor: 'Anchor', bbox: dict) -> Tuple[int, int]:
    """
    Compute the absolute (cx, cy) paste center for an Anchor on a given bbox.

    Applies the pixel offset and then clamps the result to within the bbox, so
    the patch always lands on the target object even when the eval bbox is
    smaller than the training bboxes that produced the anchor offset.
    """
    bcx, bcy = _bbox_center(bbox)
    cx = bcx + anchor.dx
    cy = bcy + anchor.dy
    cx = max(int(bbox['x1']), min(int(bbox['x2']), cx))
    cy = max(int(bbox['y1']), min(int(bbox['y2']), cy))
    return cx, cy


def _evaluate_anchor(
    anchor_dx: int,
    anchor_dy: int,
    instances: list,
    vine_model: torch.nn.Module,
    clip_encoder,
    attack_mode: str,
    target_category: str,
    coverage_ratio: float,
    warmup_steps: int,
    device: torch.device,
    clip_image_size: int,
    target_attribute: str = '',
) -> float:
    """
    Evaluate one anchor candidate (relative offset dx, dy) by running
    warmup_steps gradient steps and returning the final average CW loss.
    Lower returned loss = better anchor.
    """
    from functions.attack.patch import AdversarialPatch, paste_patch, compute_patch_size
    from functions.attack.preprocess import (
        clip_scores,
        extract_single_object_torch,
    )
    from functions.attack.losses import l_attack_cw

    eval_instances = instances[:min(len(instances), 20)]
    if len(eval_instances) == 0:
        return float('inf')

    # For anchor evaluation we need instances with ≥2 keywords so the CW loss
    # has something to distinguish.  The first N instances might all come from
    # a single clip with a 1-word vocabulary, so search the full list.
    target_str = target_attribute if attack_mode == 'attribute' else target_category
    kw_key     = 'unary_kws' if attack_mode == 'attribute' else 'consts'
    cw_eval_instances = []
    for inst in eval_instances + list(instances):   # eval_subset first, then full list
        spec = inst['gpt_spec']
        kws  = spec.get(kw_key, [])
        cidx = next((i for i, c in enumerate(kws) if target_str.lower() in c.lower()), None)
        if cidx is not None and len(kws) >= 2:
            cw_eval_instances.append((inst, kws, cidx))
        if len(cw_eval_instances) >= 20:
            break

    if not cw_eval_instances:
        # No multi-class instances found — anchor evaluation cannot use CW loss;
        # skip warmup and return 0 (all anchors are equivalent).
        return 0.0

    sample_bbox = cw_eval_instances[0][0]['bbox']
    patch_h, patch_w = compute_patch_size(sample_bbox, coverage_ratio)
    patch = AdversarialPatch(patch_h, patch_w, device)
    optimizer = optim.Adam(patch.parameters(), lr=0.01)

    for _ in range(warmup_steps):
        total     = torch.tensor(0.0, device=device)
        n_counted = 0
        optimizer.zero_grad()

        for inst, inst_kws, inst_correct_idx in cw_eval_instances:

            frame = inst['frame'].to(device)
            mask  = inst['mask'].to(device)
            bbox  = inst['bbox']
            ph, pw = compute_patch_size(bbox, coverage_ratio)

            bcx, bcy = _bbox_center(bbox)
            cx = max(int(bbox['x1']), min(int(bbox['x2']), bcx + anchor_dx))
            cy = max(int(bbox['y1']), min(int(bbox['y2']), bcy + anchor_dy))

            patched = paste_patch(frame, patch.pixels_hwc, cx, cy, ph, pw)

            # category/attribute: eval classifies on the FULL frame (no bbox crop)
            proc   = extract_single_object_torch(patched, mask, alpha=vine_model.config.alpha)
            scores = clip_scores(vine_model, clip_encoder, proc, inst_kws, device, clip_image_size)

            total     = total + l_attack_cw(scores, inst_correct_idx)
            n_counted += 1

        (total / n_counted).backward()
        optimizer.step()

    return total.item() / n_counted


def select_anchor(
    instances: list,
    vine_model: torch.nn.Module,
    attack_mode: str,
    target_category: str,
    coverage_ratio: float = 0.3,
    warmup_steps: int = 20,
    device: torch.device = None,
    sam_predictor=None,
    clip_image_size: int = 224,
    target_attribute: str = '',
) -> Anchor:
    """
    Select the best anchor (relative offset from bbox center) via SAM-guided
    candidate generation followed by warm-up loss evaluation.
    """
    if device is None:
        device = next(vine_model.parameters()).device

    if attack_mode not in ('category', 'attribute'):
        raise ValueError("Use select_relation_anchor() for relation attack mode.")

    eval_subset = instances[:min(len(instances), 100)]
    if len(eval_subset) == 0:
        raise ValueError("No instances available for anchor selection.")

    # Gather relative-offset candidates (convert SAM absolute → relative per instance)
    candidate_votes: dict = {}   # (dx, dy) -> cumulative vote count

    for inst in eval_subset[:20]:
        frame_np = inst['frame'].numpy().astype(np.uint8)
        bbox     = inst['bbox']
        abs_anchors = _get_sam_anchors(frame_np, bbox, sam_predictor)
        for cx, cy in abs_anchors:
            dx, dy = _abs_to_rel(cx, cy, bbox)
            candidate_votes[(dx, dy)] = candidate_votes.get((dx, dy), 0) + 1

    candidates = list(candidate_votes.keys())
    if not candidates:
        candidates = [(0, 0)]   # fallback: bbox center

    print(f"Evaluating {len(candidates)} anchor candidates with {warmup_steps} warmup steps each...")

    clip_encoder = (
        vine_model.clip_cate_model if attack_mode == 'category'
        else vine_model.clip_unary_model
    )

    best_loss  = float('inf')
    best_dx, best_dy = candidates[0]
    best_label = "anchor-0"

    for idx, (dx, dy) in enumerate(candidates):
        loss = _evaluate_anchor(
            anchor_dx        = dx,
            anchor_dy        = dy,
            instances        = eval_subset,
            vine_model       = vine_model,
            clip_encoder     = clip_encoder,
            attack_mode      = attack_mode,
            target_category  = target_category,
            coverage_ratio   = coverage_ratio,
            warmup_steps     = warmup_steps,
            device           = device,
            clip_image_size  = clip_image_size,
            target_attribute = target_attribute,
        )
        print(f"  candidate {idx} (dx={dx}, dy={dy}): warmup loss = {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            best_dx, best_dy = dx, dy
            best_label = f"sam-anchor-{idx}"

    print(f"Selected anchor: (dx={best_dx}, dy={best_dy}) [{best_label}] loss={best_loss:.4f}")
    return Anchor(dx=best_dx, dy=best_dy, label=best_label)


def select_relation_anchor(
    relation_instances: list,
    vine_model: torch.nn.Module,
    target_relation: str,
    coverage_ratio: float = 0.3,
    warmup_steps: int = 20,
    device: torch.device = None,
    sam_predictor=None,
    clip_image_size: int = 224,
) -> Anchor:
    """
    Select the best anchor on the SUBJECT's bbox for the relation attack.
    Returns relative (dx, dy) offset from the subject bbox center.
    """
    if device is None:
        device = next(vine_model.parameters()).device

    eval_subset = relation_instances[:min(len(relation_instances), 100)]
    if len(eval_subset) == 0:
        raise ValueError("No relation instances available for anchor selection.")

    candidate_votes: dict = {}
    for inst in eval_subset[:20]:
        frame_np  = inst['frame'].numpy().astype(np.uint8)
        subj_bbox = inst['subj_bbox']
        abs_anchors = _get_sam_anchors(frame_np, subj_bbox, sam_predictor)
        for cx, cy in abs_anchors:
            dx, dy = _abs_to_rel(cx, cy, subj_bbox)
            candidate_votes[(dx, dy)] = candidate_votes.get((dx, dy), 0) + 1

    candidates = list(candidate_votes.keys())
    if not candidates:
        candidates = [(0, 0)]

    from functions.attack.patch import AdversarialPatch, paste_patch, compute_patch_size
    from functions.attack.preprocess import (
        clip_scores,
        extract_object_subject_torch,
        crop_image_contain_bboxes_torch,
    )
    from functions.attack.losses import l_attack_cw

    binary_kws  = eval_subset[0]['gpt_spec'].get('binary_kws', [])
    correct_idx = next(
        (i for i, k in enumerate(binary_kws) if target_relation.lower() in k.lower()),
        0,
    )

    best_loss  = float('inf')
    best_dx, best_dy = candidates[0]
    best_label = "relation-anchor-0"

    print(f"Evaluating {len(candidates)} relation anchor candidates...")

    for idx, (dx, dy) in enumerate(candidates):
        sample_bbox = eval_subset[0]['subj_bbox']
        patch_h, patch_w = compute_patch_size(sample_bbox, coverage_ratio)
        patch     = AdversarialPatch(patch_h, patch_w, device)
        optimizer = optim.Adam(patch.parameters(), lr=0.01)

        for _ in range(warmup_steps):
            total = torch.tensor(0.0, device=device)
            optimizer.zero_grad()
            for inst in eval_subset[:20]:
                frame      = inst['frame'].to(device)
                subj_mask  = inst['subj_mask'].to(device)
                obj_mask   = inst['obj_mask'].to(device)
                subj_bbox  = inst['subj_bbox']
                obj_bbox   = inst['obj_bbox']
                ph, pw     = compute_patch_size(subj_bbox, coverage_ratio)

                bcx, bcy = _bbox_center(subj_bbox)
                cx = max(int(subj_bbox['x1']), min(int(subj_bbox['x2']), bcx + dx))
                cy = max(int(subj_bbox['y1']), min(int(subj_bbox['y2']), bcy + dy))

                patched = paste_patch(frame, patch.pixels_hwc, cx, cy, ph, pw)
                # relation: eval crops to the union of subject+object bboxes
                proc    = extract_object_subject_torch(
                    patched, subj_mask, obj_mask,
                    alpha=vine_model.config.alpha, white_alpha=vine_model.config.white_alpha,
                )
                crop    = crop_image_contain_bboxes_torch(proc, [subj_bbox, obj_bbox])
                scores  = clip_scores(vine_model, vine_model.clip_binary_model,
                                      crop, binary_kws, device, clip_image_size)
                total  = total + l_attack_cw(scores, correct_idx)

            (total / len(eval_subset[:20])).backward()
            optimizer.step()

        loss_val = total.item() / len(eval_subset[:20])
        print(f"  candidate {idx} (dx={dx}, dy={dy}): warmup loss = {loss_val:.4f}")
        if loss_val < best_loss:
            best_loss  = loss_val
            best_dx, best_dy = dx, dy
            best_label = f"relation-sam-anchor-{idx}"

    print(f"Selected relation anchor: (dx={best_dx}, dy={best_dy}) loss={best_loss:.4f}")
    return Anchor(dx=best_dx, dy=best_dy, label=best_label)
