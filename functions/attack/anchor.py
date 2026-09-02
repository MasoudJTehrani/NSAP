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
from functions.attack.data import load_frame as _load_frame


def _target_index(kws: List[str], target: str):
    """Index of `target` in `kws`: exact match first, else the positive (non-'not ')
    substring match, so the 2-way contrast negation is never picked."""
    t = target.lower()
    for i, c in enumerate(kws):
        if c.lower() == t:
            return i
    for i, c in enumerate(kws):
        if t in c.lower() and not c.lower().startswith("not "):
            return i
    return None


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


def anchor_center(
    anchor: 'Anchor',
    bbox: dict,
    patch_h: int = None,
    patch_w: int = None,
) -> Tuple[int, int]:
    """
    Compute the absolute (cx, cy) paste center for an Anchor on a given bbox.

    Applies the learned pixel offset, then clamps the result so the patch stays
    on the target object even when the eval bbox is smaller than the training
    bboxes that produced the offset.

    When patch_h/patch_w are given, the clamp is INSET by half the patch size so
    the WHOLE patch stays inside the bbox (otherwise a large offset pins the
    centre to the bbox edge and ~half the patch spills outside). If the patch is
    larger than the bbox on an axis, that axis falls back to the bbox centre.
    Without patch_h/patch_w it clamps the centre to the bbox edges (legacy).
    """
    x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
    bcx, bcy = _bbox_center(bbox)
    cx = bcx + anchor.dx
    cy = bcy + anchor.dy

    if patch_w is not None and patch_h is not None:
        hw, hh = patch_w // 2, patch_h // 2
        lo_x, hi_x = x1 + hw, x2 - hw
        lo_y, hi_y = y1 + hh, y2 - hh
        cx = (x1 + x2) // 2 if lo_x > hi_x else max(lo_x, min(hi_x, cx))
        cy = (y1 + y2) // 2 if lo_y > hi_y else max(lo_y, min(hi_y, cy))
    else:
        cx = max(x1, min(x2, cx))
        cy = max(y1, min(y2, cy))
    return cx, cy


def _anchor_warmup_loss(sample_bbox, coverage_ratio, warmup_steps, device, batch_loss) -> float:
    """Shared anchor warmup: init a fresh patch sized to `sample_bbox`, run `warmup_steps`
    Adam steps minimising `batch_loss(patch)` (mean CW over the eval instances at a fixed
    anchor), and return the final mean CW. Lower = better anchor. Used by both the
    category/attribute (`_evaluate_anchor`) and relation (`select_relation_anchor`) paths."""
    from functions.attack.patch import AdversarialPatch, compute_patch_size
    patch_h, patch_w = compute_patch_size(sample_bbox, coverage_ratio)
    patch = AdversarialPatch(patch_h, patch_w, device)
    optimizer = optim.Adam(patch.parameters(), lr=0.01)
    last = torch.tensor(float('inf'), device=device)
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        last = batch_loss(patch)
        last.backward()
        optimizer.step()
    return float(last.item())


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
    from functions.attack.patch import paste_patch, compute_patch_size
    from functions.attack.preprocess import (
        clip_scores,
        extract_single_object_torch,
        crop_image_contain_bboxes_torch,
    )
    from functions.attack.losses import l_attack_cw

    eval_instances = instances[:min(len(instances), 20)]
    if len(eval_instances) == 0:
        return float('inf')

    # Score anchors through the SAME objective the attack optimises:
    #   category  -> multi-way over the clip's category vocab ('consts'),
    #   attribute -> the 2-way contrast [attr, "not attr"] (matches train.py/deploy).
    is_attr    = (attack_mode == 'attribute')
    target_str = target_attribute if is_attr else target_category
    cw_eval_instances = []
    for inst in eval_instances + list(instances):   # eval_subset first, then full list
        if is_attr:
            kws, cidx = [target_attribute, f"not {target_attribute}"], 0
        else:
            kws  = inst['gpt_spec'].get('consts', [])
            cidx = _target_index(kws, target_str)
            if cidx is None or len(kws) < 2:   # need >=2 classes for a CW margin
                continue
        cw_eval_instances.append((inst, kws, cidx))
        if len(cw_eval_instances) >= 20:
            break

    if not cw_eval_instances:
        # Nothing scorable (e.g. category vocab always size 1) -- all anchors tie.
        return 0.0

    def batch_loss(patch):
        total = torch.tensor(0.0, device=device)
        for inst, inst_kws, inst_correct_idx in cw_eval_instances:
            frame = _load_frame(inst).to(device)
            mask  = inst['mask'].to(device)
            bbox  = inst['bbox']
            ph, pw = compute_patch_size(bbox, coverage_ratio)
            # Place exactly as paste-time does (inset clamp keeps the whole patch
            # inside the bbox), so the anchor is selected under the real placement.
            cx, cy = anchor_center(Anchor(anchor_dx, anchor_dy, ''), bbox, ph, pw)
            patched = paste_patch(frame, patch.pixels_hwc, cx, cy, ph, pw)
            if is_attr:
                # attribute head deployment forward: a raw bbox crop (matches train.py)
                proc = crop_image_contain_bboxes_torch(patched, [bbox])
            else:
                # category head: the FULL frame, background whitened (no bbox crop)
                proc = extract_single_object_torch(patched, mask, alpha=vine_model.config.alpha)
            scores = clip_scores(vine_model, clip_encoder, proc, inst_kws, device, clip_image_size)
            total  = total + l_attack_cw(scores, inst_correct_idx)
        return total / len(cw_eval_instances)

    return _anchor_warmup_loss(cw_eval_instances[0][0]['bbox'], coverage_ratio,
                               warmup_steps, device, batch_loss)


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

    # Gather unique relative-offset candidates (SAM absolute centroid -> offset
    # from each instance's bbox center).
    candidates, seen = [], set()
    for inst in eval_subset[:20]:
        frame_np = _load_frame(inst).numpy().astype(np.uint8)
        bbox     = inst['bbox']
        for cx, cy in _get_sam_anchors(frame_np, bbox, sam_predictor):
            dxy = _abs_to_rel(cx, cy, bbox)
            if dxy not in seen:
                seen.add(dxy)
                candidates.append(dxy)
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

    candidates, seen = [], set()
    for inst in eval_subset[:20]:
        frame_np  = _load_frame(inst).numpy().astype(np.uint8)
        subj_bbox = inst['subj_bbox']
        for cx, cy in _get_sam_anchors(frame_np, subj_bbox, sam_predictor):
            dxy = _abs_to_rel(cx, cy, subj_bbox)
            if dxy not in seen:
                seen.add(dxy)
                candidates.append(dxy)
    if not candidates:
        candidates = [(0, 0)]

    from functions.attack.patch import paste_patch, compute_patch_size
    from functions.attack.preprocess import (
        clip_scores,
        extract_object_subject_torch,
        crop_image_contain_bboxes_torch,
    )
    from functions.attack.losses import l_attack_cw

    # Score anchors through the SAME 2-way contrast the relation attack optimises
    # ([rel, "not rel"]), not one instance's multi-way binary vocab.
    binary_kws  = [target_relation, f"not {target_relation}"]
    correct_idx = 0

    best_loss  = float('inf')
    best_dx, best_dy = candidates[0]
    best_label = "relation-anchor-0"

    print(f"Evaluating {len(candidates)} relation anchor candidates...")

    for idx, (dx, dy) in enumerate(candidates):
        def batch_loss(patch, dx=dx, dy=dy):
            total = torch.tensor(0.0, device=device)
            for inst in eval_subset[:20]:
                frame      = _load_frame(inst).to(device)
                subj_mask  = inst['subj_mask'].to(device)
                obj_mask   = inst['obj_mask'].to(device)
                subj_bbox  = inst['subj_bbox']
                obj_bbox   = inst['obj_bbox']
                ph, pw     = compute_patch_size(subj_bbox, coverage_ratio)
                # Match paste-time placement (inset clamp) during selection too.
                cx, cy = anchor_center(Anchor(dx, dy, ''), subj_bbox, ph, pw)
                patched = paste_patch(frame, patch.pixels_hwc, cx, cy, ph, pw)
                # relation: eval crops to the union of subject+object bboxes
                proc    = extract_object_subject_torch(
                    patched, subj_mask, obj_mask,
                    alpha=vine_model.config.alpha, white_alpha=vine_model.config.white_alpha,
                )
                crop    = crop_image_contain_bboxes_torch(proc, [subj_bbox, obj_bbox])
                scores  = clip_scores(vine_model, vine_model.clip_binary_model,
                                      crop, binary_kws, device, clip_image_size)
                total   = total + l_attack_cw(scores, correct_idx)
            return total / len(eval_subset[:20])

        loss_val = _anchor_warmup_loss(eval_subset[0]['subj_bbox'], coverage_ratio,
                                       warmup_steps, device, batch_loss)
        print(f"  candidate {idx} (dx={dx}, dy={dy}): warmup loss = {loss_val:.4f}")
        if loss_val < best_loss:
            best_loss  = loss_val
            best_dx, best_dy = dx, dy
            best_label = f"relation-sam-anchor-{idx}"

    print(f"Selected relation anchor: (dx={best_dx}, dy={best_dy}) loss={best_loss:.4f}")
    return Anchor(dx=best_dx, dy=best_dy, label=best_label)
