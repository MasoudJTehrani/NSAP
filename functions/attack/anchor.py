"""
SAM-guided adversarial anchor selection (Phase 0).

For each target category, runs a short warm-up for each SAM sub-region candidate
and selects the anchor that maximises attack effectiveness (lowest CW loss after
warmup). This is run once per category before the main training loop.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, NamedTuple
import copy


class Anchor(NamedTuple):
    cx: int      # x-center (column) of the selected anchor
    cy: int      # y-center (row) of the selected anchor
    label: str   # informal description (e.g. "sub-mask-0")


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
    Use SAM2 with a box prompt to obtain sub-region candidates within the object bbox.
    Returns a list of (cx, cy) centroid positions — one per predicted mask.
    Falls back to the bbox center if SAM is unavailable.
    """
    if sam_predictor is None:
        cx = (bbox['x1'] + bbox['x2']) // 2
        cy = (bbox['y1'] + bbox['y2']) // 2
        return [(cx, cy)]

    box_np = np.array([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']], dtype=float)

    try:
        sam_predictor.set_image(image_rgb)
        masks, _, _ = sam_predictor.predict(
            box=box_np,
            multimask_output=True,  # returns 3 masks of varying granularity
        )
        anchors = [_mask_centroid(m) for m in masks]
        # Also include the geometric center of the bbox as a baseline candidate
        cx_geo = (bbox['x1'] + bbox['x2']) // 2
        cy_geo = (bbox['y1'] + bbox['y2']) // 2
        anchors.append((cx_geo, cy_geo))
        return anchors
    except Exception as e:
        print(f"SAM prediction failed ({e}), falling back to bbox center")
        cx = (bbox['x1'] + bbox['x2']) // 2
        cy = (bbox['y1'] + bbox['y2']) // 2
        return [(cx, cy)]


def _evaluate_anchor(
    anchor_cx: int,
    anchor_cy: int,
    instances: list,
    vine_model: torch.nn.Module,
    clip_encoder,
    attack_mode: str,
    target_category: str,
    coverage_ratio: float,
    warmup_steps: int,
    device: torch.device,
    clip_image_size: int,
) -> float:
    """
    Evaluate one anchor candidate by running warmup_steps gradient steps on a random
    patch placed at (anchor_cx, anchor_cy) and returning the final average CW loss.

    Lower returned loss = attack succeeded more easily = better anchor.
    """
    from functions.attack.patch import AdversarialPatch, paste_patch, compute_patch_size
    from functions.attack.preprocess import (
        clip_image_features,
        clip_text_features,
        extract_single_object_torch,
        extract_object_subject_torch,
        crop_image_contain_bboxes_torch,
        clip_preprocess_torch,
    )
    from functions.attack.losses import l_attack_cw

    # Use a small subset for the warm-up evaluation
    eval_instances = instances[:min(len(instances), 20)]
    if len(eval_instances) == 0:
        return float('inf')

    # Initialise a fresh patch for this candidate
    sample_bbox = eval_instances[0]['bbox']
    patch_h, patch_w = compute_patch_size(sample_bbox, coverage_ratio)
    patch = AdversarialPatch(patch_h, patch_w, device)
    optimizer = optim.Adam(patch.parameters(), lr=0.01)

    # Get correct-label index
    consts = eval_instances[0]['gpt_spec'].get('consts', [])
    correct_idx = next(
        (i for i, c in enumerate(consts) if target_category.lower() in c.lower()),
        0,
    )

    for _ in range(warmup_steps):
        total = torch.tensor(0.0, device=device)
        optimizer.zero_grad()

        for inst in eval_instances:
            frame = inst['frame'].to(device)
            mask  = inst['mask'].to(device)
            bbox  = inst['bbox']
            ph, pw = compute_patch_size(bbox, coverage_ratio)

            patched = paste_patch(frame, patch.pixels_hwc, anchor_cx, anchor_cy, ph, pw)

            if attack_mode in ('category', 'attribute'):
                proc = extract_single_object_torch(patched, mask)
                crop = crop_image_contain_bboxes_torch(proc, [bbox])
            else:
                raise ValueError(f"Use select_relation_anchor for attack_mode='relation'")

            pv   = clip_preprocess_torch(crop, clip_image_size).to(device)
            feat = clip_image_features(clip_encoder, pv)

            # Scores against all consts using clip_sim
            text_tokens = vine_model.clip_tokenizer(
                consts, return_tensors='pt', padding=True, truncation=True, max_length=75
            ).to(device)
            text_feat = clip_text_features(vine_model.clip_cate_model, **text_tokens)

            img_feat_n  = feat / feat.norm(p=2, dim=-1, keepdim=True)
            text_feat_n = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
            scores = (img_feat_n @ text_feat_n.T).squeeze(0)

            total = total + l_attack_cw(scores, correct_idx)

        (total / len(eval_instances)).backward()
        optimizer.step()

    # Return final loss value (lower = better anchor)
    return total.item() / len(eval_instances)


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
) -> Anchor:
    """
    Select the best anchor position for the universal adversarial patch via SAM-guided
    candidate generation followed by warm-up loss evaluation.

    Args:
        instances:       flat list from EscaAttackDataset.build_instance_list()
        vine_model:      loaded VineModel (for tokenizer + CLIP encoders)
        attack_mode:     'category', 'attribute', or 'relation'
        target_category: string category name being attacked (e.g. 'person')
        coverage_ratio:  patch coverage fraction of bbox (default 0.3)
        warmup_steps:    gradient steps per anchor candidate (default 20)
        device:          torch device (defaults to vine_model device)
        sam_predictor:   SAM2ImagePredictor instance, or None to skip SAM
        clip_image_size: CLIP image input size (224 or 336)

    Returns:
        Anchor(cx, cy, label)
    """
    if device is None:
        device = next(vine_model.parameters()).device

    if attack_mode not in ('category', 'attribute'):
        raise ValueError("Use select_relation_anchor() for relation attack mode.")

    # Sample a small set of representative instances for anchor evaluation
    eval_subset = instances[:min(len(instances), 100)]
    if len(eval_subset) == 0:
        raise ValueError("No instances available for anchor selection.")

    # Gather anchor candidates across instances (centroid of SAM sub-masks)
    candidate_votes: dict = {}   # (cx, cy) -> cumulative vote count

    for inst in eval_subset[:20]:   # use first 20 for SAM (expensive step)
        frame_np = inst['frame'].numpy().astype(np.uint8)
        bbox     = inst['bbox']
        anchors  = _get_sam_anchors(frame_np, bbox, sam_predictor)
        for cx, cy in anchors:
            candidate_votes[(cx, cy)] = candidate_votes.get((cx, cy), 0) + 1

    # Deduplicate nearby candidates (merge within 5% of image size)
    candidates = list(candidate_votes.keys())
    if len(candidates) == 0:
        # Fallback to bbox center of first instance
        bbox = eval_subset[0]['bbox']
        candidates = [((bbox['x1'] + bbox['x2']) // 2, (bbox['y1'] + bbox['y2']) // 2)]

    print(f"Evaluating {len(candidates)} anchor candidates with {warmup_steps} warmup steps each...")

    clip_encoder = (
        vine_model.clip_cate_model if attack_mode == 'category'
        else vine_model.clip_unary_model
    )

    best_loss = float('inf')
    best_cx, best_cy = candidates[0]
    best_label = "anchor-0"

    for idx, (cx, cy) in enumerate(candidates):
        loss = _evaluate_anchor(
            anchor_cx=cx,
            anchor_cy=cy,
            instances=eval_subset,
            vine_model=vine_model,
            clip_encoder=clip_encoder,
            attack_mode=attack_mode,
            target_category=target_category,
            coverage_ratio=coverage_ratio,
            warmup_steps=warmup_steps,
            device=device,
            clip_image_size=clip_image_size,
        )
        print(f"  candidate {idx} ({cx}, {cy}): warmup loss = {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            best_cx, best_cy = cx, cy
            best_label = f"sam-anchor-{idx}"

    print(f"Selected anchor: ({best_cx}, {best_cy}) [{best_label}] with loss {best_loss:.4f}")
    return Anchor(cx=best_cx, cy=best_cy, label=best_label)


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
    Same SAM + warmup approach as select_anchor but uses the pair crop
    and clip_binary_model for the warm-up loss.
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
        anchors   = _get_sam_anchors(frame_np, subj_bbox, sam_predictor)
        for cx, cy in anchors:
            candidate_votes[(cx, cy)] = candidate_votes.get((cx, cy), 0) + 1

    candidates = list(candidate_votes.keys())
    if not candidates:
        bbox = eval_subset[0]['subj_bbox']
        candidates = [((bbox['x1'] + bbox['x2']) // 2, (bbox['y1'] + bbox['y2']) // 2)]

    from functions.attack.patch import AdversarialPatch, paste_patch, compute_patch_size
    from functions.attack.preprocess import (
        clip_image_features,
        clip_text_features,
        extract_object_subject_torch,
        crop_image_contain_bboxes_torch,
        clip_preprocess_torch,
    )
    from functions.attack.losses import l_attack_cw

    binary_kws = eval_subset[0]['gpt_spec'].get('binary_kws', [])
    correct_idx = next(
        (i for i, k in enumerate(binary_kws) if target_relation.lower() in k.lower()),
        0,
    )

    best_loss = float('inf')
    best_cx, best_cy = candidates[0]
    best_label = "relation-anchor-0"

    print(f"Evaluating {len(candidates)} relation anchor candidates...")

    for idx, (cx, cy) in enumerate(candidates):
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

                patched = paste_patch(frame, patch.pixels_hwc, cx, cy, ph, pw)
                proc    = extract_object_subject_torch(patched, subj_mask, obj_mask)
                crop    = crop_image_contain_bboxes_torch(proc, [subj_bbox, obj_bbox])
                pv      = clip_preprocess_torch(crop, clip_image_size).to(device)
                feat    = clip_image_features(vine_model.clip_binary_model, pv)

                text_tokens = vine_model.clip_tokenizer(
                    binary_kws, return_tensors='pt', padding=True, truncation=True, max_length=75
                ).to(device)
                text_feat   = clip_text_features(vine_model.clip_binary_model, **text_tokens)
                img_n  = feat  / feat.norm(p=2, dim=-1, keepdim=True)
                text_n = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
                scores = (img_n @ text_n.T).squeeze(0)
                total  = total + l_attack_cw(scores, correct_idx)

            (total / len(eval_subset[:20])).backward()
            optimizer.step()

        loss_val = total.item() / len(eval_subset[:20])
        print(f"  candidate {idx} ({cx}, {cy}): warmup loss = {loss_val:.4f}")
        if loss_val < best_loss:
            best_loss  = loss_val
            best_cx, best_cy = cx, cy
            best_label = f"relation-sam-anchor-{idx}"

    print(f"Selected relation anchor: ({best_cx}, {best_cy}) with loss {best_loss:.4f}")
    return Anchor(cx=best_cx, cy=best_cy, label=best_label)
