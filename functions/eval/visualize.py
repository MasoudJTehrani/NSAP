"""
Save sample visualisation images from evaluation:
  - up to n_samples frames where the attack WORKED (target prediction flipped)
  - up to n_samples frames where the attack FAILED (prediction unchanged)

Each saved PNG shows the patched frame, the target object's bbox, and the
clean vs attacked prediction overlaid as text.
"""

import os
import numpy as np
import torch
from typing import List, Optional


# ── helpers ──────────────────────────────────────────────────────────────────

def _tensor_to_pil(frame: torch.Tensor):
    """Convert [H,W,3] float32 tensor (0-255) to PIL Image."""
    from PIL import Image
    arr = frame.detach().cpu().numpy().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _try_font(size: int):
    """Load a truetype font if possible, fall back to PIL default."""
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_sample(
    frame: torch.Tensor,
    bbox: dict,
    clean_pred: Optional[str],
    patched_pred: Optional[str],
    worked: bool,
    data_id: str,
    frame_id: int,
    attacked_head: str,
):
    """
    Draw a single sample image:
      - Coloured header bar (green=worked, red=failed)
      - Target-object bbox rectangle
      - Prediction text
    Returns a PIL Image.
    """
    from PIL import Image, ImageDraw

    img  = _tensor_to_pil(frame)
    W, H = img.size
    BAR  = 48                       # header height in pixels
    color = (30, 180, 30) if worked else (210, 40, 40)
    label = "WORKED" if worked else "FAILED"

    # Extend canvas downward for a clean header (paste original below)
    canvas = Image.new("RGB", (W, H + BAR), color=(20, 20, 20))
    canvas.paste(img, (0, BAR))
    draw = ImageDraw.Draw(canvas)

    # Header bar background
    draw.rectangle([0, 0, W, BAR], fill=color)

    # Header text
    f_big  = _try_font(18)
    f_small = _try_font(13)
    draw.text((8, 4),  label,   font=f_big,   fill=(255, 255, 255))
    clean_str   = clean_pred   or "—"
    patched_str = patched_pred or "(disappeared)"
    header_detail = f"{attacked_head}: {clean_str} → {patched_str}   clip={data_id}  frame={frame_id}"
    draw.text((8, 26), header_detail, font=f_small, fill=(255, 255, 255))

    # Target object bbox (shifted down by BAR)
    x1, y1 = int(bbox['x1']), int(bbox['y1']) + BAR
    x2, y2 = int(bbox['x2']), int(bbox['y2']) + BAR
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    # Small prediction label near the bbox
    txt = f"{clean_str} → {patched_str}"
    tx  = max(0, x1)
    ty  = max(BAR, y1 - 18)
    draw.rectangle([tx, ty, tx + len(txt) * 7 + 4, ty + 16], fill=(0, 0, 0, 180))
    draw.text((tx + 2, ty + 1), txt, font=f_small, fill=(255, 255, 255))

    return canvas


# ── main entry point ─────────────────────────────────────────────────────────

def save_sample_images(
    vine_model,
    clip_items: list,
    patch,
    anchor,
    config,
    device: torch.device,
    output_dir: str,
    n_samples: int = 10,
):
    """
    Generate and save worked / failed sample images from the test clips.

    Saves to <output_dir>/samples/worked_XX_<clip>_fYY.png  and
              <output_dir>/samples/failed_XX_<clip>_fYY.png

    Args:
        clip_items:  list of EscaAttackDataset items (may contain None)
        patch:       trained AdversarialPatch
        anchor:      Anchor(cx, cy, label)
        config:      AttackConfig
        device:      torch device
        output_dir:  directory where samples/ will be created
        n_samples:   max images per outcome category (default 10)
    """
    from functions.attack.patch import paste_patch, compute_patch_size
    from functions.attack.anchor import anchor_center
    from functions.eval.scene_graph import predict_scene_graph

    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    mode            = config.attack_mode
    target_category = config.target_category
    target_relation = getattr(config, 'target_relation', '')
    target_attribute = getattr(config, 'target_attribute', '')

    worked_list: list = []
    failed_list:  list = []

    for item in clip_items:
        if item is None:
            continue
        if len(worked_list) >= n_samples and len(failed_list) >= n_samples:
            break

        clean_frames = item['frames']
        masks        = item['masks']
        bboxes       = item['bboxes']
        gpt_spec     = item['gpt_spec']
        data_id      = item['data_id']

        # Clip-level scene graph on clean frames (used for prediction labels)
        sg_clean    = predict_scene_graph(vine_model, clean_frames, masks, bboxes, gpt_spec, device)

        # ── Collect per-object entries ────────────────────────────────────────
        entries = []

        if mode in ('category', 'attribute'):
            # Filter: category must match target_category
            target_obj_ids = {
                oid for oid, cat in sg_clean.categories.items()
                if target_category.lower() in cat.lower()
            }
            # For attribute mode also require the clean attribute to match target_attribute
            if mode == 'attribute' and target_attribute:
                target_obj_ids = {
                    oid for oid in target_obj_ids
                    if target_attribute.lower() in (sg_clean.attributes.get(oid) or '').lower()
                }

            attacked_head = 'category' if mode == 'category' else 'attribute'
            clean_preds   = sg_clean.categories if mode == 'category' else sg_clean.attributes

            for obj_id in target_obj_ids:
                clean_p = clean_preds.get(obj_id)
                if clean_p is None:
                    continue

                # Find a frame that contains this object
                frame_id = next(
                    (fid for fid in sorted(bboxes.keys()) if obj_id in bboxes[fid]),
                    None,
                )
                if frame_id is None or frame_id >= len(clean_frames):
                    continue

                bbox   = bboxes[frame_id][obj_id]
                ph, pw = compute_patch_size(bbox, config.coverage_ratio)
                cx, cy = anchor_center(anchor, bbox)

                # Patch ONLY this target object for the visualisation frame
                patched_frame = paste_patch(
                    clean_frames[frame_id].clone().to(device),
                    patch.pixels_hwc, cx, cy, ph, pw,
                ).cpu()

                # Get attacked prediction for this single patched frame
                sg_patched_single = predict_scene_graph(
                    vine_model, [patched_frame], {0: masks.get(frame_id, {})},
                    {0: bboxes.get(frame_id, {})}, gpt_spec, device,
                )
                patched_preds = (
                    sg_patched_single.categories if mode == 'category'
                    else sg_patched_single.attributes
                )
                patched_p = patched_preds.get(obj_id)
                did_flip  = (clean_p != patched_p) if patched_p is not None else True

                entries.append({
                    'frame':         patched_frame,
                    'bbox':          bbox,
                    'clean_pred':    clean_p,
                    'patched_pred':  patched_p,
                    'worked':        did_flip,
                    'data_id':       data_id,
                    'frame_id':      frame_id,
                    'attacked_head': attacked_head,
                })

        elif mode == 'relation':
            attacked_head = 'relation'
            target_pairs = {
                pair for pair, rel in sg_clean.relations.items()
                if target_relation.lower() in rel.lower()
            }
            if target_category:
                target_pairs = {
                    p for p in target_pairs
                    if target_category.lower() in sg_clean.categories.get(p[0], '').lower()
                }

            for pair in target_pairs:
                subj_id = pair[0]
                clean_p = sg_clean.relations.get(pair)
                if clean_p is None:
                    continue

                frame_id = next(
                    (fid for fid in sorted(bboxes.keys()) if subj_id in bboxes[fid]),
                    None,
                )
                if frame_id is None or frame_id >= len(clean_frames):
                    continue

                subj_bbox = bboxes[frame_id][subj_id]
                ph, pw    = compute_patch_size(subj_bbox, config.coverage_ratio)
                cx, cy    = anchor_center(anchor, subj_bbox)

                # Patch ONLY the subject object for this visualisation frame
                patched_frame = paste_patch(
                    clean_frames[frame_id].clone().to(device),
                    patch.pixels_hwc, cx, cy, ph, pw,
                ).cpu()

                sg_patched_single = predict_scene_graph(
                    vine_model, [patched_frame], {0: masks.get(frame_id, {})},
                    {0: bboxes.get(frame_id, {})}, gpt_spec, device,
                )
                patched_p = sg_patched_single.relations.get(pair)
                did_flip  = (clean_p != patched_p) if patched_p is not None else True

                entries.append({
                    'frame':         patched_frame,
                    'bbox':          subj_bbox,
                    'clean_pred':    clean_p,
                    'patched_pred':  patched_p,
                    'worked':        did_flip,
                    'data_id':       data_id,
                    'frame_id':      frame_id,
                    'attacked_head': attacked_head,
                })

        # ── Sort into buckets ─────────────────────────────────────────────────
        for e in entries:
            if e['worked'] and len(worked_list) < n_samples:
                worked_list.append(e)
            elif not e['worked'] and len(failed_list) < n_samples:
                failed_list.append(e)

    # ── Save images ───────────────────────────────────────────────────────────
    for i, e in enumerate(worked_list):
        img = _draw_sample(
            e['frame'], e['bbox'],
            e['clean_pred'], e['patched_pred'],
            worked=True,
            data_id=e['data_id'], frame_id=e['frame_id'],
            attacked_head=e['attacked_head'],
        )
        img.save(os.path.join(samples_dir, f"worked_{i:02d}_{e['data_id']}_f{e['frame_id']}.png"))

    for i, e in enumerate(failed_list):
        img = _draw_sample(
            e['frame'], e['bbox'],
            e['clean_pred'], e['patched_pred'],
            worked=False,
            data_id=e['data_id'], frame_id=e['frame_id'],
            attacked_head=e['attacked_head'],
        )
        img.save(os.path.join(samples_dir, f"failed_{i:02d}_{e['data_id']}_f{e['frame_id']}.png"))

    print(f"Saved {len(worked_list)} worked + {len(failed_list)} failed sample images → {samples_dir}/")
