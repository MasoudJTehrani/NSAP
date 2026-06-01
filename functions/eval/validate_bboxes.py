"""
Validate that the bounding boxes derived from segmentation masks are correct and
correctly aligned with the loaded video frames.

The boxes are not annotations — they are computed at load time from the dataset's
per-frame segmentation masks (functions/attack/data.py::_bboxes_from_masks). Two
things therefore have to hold for the patch to land on the target object:

  1. Derivation: each box is the tight bound of its object's mask in that frame.
  2. Alignment : masks[k] / bboxes[k] describe the SAME frame as frames[k].

(2) is the subtle one. load_video sub-samples frames to max_vid_len; the masks must
be sub-sampled and re-indexed in lockstep or the box for frame k is read from a
different original frame and the patch drifts off the moving object.

This module runs cheap structural assertions for both properties and, optionally,
writes overlay PNGs (mask outline + box drawn on the real frame) for visual
confirmation that the box sits on the object.
"""

import os
import numpy as np
import torch
from typing import Optional


# ── structural checks ─────────────────────────────────────────────────────────

def _check_clip(item: dict) -> dict:
    """
    Run alignment / derivation checks on one loaded clip item.
    Returns a dict of failure lists (empty lists ⇒ clip passed).
    """
    frames = item['frames']
    masks  = item['masks']
    bboxes = item['bboxes']
    n      = len(frames)

    fails = {
        'frame_mask_count': [],   # frames vs masks frame-count mismatch
        'frame_id_range':   [],   # a mask/bbox frame-id outside range(n)
        'box_out_of_frame': [],   # box exceeds the frame's pixel bounds
        'box_not_tight':    [],   # box ≠ tight bound of its mask (derivation bug)
        'center_off_mask':  [],   # box centre pixel not on the object mask (alignment bug)
    }

    # (alignment) masks/bboxes must be keyed by the same frame indices as frames
    if len(masks) != n:
        fails['frame_mask_count'].append((len(masks), n))
    for fid in set(masks) | set(bboxes):
        if not (0 <= fid < n):
            fails['frame_id_range'].append(fid)

    for fid in range(n):
        H, W = frames[fid].shape[:2]
        frame_masks  = masks.get(fid, {})
        frame_bboxes = bboxes.get(fid, {})
        for obj_id, box in frame_bboxes.items():
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

            # (derivation) box must lie inside the frame
            if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
                fails['box_out_of_frame'].append((fid, obj_id, (x1, y1, x2, y2), (W, H)))

            mask = frame_masks.get(obj_id)
            if mask is None:
                continue
            ys, xs = mask.nonzero(as_tuple=True)
            if ys.numel() == 0:
                continue

            # (derivation) box must be the tight bound of THIS mask
            tx1, tx2 = int(xs.min()), int(xs.max())
            ty1, ty2 = int(ys.min()), int(ys.max())
            if (x1, y1) != (tx1, ty1) or x2 < tx2 or y2 < ty2:
                fails['box_not_tight'].append((fid, obj_id))

            # (alignment) the box centre should fall on the object's mask. A box
            # read from the wrong frame almost always centres on background.
            cx = min(int((x1 + x2) // 2), mask.shape[1] - 1)
            cy = min(int((y1 + y2) // 2), mask.shape[0] - 1)
            if not bool(mask[cy, cx]):
                fails['center_off_mask'].append((fid, obj_id))

    return fails


# ── overlay rendering ──────────────────────────────────────────────────────────

def _save_overlay(item: dict, frame_id: int, out_path: str) -> None:
    """Draw every object's mask outline + box on one frame and save it."""
    from PIL import Image, ImageDraw

    frame = item['frames'][frame_id]
    arr   = frame.detach().cpu().numpy().clip(0, 255).astype(np.uint8)
    img   = Image.fromarray(arr).convert('RGB')
    draw  = ImageDraw.Draw(img)

    palette = [
        (255, 60, 60), (60, 200, 60), (70, 130, 255), (255, 190, 0),
        (200, 60, 220), (0, 200, 200), (255, 130, 0), (160, 160, 160),
    ]
    frame_masks  = item['masks'].get(frame_id, {})
    frame_bboxes = item['bboxes'].get(frame_id, {})

    for i, (obj_id, box) in enumerate(sorted(frame_bboxes.items())):
        color = palette[i % len(palette)]
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])

        # mask boundary as a light tint of the bbox colour
        mask = frame_masks.get(obj_id)
        if mask is not None:
            m = mask.detach().cpu().numpy().astype(bool)
            edge = m ^ np.pad(m, 1, mode='constant')[1:-1, 1:-1]   # 1-px boundary
            ys, xs = np.nonzero(edge)
            for yy, xx in zip(ys.tolist(), xs.tolist()):
                img.putpixel((xx, yy), color)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        # centre cross — should sit on the object if alignment is correct
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        draw.line([cx - 4, cy, cx + 4, cy], fill=color, width=1)
        draw.line([cx, cy - 4, cx, cy + 4], fill=color, width=1)
        draw.text((x1 + 2, max(0, y1 + 1)), str(obj_id), fill=color)

    img.save(out_path)


# ── entry point ─────────────────────────────────────────────────────────────────

def validate_bboxes(
    dataset,
    n_clips: int = 10,
    out_dir: Optional[str] = None,
    frames_per_clip: int = 3,
) -> dict:
    """
    Validate bbox derivation + frame/mask alignment over up to n_clips clips.

    Args:
        dataset:         an EscaAttackDataset (any split)
        n_clips:         max clips to check
        out_dir:         if given, overlay PNGs are written to <out_dir>/bbox_validation/
        frames_per_clip: how many evenly-spaced frames to render per clip

    Returns a report dict with aggregate counts and a per-clip breakdown.
    """
    report = {
        'n_clips_checked':  0,
        'n_clips_passed':   0,
        'n_objects':        0,
        'failures':         {
            'frame_mask_count': 0, 'frame_id_range': 0,
            'box_out_of_frame': 0, 'box_not_tight': 0, 'center_off_mask': 0,
        },
        'per_clip': [],
    }

    samples_dir = None
    if out_dir:
        samples_dir = os.path.join(out_dir, "bbox_validation")
        os.makedirs(samples_dir, exist_ok=True)

    checked = 0
    for idx in range(len(dataset)):
        if checked >= n_clips:
            break
        item = dataset[idx]
        if item is None:
            continue
        checked += 1

        fails = _check_clip(item)
        n_obj = sum(len(b) for b in item['bboxes'].values())
        clip_failed = any(fails.values())

        report['n_clips_checked'] += 1
        report['n_objects']       += n_obj
        if not clip_failed:
            report['n_clips_passed'] += 1
        for k, v in fails.items():
            report['failures'][k] += len(v)
        report['per_clip'].append({
            'data_id':  item['data_id'],
            'n_frames': len(item['frames']),
            'n_objects': n_obj,
            'passed':   not clip_failed,
            'fails':    {k: v for k, v in fails.items() if v},
        })

        if samples_dir:
            n_frames = len(item['frames'])
            if n_frames:
                step = max(1, n_frames // frames_per_clip)
                for fid in range(0, n_frames, step):
                    _save_overlay(
                        item, fid,
                        os.path.join(samples_dir, f"{item['data_id']}_f{fid}.png"),
                    )

    # ── print summary ──────────────────────────────────────────────────────────
    print(f"\n=== Bounding-box validation ({report['n_clips_checked']} clips, "
          f"{report['n_objects']} object-boxes) ===")
    print(f"Clips fully passing : {report['n_clips_passed']} / {report['n_clips_checked']}")
    f = report['failures']
    print(f"  frame/mask count mismatch : {f['frame_mask_count']}   (alignment)")
    print(f"  frame-id out of range     : {f['frame_id_range']}   (alignment)")
    print(f"  box centre off its mask   : {f['center_off_mask']}   (alignment)")
    print(f"  box not tight to mask     : {f['box_not_tight']}   (derivation)")
    print(f"  box outside frame bounds  : {f['box_out_of_frame']}   (derivation)")
    if samples_dir:
        print(f"Overlay images written to : {samples_dir}/")
    if report['n_clips_passed'] != report['n_clips_checked']:
        print("  ⚠ some clips failed — inspect report['per_clip'] and the overlays.")
    else:
        print("  ✓ all checked clips passed.")
    return report
