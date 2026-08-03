"""
Shared eval helpers used by flip_rate / edit_distance / visualize, so the target
matcher and the patch-application logic live in exactly one place.
"""
import torch


def _cat_ok(label: str, target: str) -> bool:
    """Category match: exact word ('pot' != 'potato')."""
    return label.lower() == target.lower()


def _patch_object_on_frame(frame, bbox, patch, anchor, config, device: torch.device):
    """Paste the patch onto ONE frame at bbox's anchored location.

    Returns a CPU tensor [H,W,3]. The source frame is cloned (never mutated), and
    no EoT jitter is applied — eval places the patch deterministically.
    """
    from functions.attack.patch import paste_patch, compute_patch_size
    from functions.attack.anchor import anchor_center
    ph, pw = compute_patch_size(bbox, config.coverage_ratio)
    cx, cy = anchor_center(anchor, bbox, ph, pw)
    return paste_patch(
        frame.clone().to(device), patch.pixels_hwc, cx, cy, ph, pw,
    ).cpu()


def _patch_object_across_frames(clean_frames, bboxes, patch, anchor, config,
                                device: torch.device, obj_id: int) -> list:
    """
    Return a copy of clean_frames with the patch applied only to obj_id in every
    frame where that object appears. Frames without obj_id are returned unchanged.
    """
    patched = []
    for frame_id, frame in enumerate(clean_frames):
        frame_bboxes = bboxes.get(frame_id, {})
        if obj_id in frame_bboxes:
            f = _patch_object_on_frame(frame, frame_bboxes[obj_id], patch, anchor, config, device)
        else:
            f = frame
        patched.append(f)
    return patched
