import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AdversarialPatch(nn.Module):
    """
    Universal adversarial patch using tanh parameterization.

    Stores an unconstrained weight tensor w; pixel values are derived as:
        pixels = 127.5 * (tanh(w) + 1)   → range [0, 255] automatically

    This avoids projection steps and keeps the optimization landscape smooth.
    """

    def __init__(self, patch_h: int, patch_w: int, device: torch.device):
        super().__init__()
        # Initialise w ~ Uniform(-0.5, 0.5) so initial pixels are near 127 (mid-grey)
        w_init = torch.zeros(3, patch_h, patch_w).uniform_(-0.5, 0.5)
        self.w = nn.Parameter(w_init.to(device))

    @property
    def pixels(self) -> torch.Tensor:
        """[3, H, W] float tensor with pixel values in [0, 255]."""
        return 127.5 * (torch.tanh(self.w) + 1.0)

    @property
    def pixels_hwc(self) -> torch.Tensor:
        """[H, W, 3] float tensor — matches frame layout used in preprocess.py."""
        return self.pixels.permute(1, 2, 0)


def paste_patch(
    frame: torch.Tensor,
    patch_hwc: torch.Tensor,
    anchor_cx: int,
    anchor_cy: int,
    patch_h: int,
    patch_w: int,
    offset_range: float = 0.0,
) -> torch.Tensor:
    """
    Paste patch_hwc onto frame at the specified anchor position.

    The anchor (anchor_cx, anchor_cy) is the center of the paste region.
    An optional random offset is added per EoT sample to improve robustness.

    Tensor indexing preserves gradients through the patch pixels.

    Args:
        frame:       [H, W, 3] float tensor — will NOT be modified in-place
        patch_hwc:   [ph, pw, 3] float tensor — the adversarial patch
        anchor_cx:   x-coordinate of patch center (column index)
        anchor_cy:   y-coordinate of patch center (row index)
        patch_h:     height of patch to paste (may differ from patch_hwc size if resized)
        patch_w:     width of patch to paste
        offset_range: fraction of patch size for random displacement (default 0 = no offset)

    Returns:
        [H, W, 3] float tensor with patch pasted — new tensor, no in-place ops
    """
    H, W = frame.shape[:2]

    # Random offset (EoT stochastic displacement within ±offset_range of patch size)
    if offset_range > 0.0:
        dx = int(torch.empty(1).uniform_(-offset_range * patch_w, offset_range * patch_w).item())
        dy = int(torch.empty(1).uniform_(-offset_range * patch_h, offset_range * patch_h).item())
    else:
        dx, dy = 0, 0

    cx = anchor_cx + dx
    cy = anchor_cy + dy

    # Resize patch_hwc to (patch_h, patch_w) if dimensions differ
    if patch_hwc.shape[0] != patch_h or patch_hwc.shape[1] != patch_w:
        p = patch_hwc.permute(2, 0, 1).unsqueeze(0)   # [1,3,ph,pw]
        p = F.interpolate(p, size=(patch_h, patch_w), mode='bilinear', align_corners=False)
        patch_hwc = p.squeeze(0).permute(1, 2, 0)     # [patch_h, patch_w, 3]

    # Compute top-left frame corner, then offset into patch for edge-clamped cases
    raw_x1 = cx - patch_w // 2
    raw_y1 = cy - patch_h // 2
    x1 = max(0, raw_x1)
    y1 = max(0, raw_y1)
    px1 = x1 - raw_x1   # > 0 when anchor is near left/top edge
    py1 = y1 - raw_y1

    # Available patch rows/cols after the edge-offset
    rh = min(patch_h - py1, H - y1)
    rw = min(patch_w - px1, W - x1)

    if rh <= 0 or rw <= 0:
        # Patch is entirely outside the frame. Return frame unchanged in value but
        # keep the grad_fn chain alive through a zero contribution from patch_hwc so
        # .backward() can reach patch.w even when the patch isn't visible.
        return frame + patch_hwc.sum() * 0.0

    x2 = x1 + rw
    y2 = y1 + rh

    patch_crop = patch_hwc[py1: py1 + rh, px1: px1 + rw]  # [rh, rw, 3]

    # Differentiable paste using F.pad + spatial mask (no in-place ops) so
    # gradients flow cleanly from patch_crop back through patch.w.
    patch_chw = patch_crop.permute(2, 0, 1).unsqueeze(0)         # [1,3,rh,rw]
    patch_full = F.pad(patch_chw, (x1, W - x2, y1, H - y2))    # [1,3,H,W]
    patch_full = patch_full.squeeze(0).permute(1, 2, 0)          # [H,W,3]

    rows = torch.arange(H, device=frame.device)
    cols = torch.arange(W, device=frame.device)
    row_m = ((rows >= y1) & (rows < y2)).float().unsqueeze(1)    # [H,1]
    col_m = ((cols >= x1) & (cols < x2)).float().unsqueeze(0)    # [1,W]
    spatial_mask = (row_m * col_m).unsqueeze(-1)                  # [H,W,1]

    return frame * (1.0 - spatial_mask) + patch_full * spatial_mask


def compute_patch_size(bbox: dict, coverage_ratio: float) -> Tuple[int, int]:
    """
    Compute patch dimensions as a fraction of the target bounding box.

    Args:
        bbox:           {x1, y1, x2, y2}
        coverage_ratio: fraction of bbox dimensions (e.g., 0.3 = 30%)

    Returns:
        (patch_h, patch_w) in pixels, minimum 1×1
    """
    bw = bbox['x2'] - bbox['x1']
    bh = bbox['y2'] - bbox['y1']
    patch_w = max(1, int(coverage_ratio * bw))
    patch_h = max(1, int(coverage_ratio * bh))
    return patch_h, patch_w


def inject_patch(frame_uint8, bbox, w, anchor, cov):
    """Decode a trained patch tensor and paste it onto `bbox` in a full-frame RGB uint8
    image, using the exact training/eval placement math. Single source of truth for the
    MIM injectors (rung2/harness.render_inject and rung2/nsap_inject.paste_on_bbox).

    Args:
        frame_uint8: [H, W, 3] uint8 RGB frame (not modified in place).
        bbox:        {x1, y1, x2, y2} pixel dict, or None -> pass through.
        w:           trained patch weight [3, ph, pw] (from load_patch), or None -> pass through.
        anchor:      the patch Anchor (from load_patch).
        cov:         coverage_ratio (from load_patch).

    Returns:
        (patched_uint8, injected_bool).
    """
    from functions.attack.anchor import anchor_center   # lazy: keep patch.py import-cycle-free
    if w is None or bbox is None:
        return frame_uint8, False
    ph, pw = compute_patch_size(bbox, cov)
    patch_hwc = (127.5 * (torch.tanh(w) + 1.0)).permute(1, 2, 0)
    cx, cy = anchor_center(anchor, bbox, ph, pw)
    ft = torch.from_numpy(frame_uint8.astype(np.float32))
    out = paste_patch(ft, patch_hwc, cx, cy, ph, pw).clamp(0, 255).numpy().astype(np.uint8)
    return out, True
