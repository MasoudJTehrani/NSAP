"""
Train-vs-eval forward consistency check.

Eval is the source of truth: VineModel.predict() (numpy + cv2 + CLIPProcessor,
no-grad) produces the scene-graph predictions the attack is scored against.
Training optimises the patch through a differentiable torch reimplementation of
that same forward. If the two diverge, the patch is optimised against a function
it is never measured on, and good training loss will not transfer to flip rate.

This module runs BOTH forwards on the same (frame, mask, bbox) and reports how
far the per-keyword probability vectors differ. It is used both as a one-off
diagnostic and as a permanent guardrail: after the fixes, train≈eval to a small
tolerance.

Run:  python -m functions.eval.consistency_check
"""

import torch
import torch.nn.functional as F
from typing import List


# ── eval-side probabilities (the source of truth) ──────────────────────────────

def eval_category_probs(vine_model, frame_hwc, mask, bbox, keywords, device):
    """Per-keyword category probabilities from VineModel.predict() for one object."""
    video  = frame_hwc.unsqueeze(0).to(device)            # [1,H,W,3]
    masks  = {0: {0: mask.to(device)}}
    bboxes = {0: {0: [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]}}
    res = vine_model.predict(
        video_frames=video, masks=masks, bboxes=bboxes,
        categorical_keywords=keywords, return_top_k=len(keywords),
    )
    preds = res.get('categorical_predictions', {}).get(0, [])   # [(prob,label),...]
    by_label = {lbl: p for p, lbl in preds}
    return torch.tensor([by_label.get(k, 0.0) for k in keywords], dtype=torch.float32)


# ── train-side probabilities (the differentiable mirror) ────────────────────────

def train_category_probs(vine_model, frame_hwc, mask, bbox, keywords, device,
                          image_size, alpha, crop_to_bbox):
    """
    Per-keyword category probabilities through the SAME helpers train.py uses.
    crop_to_bbox / alpha are the factors we are A/B testing against eval.
    """
    from functions.attack.preprocess import (
        extract_single_object_torch, crop_image_contain_bboxes_torch,
    )
    from functions.attack.train import _clip_scores

    proc = extract_single_object_torch(frame_hwc.to(device), mask.to(device), alpha=alpha)
    img  = crop_image_contain_bboxes_torch(proc, [bbox]) if crop_to_bbox else proc
    with torch.no_grad():
        scores = _clip_scores(vine_model, vine_model.clip_cate_model,
                              img, keywords, device, image_size)
    return torch.softmax(scores.float().cpu(), dim=-1)


# ── comparison ──────────────────────────────────────────────────────────────────

# Max per-keyword probability gap the shipped forward may differ from eval before
# it counts as a train/eval parity regression.
TOLERANCE = 0.05


def _report(name, eval_p, train_p, keywords):
    max_abs = (eval_p - train_p).abs().max().item()
    l1      = (eval_p - train_p).abs().sum().item()
    same_top1 = (eval_p.argmax().item() == train_p.argmax().item())
    print(f"  [{name:30s}] max|Δp|={max_abs:6.3f}  L1={l1:6.3f}  "
          f"top1 {'MATCH' if same_top1 else 'DIFFER'} "
          f"(eval={keywords[eval_p.argmax()]:>12s} | train={keywords[train_p.argmax()]:>12s})")
    return max_abs, same_top1


def run(vine_model, frame_hwc, mask, bbox, keywords, device, image_size,
        eval_alpha):
    """Print eval-vs-train divergence across train configs; return True iff the
    SHIPPED forward matches eval within tolerance (top-1 + max|Δp|)."""
    eval_p = eval_category_probs(vine_model, frame_hwc, mask, bbox, keywords, device)
    print(f"  eval top-1 = {keywords[eval_p.argmax()]}   probs={[round(x,3) for x in eval_p.tolist()]}")

    # The FIRST entry is what train.py actually uses: the full frame (no bbox crop)
    # with the eval extract alpha. The rest are A/B references showing how crop /
    # alpha choices move parity.
    SHIPPED = "SHIPPED train (no-crop, a=eval)"
    configs = [
        (SHIPPED,          dict(crop_to_bbox=False, alpha=eval_alpha)),
        ("no-crop, a=0.8", dict(crop_to_bbox=False, alpha=0.8)),
        ("crop, a=eval",   dict(crop_to_bbox=True,  alpha=eval_alpha)),
        ("crop, a=0.8",    dict(crop_to_bbox=True,  alpha=0.8)),
    ]
    results = {}
    for name, kw in configs:
        train_p = train_category_probs(
            vine_model, frame_hwc, mask, bbox, keywords, device, image_size, **kw)
        results[name] = _report(name, eval_p, train_p, keywords)

    max_abs, same_top1 = results[SHIPPED]
    ok = same_top1 and max_abs <= TOLERANCE
    print(f"\n  VERDICT: shipped train forward {'PASS' if ok else 'FAIL'} "
          f"(top1 {'match' if same_top1 else 'DIFFER'}, max|Δp|={max_abs:.3f} vs tol {TOLERANCE}) "
          f"— train {'≈' if ok else '≠'} eval")
    return ok


def main():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "LASER"))
    from run_attack import load_vine

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vine_model = load_vine(device)
    image_size = vine_model.clip_processor.image_processor.crop_size["height"] \
        if hasattr(vine_model.clip_processor, "image_processor") else 224
    eval_alpha = vine_model.config.alpha
    print(f"\nCLIP crop size = {image_size}, eval extract alpha = {eval_alpha}\n")

    # Self-contained probe: a synthetic elliptical object mask on a deterministic
    # random frame. The random frame makes channel-order / scale differences move
    # the CLIP output; the parity check only cares that eval and train AGREE on the
    # same input, so no dataset is coupled in.
    H, W = 300, 400
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    cy0, cx0, ry, rx = H * 0.5, W * 0.5, H * 0.28, W * 0.20
    m = (((yy - cy0) / ry) ** 2 + ((xx - cx0) / rx) ** 2) <= 1.0          # [H,W] bool

    g = torch.Generator().manual_seed(0)
    frame = (torch.rand(H, W, 3, generator=g) * 255.0)
    ys, xs = m.nonzero(as_tuple=True)
    bbox = {'x1': int(xs.min()), 'y1': int(ys.min()),
            'x2': int(xs.max()) + 1, 'y2': int(ys.max()) + 1}

    # Any fixed vocab works — the frame is noise and we test eval-vs-train AGREEMENT,
    # not correctness. Use a few AI2-THOR/sim categories for relevance.
    keywords = ["Microwave", "Fridge", "Toaster", "Mug", "Apple", "CoffeeMachine"]
    print(f"synthetic ellipse  obj bbox={bbox}  frame={H}x{W}")
    ok = run(vine_model, frame, m, bbox, keywords, device, image_size, eval_alpha)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
