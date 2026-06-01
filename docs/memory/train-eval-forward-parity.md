---
name: train-eval-forward-parity
description: "The differentiable training forward must mirror VINE.predict() (eval) exactly, or patches don't transfer"
metadata: 
  node_type: memory
  type: project
  originSessionId: a0cb2bdd-386f-4320-af2c-8f57133844da
---

Eval is the source of truth: flip rate / scene-graph metrics come from `VineModel.predict()` (numpy + cv2 + CLIPProcessor, no-grad). Training optimizes a differentiable torch reimplementation of that forward, so the two MUST agree op-for-op or training loss falls while flip rate doesn't follow.

The single shared train-side forward is `functions/attack/preprocess.py::clip_scores` (used by `train.py` and both `anchor.py` warmups — keep it the only one). To match eval it must:
- **category/attribute**: classify on the **FULL frame** (background whitened), NO bbox crop. **relation**: crop to the union of subject+object bboxes.
- CLIP preprocessing (`clip_preprocess_torch`): **BGR** channel order (eval's `cv2.cvtColor(BGR2RGB)` on an RGB frame at vine_model.py:594 is a real swap), **resize-shortest-edge + center-crop** (NOT squash-to-square), bicubic + antialias.
- `alpha`/`white_alpha` from `vine_model.config` (0.5 / 0.8), not the torch function defaults (0.8).
- tokenize with `padding='max_length', max_length=75`.

Guardrail: `python -m functions.eval.consistency_check`. Healthy = image-feature cosine ≈ 1.0 (mean ~0.998), per-keyword max|Δp| ~0.001, 0 top-1 mismatches. The fix took cosine 0.90 → 1.0. Residual ~0.002 is uint8 (eval) vs float (train) quantization — do NOT round to uint8, it kills gradients.

**Why:** old training cropped to bbox + squashed + RGB + alpha 0.8, so it optimized ~0.90-similar embeddings at a different object scale than eval saw → weak/erratic flip rate.
**How to apply:** any change to the attack's CLIP forward, anchor warmup, or preprocessing must be re-checked with consistency_check before trusting eval numbers. Complements the mask/frame bbox-alignment fix that makes the patch land on the object — see [[bbox-mask-frame-alignment]].

Known still-unmatched (lower priority): `reference_gram.py` style target still crops+alpha0.8 (not a classifier input); relation eval skips non-overlapping bbox pairs (vine_model.py:493) while training scores all pairs.
