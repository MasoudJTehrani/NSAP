# NSAP — Session Handoff / Continuation Brief

Written 2026-06-01 to migrate work to another machine. NSAP = Neuro-Symbolic
Adversarial Patch: a universal adversarial patch attack against the **VINE**
CLIP-based scene-graph model. This brief lets you (or a fresh Claude Code
session) pick up exactly where we left off.

---

## 0. Resume checklist (do this first on the new PC)

1. **Clone the repo** (code is already pushed to `origin/main`, through commit
   `8c16ff7` "after changing training based on eval").
2. **Recreate the environment**: `conda env create -f environment.yml`
   (torch 2.10 / cu128, transformers 5.1, CUDA). Commit `environment.yml` if it
   isn't tracked yet — it's the migration enabler.
3. **Restore the Claude memory** (kept in the repo at `docs/memory/`, but Claude
   Code reads it from `~/.claude/`):
   ```bash
   # slug = clone path with non-alphanumerics → '-'. /home/vortex/NSAP → -home-vortex-NSAP
   mkdir -p ~/.claude/projects/-home-vortex-NSAP/memory
   cp docs/memory/*.md ~/.claude/projects/-home-vortex-NSAP/memory/
   ```
   If you clone to a different path/user, adjust the slug accordingly.
4. **Re-fetch the gitignored large assets** (NOT in git — see §4): VINE weights,
   CLIP cache, dataset masks, LASER source. See `how_to_run.txt`.
5. **Sanity check** the install:
   ```bash
   python -m functions.eval.consistency_check     # expect: no-crop a=eval → max|Δp|≈0.001, top1 MATCH
   python run_attack.py --validate-bboxes --target person --n-clips 5 --keep-videos
   ```

---

## 1. What this session accomplished (all in commit `8c16ff7`)

Two root causes of "patch off-target / weak flip rate" were found and fixed.

### A. Bbox ↔ frame misalignment  → `functions/attack/data.py`
`load_video` sub-sampled frames to `max_vid_len` but the masks were **not**
sub-sampled/re-indexed in lockstep, so `bboxes[k]` described a *different*
original frame than `frames[k]`. Fix in `__getitem__`: guard
`len(frames)==len(masks)`, then sub-sample frames AND masks with the same rate
and re-index mask frame-ids (`new_fid = fid // rate`). This fixes **both**
training and eval (both read boxes from the same loader).

### B. Train forward ≠ eval forward  → `preprocess.py`, `train.py`, `anchor.py`
Eval is the source of truth (`VineModel.predict`). Training optimised a
differentiable torch reimplementation that had **drifted** from it in 4 ways,
giving CLIP image-feature cosine of only **0.90**:

| Factor | was (train) | eval (truth) → fix |
|---|---|---|
| resize | squash→224² bilinear | resize-shortest-edge + center-crop, bicubic |
| channels | RGB | **BGR** (eval's `cv2.cvtColor` at vine_model.py:594 is a real swap) |
| crop (cat/attr) | crop to bbox | **full frame**, no crop |
| alpha | 0.8 (default) | 0.5 (`vine_model.config.alpha`) |

Fixes: rewrote `clip_preprocess_torch` (BGR + resize-short + center-crop);
added one shared forward **`preprocess.clip_scores`** used by `train.py` and
both `anchor.py` warmups (no more duplicated/divergent forwards); cat/attr now
classify on the full frame; alpha/white_alpha pulled from `vine_model.config`;
tokenisation padded to `max_length=75`. Result: cosine **1.0000** (mean 0.998).

### C. New tooling + docs
- `functions/eval/consistency_check.py` — train-vs-eval parity guardrail.
- `functions/eval/validate_bboxes.py` + `run_attack.py --validate-bboxes` —
  bbox derivation/alignment checks + overlay images.
- `functions/attack/losses.py::total_loss` — full loss definition in the docstring.

---

## 2. Verification status (measured this session)

- **Forward parity**: matched config `max|Δp| = 0.001`, image-feature cosine
  mean 0.998 / min 0.986, **0 top-1 mismatches across 21 objects**. Gradient
  still flows to the patch (norm ≈ 4.17) — fixes stay differentiable.
- **Bbox alignment**: old loader had 6 alignment failures on a sample clip
  (frame/mask count + out-of-range ids); fixed loader has 0.
- Residual ~0.002 prob is uint8 (eval) vs float (train) quantisation — left on
  purpose; rounding to uint8 would kill gradients.

---

## 3. Pending / next steps

1. **End-to-end run** once videos are downloaded (the real final check):
   ```bash
   python run_attack.py --mode category --target person --steps 50
   python run_attack.py --eval --patch output/patches/category_person/category_person_final.pt
   ```
2. **Known still-unmatched** (lower priority):
   - `functions/attack/reference_gram.py` (Chameleon style target) still crops +
     alpha 0.8 — not a classifier input, so not eval-critical; revisit when
     adding camouflage.
   - Relation head: eval skips non-overlapping `(subj,obj)` bbox pairs
     (vine_model.py:493); training scores all pairs. Category/attribute unaffected.
3. **Planned research direction** (user's): incremental **ablation** —
   center-of-bbox → learned anchor; no-Chameleon → Chameleon; category →
   attribute → relation. Possibly archive the old prototype (git tag/branch, not
   file copies) once the new baseline reaches parity on flip rate.

---

## 4. What is NOT in git (must transfer separately)

`.gitignore` excludes the heavyweight assets:
- `model/` — VINE weights (`model/vine/model.safetensors`, ~1.8 GB) + config.
- `dataset/` — `mini_LLaVA_10000_masks/` (`*_mask.pkl`), `videos/`, annotations
  (`gpt_specs2_id_LLaVA_all.jsonl`).
- `src/` — cloned LASER source (`src/LASER`).
- `output/`, `checkpoints/`, `*.safetensors`, `*.pkl`, `*.pt`.
- CLIP backbone cache: `~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32`.

Either copy these from the old PC or re-download per `how_to_run.txt`.
The conversation transcript (`~/.claude/projects/-home-vortex-NSAP/*.jsonl`,
~16 MB) is optional and not needed to continue — this brief + `docs/memory/`
carry the context.

---

## 5. Key invariant to preserve (also in `docs/memory/train-eval-forward-parity.md`)

**The differentiable training forward must mirror `VineModel.predict()` op-for-op.**
Any change to the attack's CLIP forward / anchor warmup / preprocessing must be
re-checked with `python -m functions.eval.consistency_check` before trusting any
eval number. Healthy = feature cosine ≈ 1.0, max|Δp| ~0.001, 0 top-1 mismatches.
