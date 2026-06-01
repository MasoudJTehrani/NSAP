"""
Main adversarial patch training loop.

Call train_patch() to run the full pipeline for one attack mode and one target category.
Phase 0 : SAM anchor selection + DINO diversity clustering (run once)
Precompute: reference Gram matrices
Training  : Adam + EoT over diversity-sampled instances
"""

import os
import math
import random
import torch
import torch.optim as optim
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AttackConfig:
    attack_mode: str            # 'category' | 'attribute' | 'relation'
    target_category: str        # e.g. 'person'
    target_relation: str = ''   # e.g. 'next to'  (relation attack)
    target_attribute: str = ''  # e.g. 'standing' (attribute attack — which attribute to flip)

    # Patch
    coverage_ratio: float = 0.3
    offset_range: float   = 0.1

    # Optimizer
    learning_rate: float = 0.01
    num_steps: int       = 2000
    eot_batch_size: int  = 8

    # Loss weights
    lambda_c:    float = 1.0
    lambda_tv:   float = 1e-4
    lambda_dino: float = 0.5
    lambda_p:    float = 0.5
    kappa:       float = 0.0

    # Reference Gram
    ref_gram_n: int = 500

    # SAM anchor selection
    sam_anchor_warmup: int       = 20
    sam_anchor_n_instances: int  = 100

    # DINO diversity clustering
    dino_cluster_k: int = 20
    dino_model_name: str = "facebook/dinov2-base"

    # CLIP image size (224 for ViT-B/32; 336 for ViT-L/14-336)
    clip_image_size: int = 224

    # Checkpointing
    save_every: int          = 200
    output_dir: str          = "output/patches"
    ref_gram_cache: str      = "output/reference_grams"

    # Early stopping
    early_stop_flip_rate: float = 0.9   # stop if flip rate exceeds this on val set
    val_every: int              = 100


# ──────────────────────────────────────────────────────────────────────────────
# DINO-based diversity clustering for EoT sampling
# ──────────────────────────────────────────────────────────────────────────────

def _build_dino_clusters(
    instances: list,
    dino_model,
    dino_preprocess_fn,
    k: int,
    device: torch.device,
) -> List[int]:
    """
    Assign each instance a cluster label using k-means on DINOv2 CLS features.
    Returns a list of integer cluster labels, one per instance.
    """
    from sklearn.cluster import MiniBatchKMeans

    print(f"Building DINO diversity clusters (k={k}) over {len(instances)} instances...")
    cls_feats = []

    dino_model.eval()
    with torch.no_grad():
        for inst in instances:
            frame = inst['frame'].to(device)
            pv    = dino_preprocess_fn(frame)
            out   = dino_model(pixel_values=pv)
            cls   = out.last_hidden_state[:, 0, :].cpu().float()
            cls_feats.append(cls.squeeze(0).numpy())

    cls_array = np.stack(cls_feats)
    km        = MiniBatchKMeans(n_clusters=min(k, len(instances)), random_state=42)
    labels    = km.fit_predict(cls_array).tolist()
    print(f"  Clustering done. Cluster sizes: {np.bincount(labels).tolist()}")
    return labels


def _diversity_sample(
    instances: list,
    cluster_labels: List[int],
    batch_size: int,
    k: int,
) -> list:
    """
    Sample batch_size instances with maximum cluster diversity.
    Cycles through clusters round-robin until the batch is full.
    """
    clusters: dict = {}
    for idx, label in enumerate(cluster_labels):
        clusters.setdefault(label, []).append(idx)

    for c in clusters:
        random.shuffle(clusters[c])

    cluster_ids = list(clusters.keys())
    random.shuffle(cluster_ids)

    sampled_indices = []
    pool = {c: list(clusters[c]) for c in cluster_ids}
    round_robin = cluster_ids.copy()

    while len(sampled_indices) < batch_size:
        if not round_robin:
            round_robin = [c for c in cluster_ids if pool[c]]
            if not round_robin:
                break
        cid = round_robin.pop(0)
        if pool[cid]:
            sampled_indices.append(pool[cid].pop())

    return [instances[i] for i in sampled_indices]


# ──────────────────────────────────────────────────────────────────────────────
# Differentiable VINE forward pass (bypasses numpy preprocessing)
# ──────────────────────────────────────────────────────────────────────────────

# The differentiable train-side forward lives in preprocess.clip_scores so that
# train.py and the anchor warmups share ONE implementation and cannot drift from
# eval. Imported under the existing private name to keep all call sites unchanged.
from functions.attack.preprocess import clip_scores as _clip_scores


def _get_clean_probs(
    vine_model,
    clip_encoder,
    frame,
    mask,
    bbox,
    keywords: List[str],
    device: torch.device,
    clip_image_size: int,
    is_pair: bool = False,
    obj_mask=None,
    obj_bbox=None,
) -> torch.Tensor:
    """
    Run the clean (un-patched) forward pass to obtain reference probabilities
    for the KL preservation loss. No gradients are needed here.
    """
    from functions.attack.preprocess import (
        extract_single_object_torch,
        extract_object_subject_torch,
        crop_image_contain_bboxes_torch,
    )
    # alpha/white_alpha come from the VINE config so they match eval exactly
    alpha       = vine_model.config.alpha
    white_alpha = vine_model.config.white_alpha
    with torch.no_grad():
        if is_pair:
            # relation head: eval crops to the union of subject+object bboxes
            proc    = extract_object_subject_torch(frame.to(device), mask.to(device), obj_mask.to(device),
                                                   alpha=alpha, white_alpha=white_alpha)
            clip_in = crop_image_contain_bboxes_torch(proc, [bbox, obj_bbox])
        else:
            # category/attribute head: eval uses the FULL frame, not a bbox crop
            clip_in = extract_single_object_torch(frame.to(device), mask.to(device), alpha=alpha)
        scores = _clip_scores(vine_model, clip_encoder, clip_in, keywords, device, clip_image_size)
        return torch.softmax(scores, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_patch(
    vine_model,
    dataset_dir: str,
    dataset_filename: str,
    config: AttackConfig,
    sam_predictor=None,
) -> 'AdversarialPatch':
    """
    Full training pipeline. Returns the trained AdversarialPatch.

    Args:
        vine_model:       loaded VineModel (weights frozen — we only train the patch)
        dataset_dir:      ESCA dataset root directory
        dataset_filename: ESCA annotation filename
        config:           AttackConfig dataclass
        sam_predictor:    SAM2ImagePredictor instance (None = use bbox center as anchor)
    """
    from functions.attack.data import EscaAttackDataset
    from functions.attack.patch import AdversarialPatch, paste_patch, compute_patch_size
    from functions.attack.anchor import select_anchor, select_relation_anchor, anchor_center
    from functions.attack.reference_gram import (
        compute_reference_grams, build_vgg_normalisation_tensors
    )
    from functions.attack.losses import (
        l_attack_cw, l_chameleon_gram, l_chameleon_tv,
        l_chameleon_dino, l_preserve_kl, total_loss,
    )
    from functions.attack.preprocess import (
        extract_single_object_torch,
        extract_object_subject_torch,
        crop_image_contain_bboxes_torch,
        dino_preprocess_torch,
    )
    from transformers import AutoModel
    import torchvision.models as tv_models

    device = next(vine_model.parameters()).device

    # Freeze the VINE model — only the patch is trainable
    for p in vine_model.parameters():
        p.requires_grad_(False)
    vine_model.eval()

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\n=== Loading dataset for {config.attack_mode} attack on '{config.target_category}' ===")
    dataset = EscaAttackDataset(
        dataset_dir=dataset_dir,
        dataset_filename=dataset_filename,
        target_category=config.target_category if config.attack_mode != 'relation' else None,
        target_attribute=config.target_attribute if config.attack_mode == 'attribute' else None,
        split='train',
    )

    if config.attack_mode == 'relation':
        instances = dataset.build_relation_instance_list(
            target_relation=config.target_relation,
        )
    else:
        instances = dataset.build_instance_list(
            target_category=config.target_category,
            target_predicate=config.target_attribute if config.attack_mode == 'attribute' else None,
        )

    if len(instances) == 0:
        raise RuntimeError(f"No instances found for '{config.target_category}'.")
    print(f"Found {len(instances)} instances. Filtering by clean VINE prediction...")
    instances = _filter_by_clean_prediction(instances, vine_model, config, device)
    if len(instances) == 0:
        raise RuntimeError(
            f"No instances remain after clean-prediction filter — VINE does not predict "
            f"'{config.target_attribute or config.target_category or config.target_relation}' "
            f"on any training frame."
        )
    print(f"  → {len(instances)} instances kept.")

    # ── Phase 0a: SAM anchor selection ───────────────────────────────────────
    print("\n=== Phase 0a: SAM anchor selection ===")
    if config.attack_mode == 'relation':
        anchor = select_relation_anchor(
            relation_instances=instances,
            vine_model=vine_model,
            target_relation=config.target_relation,
            coverage_ratio=config.coverage_ratio,
            warmup_steps=config.sam_anchor_warmup,
            device=device,
            sam_predictor=sam_predictor,
            clip_image_size=config.clip_image_size,
        )
    else:
        anchor = select_anchor(
            instances=instances,
            vine_model=vine_model,
            attack_mode=config.attack_mode,
            target_category=config.target_category,
            coverage_ratio=config.coverage_ratio,
            warmup_steps=config.sam_anchor_warmup,
            device=device,
            sam_predictor=sam_predictor,
            clip_image_size=config.clip_image_size,
            target_attribute=config.target_attribute,
        )

    # ── Phase 0b: DINO diversity clustering ──────────────────────────────────
    print("\n=== Phase 0b: DINO diversity clustering ===")
    dino_model = AutoModel.from_pretrained(config.dino_model_name)
    dino_model.eval().to(device)
    for p in dino_model.parameters():
        p.requires_grad_(False)

    cluster_labels = _build_dino_clusters(
        instances=instances,
        dino_model=dino_model,
        dino_preprocess_fn=dino_preprocess_torch,
        k=config.dino_cluster_k,
        device=device,
    )

    # ── Precompute reference Gram matrices ────────────────────────────────────
    print("\n=== Precomputing reference Gram matrices ===")
    gram_cache = os.path.join(
        config.ref_gram_cache,
        f"{config.attack_mode}_{config.target_category}_gram.pt",
    )
    reference_grams = compute_reference_grams(
        instance_list=instances if config.attack_mode != 'relation' else
                      [{'frame': i['frame'], 'mask': i['subj_mask'], 'bbox': i['subj_bbox']} for i in instances],
        n_instances=config.ref_gram_n,
        device=device,
        cache_path=gram_cache,
    )

    vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).eval().to(device)
    for p in vgg.parameters():
        p.requires_grad_(False)
    vgg_mean, vgg_std = build_vgg_normalisation_tensors(device)

    # ── Initialise patch and optimizer ───────────────────────────────────────
    sample_bbox = (
        instances[0]['bbox'] if config.attack_mode != 'relation'
        else instances[0]['subj_bbox']
    )
    patch_h, patch_w = compute_patch_size(sample_bbox, config.coverage_ratio)
    patch     = AdversarialPatch(patch_h, patch_w, device)
    optimizer = optim.Adam(patch.parameters(), lr=config.learning_rate)

    os.makedirs(config.output_dir, exist_ok=True)

    # ── Main training loop ────────────────────────────────────────────────────
    print(f"\n=== Training patch ({config.num_steps} steps) ===")

    for step in range(config.num_steps):
        optimizer.zero_grad()

        # Diversity-sampled mini-batch
        batch = _diversity_sample(instances, cluster_labels, config.eot_batch_size, config.dino_cluster_k)

        step_attack   = torch.tensor(0.0, device=device)
        step_gram     = torch.tensor(0.0, device=device)
        step_tv       = torch.tensor(0.0, device=device)
        step_dino_l   = torch.tensor(0.0, device=device)
        step_preserve = torch.tensor(0.0, device=device)

        for inst in batch:
            # Per-instance vocabulary: each clip has its own keyword list
            inst_spec       = inst['gpt_spec']
            inst_cat_kws    = inst_spec.get('consts',     [])
            inst_unary_kws  = inst_spec.get('unary_kws',  [])
            inst_binary_kws = inst_spec.get('binary_kws', [])

            if config.attack_mode == 'relation':
                frame     = inst['frame'].to(device)
                subj_mask = inst['subj_mask'].to(device)
                obj_mask  = inst['obj_mask'].to(device)
                subj_bbox = inst['subj_bbox']
                obj_bbox  = inst['obj_bbox']

                binary_kws  = inst_binary_kws
                correct_idx = next(
                    (i for i, k in enumerate(binary_kws)
                     if config.target_relation.lower() in k.lower()), None
                )
                if correct_idx is None:
                    continue

                ph, pw    = compute_patch_size(subj_bbox, config.coverage_ratio)
                cx, cy    = anchor_center(anchor, subj_bbox)
                patched   = paste_patch(
                    frame, patch.pixels_hwc,
                    cx, cy, ph, pw,
                    offset_range=config.offset_range,
                )

                # Relation head: eval extracts subject+object (red/blue channel boost)
                # then crops to the union of both bboxes. Mirror that exactly.
                proc  = extract_object_subject_torch(
                    patched, subj_mask, obj_mask,
                    alpha=vine_model.config.alpha, white_alpha=vine_model.config.white_alpha,
                )
                crop  = crop_image_contain_bboxes_torch(proc, [subj_bbox, obj_bbox])
                scores = _clip_scores(
                    vine_model, vine_model.clip_binary_model,
                    crop, binary_kws, device, config.clip_image_size,
                )
                step_attack = step_attack + l_attack_cw(scores, correct_idx, config.kappa)

                # Preservation: category and attribute heads of the subject — these
                # are classified on the FULL frame in eval (no crop).
                for enc, kws in [
                    (vine_model.clip_cate_model,  inst_cat_kws),
                    (vine_model.clip_unary_model, inst_unary_kws),
                ]:
                    if not kws:
                        continue
                    proc_p = extract_single_object_torch(patched, subj_mask, alpha=vine_model.config.alpha)
                    s_pat  = _clip_scores(vine_model, enc, proc_p, kws, device, config.clip_image_size)
                    p_pat  = torch.softmax(s_pat, dim=-1)
                    p_orig = _get_clean_probs(vine_model, enc, inst['frame'], inst['subj_mask'],
                                              subj_bbox, kws, device, config.clip_image_size)
                    step_preserve = step_preserve + l_preserve_kl(p_orig, p_pat)

            else:
                frame = inst['frame'].to(device)
                mask  = inst['mask'].to(device)
                bbox  = inst['bbox']

                if config.attack_mode == 'category':
                    enc         = vine_model.clip_cate_model
                    kws         = inst_cat_kws
                    target_str  = config.target_category
                else:
                    enc         = vine_model.clip_unary_model
                    kws         = inst_unary_kws
                    target_str  = config.target_attribute

                correct_idx = next(
                    (i for i, c in enumerate(kws) if target_str.lower() in c.lower()), None
                )
                if correct_idx is None:
                    continue

                ph, pw  = compute_patch_size(bbox, config.coverage_ratio)
                cx, cy  = anchor_center(anchor, bbox)
                patched = paste_patch(
                    frame, patch.pixels_hwc,
                    cx, cy, ph, pw,
                    offset_range=config.offset_range,
                )

                # Category/attribute heads: eval classifies on the FULL frame with
                # background whitened (no bbox crop). Mirror that here so the patch
                # is optimised against the same input it is scored on.
                proc = extract_single_object_torch(patched, mask, alpha=vine_model.config.alpha)

                scores = _clip_scores(vine_model, enc, proc, kws, device, config.clip_image_size)
                step_attack = step_attack + l_attack_cw(scores, correct_idx, config.kappa)

                # Preservation: the other two heads (same full-frame input)
                preserve_heads = [
                    (vine_model.clip_cate_model,   inst_cat_kws)   if config.attack_mode == 'attribute'
                    else (vine_model.clip_unary_model, inst_unary_kws),
                    (vine_model.clip_binary_model, inst_binary_kws),
                ]
                for penc, pkws in preserve_heads:
                    if not pkws:
                        continue
                    s_pat  = _clip_scores(vine_model, penc, proc, pkws, device, config.clip_image_size)
                    p_pat  = torch.softmax(s_pat, dim=-1)
                    p_orig = _get_clean_probs(vine_model, penc, inst['frame'], inst['mask'],
                                              bbox, pkws, device, config.clip_image_size)
                    step_preserve = step_preserve + l_preserve_kl(p_orig, p_pat)

            # Chameleon losses (computed on the patch directly — not on the crop)
            patch_px = patch.pixels_hwc  # [H, W, 3]
            step_gram   = step_gram   + l_chameleon_gram(patch_px, reference_grams, vgg, vgg_mean, vgg_std)
            step_tv     = step_tv     + l_chameleon_tv(patch_px)
            step_dino_l = step_dino_l + l_chameleon_dino(
                patched, _patch_bbox_from_anchor(cx, cy, ph, pw, patched.shape),
                dino_model, dino_preprocess_torch,
            )

        n = len(batch)
        loss = total_loss(
            attack_loss   = step_attack   / n,
            gram_loss     = step_gram     / n,
            tv_loss       = step_tv       / n,
            dino_loss     = step_dino_l   / n,
            preserve_loss = step_preserve / n,
            lambda_c      = config.lambda_c,
            lambda_tv     = config.lambda_tv,
            lambda_dino   = config.lambda_dino,
            lambda_p      = config.lambda_p,
        )

        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(
                f"step {step+1:4d}/{config.num_steps} | "
                f"loss={loss.item():.4f} | "
                f"attack={step_attack.item()/n:.4f} | "
                f"gram={step_gram.item()/n:.4f} | "
                f"dino={step_dino_l.item()/n:.4f} | "
                f"preserve={step_preserve.item()/n:.4f}"
            )

        if (step + 1) % config.save_every == 0:
            ckpt_base = os.path.join(
                config.output_dir,
                f"{_attack_slug(config)}_step{step+1}",
            )
            torch.save({
                'step':   step + 1,
                'w':      patch.w.data,
                'anchor': anchor,
                'config': config,
            }, ckpt_base + '.pt')
            _save_patch_png(patch, ckpt_base)
            print(f"Checkpoint saved: {ckpt_base}.pt + .png")

    final_base = os.path.join(
        config.output_dir,
        f"{_attack_slug(config)}_final",
    )
    torch.save({'w': patch.w.data, 'anchor': anchor, 'config': config}, final_base + '.pt')
    _save_patch_png(patch, final_base)
    print(f"\nFinal patch saved: {final_base}.pt + .png")
    return patch


def _bbox_center(bbox: dict):
    return (bbox['x1'] + bbox['x2']) // 2, (bbox['y1'] + bbox['y2']) // 2


def _filter_by_clean_prediction(
    instances: list,
    vine_model,
    config: 'AttackConfig',
    device: torch.device,
) -> list:
    """
    Keep only instances where VINE's top-1 clean prediction matches the target.

    - category attack: predicted category contains target_category
    - attribute attack: predicted attribute contains target_attribute
    - relation attack: predicted relation contains target_relation

    This ensures the attack has something to flip — training on instances
    where the model already predicts the wrong label is wasted signal.
    """
    from functions.attack.preprocess import (
        extract_single_object_torch,
        extract_object_subject_torch,
        crop_image_contain_bboxes_torch,
    )

    kept = []
    for inst in instances:
        frame = inst['frame'].to(device)
        spec  = inst['gpt_spec']

        try:
            if config.attack_mode in ('category', 'attribute'):
                mask = inst['mask'].to(device)

                # Must mirror eval: category/attribute classify on the full frame
                proc = extract_single_object_torch(frame, mask, alpha=vine_model.config.alpha)

                if config.attack_mode == 'category':
                    enc    = vine_model.clip_cate_model
                    kws    = spec.get('consts', [])
                    target = config.target_category
                else:
                    enc    = vine_model.clip_unary_model
                    kws    = spec.get('unary_kws', [])
                    target = config.target_attribute

                if not kws:
                    continue

                with torch.no_grad():
                    scores  = _clip_scores(vine_model, enc, proc, kws, device, config.clip_image_size)
                    top1_lbl = kws[scores.argmax().item()]

            else:  # relation
                subj_mask = inst['subj_mask'].to(device)
                obj_mask  = inst['obj_mask'].to(device)
                subj_bbox = inst['subj_bbox']
                obj_bbox  = inst['obj_bbox']
                kws       = spec.get('binary_kws', [])
                target    = config.target_relation

                if not kws:
                    continue

                proc = extract_object_subject_torch(
                    frame, subj_mask, obj_mask,
                    alpha=vine_model.config.alpha, white_alpha=vine_model.config.white_alpha,
                )
                crop = crop_image_contain_bboxes_torch(proc, [subj_bbox, obj_bbox])

                with torch.no_grad():
                    scores  = _clip_scores(vine_model, vine_model.clip_binary_model,
                                           crop, kws, device, config.clip_image_size)
                    top1_lbl = kws[scores.argmax().item()]

            if target.lower() in top1_lbl.lower():
                kept.append(inst)

        except Exception:
            continue

    return kept


def _attack_slug(config: AttackConfig) -> str:
    """Build a filename slug that matches the output subdirectory name."""
    if config.attack_mode == 'relation' and config.target_relation:
        return f"relation_{config.target_relation.replace(' ', '_')}_{config.target_category.replace(' ', '_')}"
    if config.attack_mode == 'attribute' and config.target_attribute:
        return f"attribute_{config.target_category.replace(' ', '_')}_{config.target_attribute.replace(' ', '_')}"
    return f"{config.attack_mode}_{config.target_category.replace(' ', '_')}"


def _save_patch_png(patch, base_path: str, upscale: int = 8) -> None:
    """
    Save the adversarial patch as a lossless PNG image.

    PNG is chosen over JPEG because adversarial patches contain subtle pixel-level
    perturbations that JPEG compression would destroy.

    Two files are written:
      <base_path>.png         — native patch size (exact pixel values)
      <base_path>_preview.png — upscaled by `upscale`× for easy visual inspection

    Args:
        patch:     AdversarialPatch
        base_path: path without extension (e.g. "output/patches/category_person_final")
        upscale:   integer scale factor for the preview image (default 8)
    """
    from PIL import Image
    import numpy as np

    pixels_np = patch.pixels_hwc.detach().cpu().numpy()
    pixels_u8 = pixels_np.clip(0, 255).astype(np.uint8)   # [H, W, 3] uint8

    # Native size — exact adversarial pixel values
    img_native = Image.fromarray(pixels_u8, mode='RGB')
    img_native.save(base_path + '.png', format='PNG')

    # Upscaled preview (nearest-neighbour to preserve block structure)
    ph, pw = pixels_u8.shape[:2]
    img_preview = img_native.resize((pw * upscale, ph * upscale), Image.NEAREST)
    img_preview.save(base_path + '_preview.png', format='PNG')


def _patch_bbox_from_anchor(cx: int, cy: int, patch_h: int, patch_w: int, frame_shape) -> dict:
    """Compute the actual bbox occupied by the patch given absolute center and frame shape."""
    H, W = frame_shape[:2]
    x1 = max(0, min(cx - patch_w // 2, W - 1))
    y1 = max(0, min(cy - patch_h // 2, H - 1))
    x2 = max(x1 + 1, min(x1 + patch_w, W))
    y2 = max(y1 + 1, min(y1 + patch_h, H))
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
