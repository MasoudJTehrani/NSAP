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

def _clip_scores(
    vine_model,
    clip_encoder,
    crop_hwc: torch.Tensor,
    keywords: List[str],
    device: torch.device,
    clip_image_size: int,
) -> torch.Tensor:
    """
    Run the differentiable CLIP forward pass on a single cropped image.
    Returns a [N_keywords] score tensor (cosine similarity × logit_scale).
    Gradients flow back through crop_hwc.
    """
    from functions.attack.preprocess import clip_preprocess_torch, clip_image_features, clip_text_features

    pv        = clip_preprocess_torch(crop_hwc, clip_image_size)   # [1,3,H,W]
    img_feat  = clip_image_features(clip_encoder, pv)              # [1, D]

    text_tokens = vine_model.clip_tokenizer(
        keywords, return_tensors='pt', padding=True, truncation=True, max_length=75
    ).to(device)
    text_feat   = clip_text_features(clip_encoder, **text_tokens)  # [N, D]

    img_n  = img_feat  / img_feat.norm(p=2, dim=-1, keepdim=True)
    text_n = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
    logit_scale = clip_encoder.logit_scale.exp() if hasattr(clip_encoder, 'logit_scale') else 1.0
    return (img_n @ text_n.T).squeeze(0) * logit_scale            # [N]


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
    with torch.no_grad():
        if is_pair:
            proc = extract_object_subject_torch(frame.to(device), mask.to(device), obj_mask.to(device))
            crop = crop_image_contain_bboxes_torch(proc, [bbox, obj_bbox])
        else:
            proc = extract_single_object_torch(frame.to(device), mask.to(device))
            crop = crop_image_contain_bboxes_torch(proc, [bbox])
        scores = _clip_scores(vine_model, clip_encoder, crop, keywords, device, clip_image_size)
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
    from functions.attack.anchor import select_anchor, select_relation_anchor
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
    print(f"Found {len(instances)} instances.")

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

    # ── Identify label indices for attack and preservation ───────────────────
    sample_spec = instances[0]['gpt_spec']
    cat_kws    = sample_spec.get('consts',       [])
    unary_kws  = sample_spec.get('unary_kws',    [])
    binary_kws = sample_spec.get('binary_kws',   [])

    if config.attack_mode == 'category':
        correct_idx = next(
            (i for i, c in enumerate(cat_kws) if config.target_category.lower() in c.lower()), 0
        )
    elif config.attack_mode == 'attribute':
        correct_idx = next(
            (i for i, c in enumerate(unary_kws) if config.target_attribute.lower() in c.lower()), 0
        )
    else:
        correct_idx = next(
            (i for i, k in enumerate(binary_kws) if config.target_relation.lower() in k.lower()), 0
        )

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
            if config.attack_mode == 'relation':
                frame     = inst['frame'].to(device)
                subj_mask = inst['subj_mask'].to(device)
                obj_mask  = inst['obj_mask'].to(device)
                subj_bbox = inst['subj_bbox']
                obj_bbox  = inst['obj_bbox']

                ph, pw    = compute_patch_size(subj_bbox, config.coverage_ratio)
                patched   = paste_patch(
                    frame, patch.pixels_hwc,
                    anchor.cx, anchor.cy, ph, pw,
                    offset_range=config.offset_range,
                )

                proc  = extract_object_subject_torch(patched, subj_mask, obj_mask)
                crop  = crop_image_contain_bboxes_torch(proc, [subj_bbox, obj_bbox])
                scores = _clip_scores(
                    vine_model, vine_model.clip_binary_model,
                    crop, binary_kws, device, config.clip_image_size,
                )
                probs_attacked = torch.softmax(scores, dim=-1)
                step_attack = step_attack + l_attack_cw(scores, correct_idx, config.kappa)

                # Preservation: category and attribute heads
                for (enc, kws, is_pair) in [
                    (vine_model.clip_cate_model,   cat_kws,   False),
                    (vine_model.clip_unary_model,  unary_kws, False),
                ]:
                    if not kws:
                        continue
                    proc_p = extract_single_object_torch(patched, subj_mask)
                    crop_p = crop_image_contain_bboxes_torch(proc_p, [subj_bbox])
                    s_pat  = _clip_scores(vine_model, enc, crop_p, kws, device, config.clip_image_size)
                    p_pat  = torch.softmax(s_pat, dim=-1)
                    p_orig = _get_clean_probs(vine_model, enc, inst['frame'], inst['subj_mask'],
                                              subj_bbox, kws, device, config.clip_image_size)
                    step_preserve = step_preserve + l_preserve_kl(p_orig, p_pat)

            else:
                frame = inst['frame'].to(device)
                mask  = inst['mask'].to(device)
                bbox  = inst['bbox']

                ph, pw  = compute_patch_size(bbox, config.coverage_ratio)
                patched = paste_patch(
                    frame, patch.pixels_hwc,
                    anchor.cx, anchor.cy, ph, pw,
                    offset_range=config.offset_range,
                )

                proc  = extract_single_object_torch(patched, mask)
                crop  = crop_image_contain_bboxes_torch(proc, [bbox])

                enc = (vine_model.clip_cate_model if config.attack_mode == 'category'
                       else vine_model.clip_unary_model)
                kws = cat_kws if config.attack_mode == 'category' else unary_kws

                scores = _clip_scores(vine_model, enc, crop, kws, device, config.clip_image_size)
                step_attack = step_attack + l_attack_cw(scores, correct_idx, config.kappa)

                # Preservation: the other two heads
                preserve_heads = [
                    (vine_model.clip_cate_model,  cat_kws)    if config.attack_mode == 'attribute' else
                    (vine_model.clip_unary_model, unary_kws),
                    (vine_model.clip_binary_model, binary_kws),
                ]
                for (penc, pkws) in preserve_heads:
                    if not pkws:
                        continue
                    s_pat  = _clip_scores(vine_model, penc, crop, pkws, device, config.clip_image_size)
                    p_pat  = torch.softmax(s_pat, dim=-1)
                    p_orig = _get_clean_probs(vine_model, penc, inst['frame'], inst['mask'],
                                              bbox, pkws, device, config.clip_image_size)
                    step_preserve = step_preserve + l_preserve_kl(p_orig, p_pat)

            # Chameleon losses (computed on the patch directly — not on the crop)
            patch_px = patch.pixels_hwc  # [H, W, 3]
            step_gram   = step_gram   + l_chameleon_gram(patch_px, reference_grams, vgg, vgg_mean, vgg_std)
            step_tv     = step_tv     + l_chameleon_tv(patch_px)
            step_dino_l = step_dino_l + l_chameleon_dino(
                patched, _patch_bbox_from_anchor(anchor, ph, pw, patched.shape),
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
                f"{config.attack_mode}_{config.target_category}_step{step+1}",
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
        f"{config.attack_mode}_{config.target_category}_final",
    )
    torch.save({'w': patch.w.data, 'anchor': anchor, 'config': config}, final_base + '.pt')
    _save_patch_png(patch, final_base)
    print(f"\nFinal patch saved: {final_base}.pt + .png")
    return patch


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


def _patch_bbox_from_anchor(anchor, patch_h: int, patch_w: int, frame_shape) -> dict:
    """Compute the actual bbox occupied by the patch given anchor and frame shape."""
    H, W = frame_shape[:2]
    x1 = max(0, min(anchor.cx - patch_w // 2, W - 1))
    y1 = max(0, min(anchor.cy - patch_h // 2, H - 1))
    x2 = max(x1 + 1, min(x1 + patch_w, W))
    y2 = max(y1 + 1, min(y1 + patch_h, H))
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
