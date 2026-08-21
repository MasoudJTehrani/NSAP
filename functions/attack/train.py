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
    # Fix-B: expectation over GroundingDINO-like crops. >0 also trains the patch to
    # flip SGClip across jittered bbox crops (mimicking detector boxes), not just the
    # full-frame view -> transfers to ESCA's live cropped multi-box detections.
    eot_crops: int = 0
    eot_scale_jitter: float = 0.3   # crop scale in [1-j, 1+j]; centre shift up to j*bbox

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

    # Appearance/content camouflage (the genuine "hide it" mechanism):
    #   object_init    — initialise the patch FROM a real crop of the target object
    #                    (the object's appearance under the patch footprint) instead
    #                    of a loud warm-start / random noise, so it STARTS hidden.
    #   lambda_content — weight of the pixel-L2 anchor to that crop (stealth↔flip
    #                    knob; 0 = off). Built whenever object_init OR lambda_content>0.
    object_init:    bool  = False
    lambda_content: float = 0.0

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

    # Chameleon naturalness losses (gram/tv/dino). Disable to free VGG16 + DINOv2
    # from GPU on small-memory cards. DINOv2 is still used for diversity clustering.
    use_chameleon: bool = True

    # Cap clips loaded into RAM (0 = no cap). Each clip's per-instance masks cost
    # ~tens of MB; tens of thousands of them exhaust CPU RAM during dataset build.
    max_clips: int = 0

    # Minimum VINE softmax confidence for the target CATEGORY to accept an object
    # as a target (0 = off). Drops objects VINE only weakly calls "person" (e.g. a
    # table misclassified at 0.3), in both training and eval. Applied to the
    # category head (relation: the subject's category).
    min_target_conf: float = 0.0

    # Path to the SceneGraphBench (AI2-THOR) dataset to train/eval on — NSAP's
    # data source. Frames are PNGs and labels are sim ground truth
    # (see functions/attack/sim_data.py::SimAttackDataset). Required.
    sim_data_dir: str = ''

    # Two-phase chameleon experiment (sim only):
    #   use_all_data    — train on ALL scenarios (no train/test split).
    #   init_patch_path — warm-start from this patch (loads its weights + anchor,
    #                     skips SAM anchor selection). Phase 2 continues training it.
    #   only_scenario   — restrict train/eval to this single scenario id, so the
    #                     chameleon camouflage overfits one specific kitchen.
    use_all_data: bool = False
    init_patch_path: str = ''
    only_scenario: str = ''

    # Checkpointing
    save_every: int          = 200
    output_dir: str          = "output/patches"
    ref_gram_cache: str      = "output/reference_grams"

    # Reproducibility: seed for the Python/NumPy/Torch RNGs used downstream
    # (diversity mini-batch sampling + EoT positional/scale jitter), applied once
    # at the start of train_patch. Bit-exact GPU determinism would additionally
    # require torch.use_deterministic_algorithms(True).
    seed: int = 0


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
            frame = _load_frame(inst).to(device)
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
from functions.attack.data import load_frame as _load_frame


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
# Fix-B: Expectation-Over-Crops — mimic GroundingDINO detection boxes
# ──────────────────────────────────────────────────────────────────────────────
def _eot_crop_object(patched, mask, bbox, alpha, scale_jitter, whiten=True):
    """Crop the patched frame to a JITTERED version of bbox (mimicking a detector
    box: random scale + centre shift). Differentiable through the slice (grads flow
    to patch pixels inside the crop). Returns None if the crop is degenerate.

    whiten=True  (category): whiten background by the instance mask -> a cropped
                 object view, matching the category head's full-frame eval forward.
    whiten=False (attribute): return the RAW bbox crop with no whitening. The model's
                 unary forward is extract_single_object(bbox_mask)+crop_to_bbox, which
                 is exactly a raw bbox crop (outside-bbox pixels are cropped away), so
                 this is what the planner grounds attributes through."""
    import random
    from functions.attack.preprocess import extract_single_object_torch
    H, W = patched.shape[0], patched.shape[1]
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    s = 1.0 + random.uniform(-scale_jitter, scale_jitter)
    cx = (x1 + x2) / 2 + random.uniform(-scale_jitter, scale_jitter) * bw
    cy = (y1 + y2) / 2 + random.uniform(-scale_jitter, scale_jitter) * bh
    nx1 = int(max(0, cx - bw * s / 2)); ny1 = int(max(0, cy - bh * s / 2))
    nx2 = int(min(W, cx + bw * s / 2)); ny2 = int(min(H, cy + bh * s / 2))
    if nx2 - nx1 < 4 or ny2 - ny1 < 4:
        return None
    if not whiten:
        return patched[ny1:ny2, nx1:nx2, :]
    m = mask.squeeze(-1) if mask.dim() == 3 else mask
    return extract_single_object_torch(patched[ny1:ny2, nx1:nx2, :],
                                       m[ny1:ny2, nx1:nx2], alpha=alpha)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_patch(
    vine_model,
    config: AttackConfig,
    sam_predictor=None,
) -> 'AdversarialPatch':
    """
    Full training pipeline. Returns the trained AdversarialPatch.

    Args:
        vine_model:       loaded VineModel (weights frozen — we only train the patch)
        config:           AttackConfig dataclass (config.sim_data_dir names the dataset)
        sam_predictor:    SAM2ImagePredictor instance (None = use bbox center as anchor)
    """
    from functions.attack.patch import AdversarialPatch, paste_patch, compute_patch_size
    from functions.attack.anchor import select_anchor, select_relation_anchor, anchor_center
    from functions.attack.reference_gram import (
        compute_reference_grams, build_vgg_normalisation_tensors
    )
    from functions.attack.losses import (
        l_attack_cw, l_chameleon_gram, l_chameleon_tv, l_chameleon_content,
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

    # Reproducibility: seed every RNG the pipeline uses (diversity sampling, EoT
    # jitter) so identical args reproduce identical runs.
    _seed = int(getattr(config, 'seed', 0))
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)

    # Freeze the VINE model — only the patch is trainable
    for p in vine_model.parameters():
        p.requires_grad_(False)
    vine_model.eval()

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\n=== Loading dataset for {config.attack_mode} attack on '{config.target_category}' ===")
    if not getattr(config, 'sim_data_dir', ''):
        raise ValueError(
            "NSAP trains on the sim (SceneGraphBench) dataset only. "
            "Pass --sim-data <dataset dir> (sets config.sim_data_dir)."
        )
    from functions.attack.sim_data import SimAttackDataset
    _only = [config.only_scenario] if getattr(config, 'only_scenario', '') else None
    _split = None if (getattr(config, 'use_all_data', False) or _only) else 'train'
    dataset = SimAttackDataset(
        config.sim_data_dir,
        target_category=config.target_category if config.attack_mode != 'relation' else None,
        target_attribute=config.target_attribute if config.attack_mode == 'attribute' else None,
        target_relation=config.target_relation if config.attack_mode == 'relation' else None,
        split=_split,
        max_clips=config.max_clips,
        only_scenarios=_only,
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

    # Targeting policy: attack only the LARGEST target object per clip. Masks are
    # over-segmented (many noise fragments VINE may also label as the target), so
    # without this the patch is trained against fragments, not the real subject.
    instances = _keep_largest_target_per_clip(instances, config)
    print(f"  → {len(instances)} instances after largest-target-per-clip filter.")

    # ── Phase 0a: anchor (SAM selection, or reuse a warm-start patch's anchor) ──
    if getattr(config, 'init_patch_path', ''):
        print("\n=== Phase 0a: reusing anchor from warm-start patch ===")
        _init_ckpt = torch.load(config.init_patch_path, map_location=device, weights_only=False)
        anchor = _init_ckpt['anchor']
        print(f"  anchor = {anchor}  (from {config.init_patch_path})")
    else:
        _init_ckpt = None
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

    # SAM2 is only needed for Phase 0a — free it before loading DINOv2 + VGG
    if sam_predictor is not None:
        del sam_predictor
    torch.cuda.empty_cache()

    # ── Phase 0b: DINO diversity clustering ──────────────────────────────────
    print("\n=== Phase 0b: DINO diversity clustering ===")
    dino_model = AutoModel.from_pretrained(config.dino_model_name)
    dino_model.eval().to(device)
    for p in dino_model.parameters():
        p.requires_grad_(False)

    # Cap clusters at the instance count (a single scenario has only ~12 frames,
    # so k=20 would crash KMeans).
    eff_k = max(1, min(config.dino_cluster_k, len(instances)))
    cluster_labels = _build_dino_clusters(
        instances=instances,
        dino_model=dino_model,
        dino_preprocess_fn=dino_preprocess_torch,
        k=eff_k,
        device=device,
    )

    # DINOv2 is only needed past this point for the Chameleon dino loss. With
    # Chameleon off, free it now (clustering is already done).
    if not config.use_chameleon:
        del dino_model
        dino_model = None
        torch.cuda.empty_cache()

    # ── Precompute reference Gram matrices (Chameleon only) ──────────────────
    reference_grams = vgg = vgg_mean = vgg_std = None
    if config.use_chameleon:
        print("\n=== Precomputing reference Gram matrices ===")
        gram_cache = os.path.join(
            config.ref_gram_cache,
            f"{config.attack_mode}_{config.target_category}_gram.pt",
        )
        # Relation gram-instances need the frame path plus the subject mask/bbox;
        # load_frame reads only _png_path (.get in case it's ever absent).
        gram_instances = instances if config.attack_mode != 'relation' else [
            {'_png_path': i.get('_png_path'), 'mask': i['subj_mask'], 'bbox': i['subj_bbox']}
            for i in instances
        ]
        reference_grams = compute_reference_grams(
            instance_list=gram_instances,
            n_instances=config.ref_gram_n,
            device=device,
            cache_path=gram_cache,
        )

        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).eval().to(device)
        for p in vgg.parameters():
            p.requires_grad_(False)
        vgg_mean, vgg_std = build_vgg_normalisation_tensors(device)

    # ── Initialise patch and optimizer ───────────────────────────────────────
    content_ref = None   # [ph, pw, 3] in [0,255]: the real object under the footprint

    def _build_content_ref(ph_canon, pw_canon):
        """Crop the target object's appearance UNDER the patch footprint from the
        clearest (largest-bbox) instance, resized to the canonical patch size — the
        'what should be here' image used for object-init and the L_content anchor."""
        import torch.nn.functional as _F
        def _area(i):
            b = i['subj_bbox'] if config.attack_mode == 'relation' else i['bbox']
            return (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
        ci = max(instances, key=_area)
        cb = ci['subj_bbox'] if config.attack_mode == 'relation' else ci['bbox']
        cframe = _load_frame(ci).to(device)                            # [H,W,3] in [0,255]
        ih, iw = compute_patch_size(cb, config.coverage_ratio)
        icx, icy = anchor_center(anchor, cb, ih, iw)
        H, W = cframe.shape[:2]
        x1 = max(0, icx - iw // 2); y1 = max(0, icy - ih // 2)
        x2 = min(W, x1 + iw);       y2 = min(H, y1 + ih)
        crop = cframe[y1:y2, x1:x2, :].permute(2, 0, 1).unsqueeze(0)   # [1,3,h,w]
        crop = _F.interpolate(crop, size=(ph_canon, pw_canon), mode='bilinear', align_corners=False)
        return crop.squeeze(0).permute(1, 2, 0).contiguous()          # [ph,pw,3] in [0,255]

    def _object_init_w(ref_hwc):
        """atanh-invert the object crop into the tanh-parameterised weights so the
        patch STARTS visually equal to that crop (pixels = 127.5·(tanh(w)+1))."""
        t = (ref_hwc.permute(2, 0, 1) / 127.5 - 1.0).clamp(-0.999, 0.999)
        return torch.atanh(t)

    if _init_ckpt is not None:
        # Reuse the Phase-1 patch's canonical size + anchor. Pixels: object-init from
        # the real object crop (hidden start), else loud warm-start (Phase-2 default).
        patch_h, patch_w = _init_ckpt['w'].shape[1], _init_ckpt['w'].shape[2]
        patch = AdversarialPatch(patch_h, patch_w, device)
        if config.object_init:
            content_ref = _build_content_ref(patch_h, patch_w)
            patch.w.data.copy_(_object_init_w(content_ref))
            print(f"  Object-init patch ({patch_h}x{patch_w}) from real object crop "
                  f"(anchor/size from {config.init_patch_path})")
        else:
            patch.w.data.copy_(_init_ckpt['w'].to(device))
            print(f"  Warm-started patch ({patch_h}x{patch_w}) from {config.init_patch_path}")
    else:
        sample_bbox = (
            instances[0]['bbox'] if config.attack_mode != 'relation'
            else instances[0]['subj_bbox']
        )
        patch_h, patch_w = compute_patch_size(sample_bbox, config.coverage_ratio)
        patch = AdversarialPatch(patch_h, patch_w, device)
        if config.object_init:
            content_ref = _build_content_ref(patch_h, patch_w)
            patch.w.data.copy_(_object_init_w(content_ref))
            print(f"  Object-init patch ({patch_h}x{patch_w}) from real object crop")

    # Content anchor for the loss (built lazily if requested without object-init).
    if config.lambda_content > 0 and content_ref is None:
        content_ref = _build_content_ref(patch_h, patch_w)

    optimizer = optim.Adam(patch.parameters(), lr=config.learning_rate)

    os.makedirs(config.output_dir, exist_ok=True)

    # ── Main training loop ────────────────────────────────────────────────────
    print(f"\n=== Training patch ({config.num_steps} steps) ===")

    for step in range(config.num_steps):
        optimizer.zero_grad()

        # Diversity-sampled mini-batch
        batch = _diversity_sample(instances, cluster_labels, config.eot_batch_size, eff_k)

        step_attack   = torch.tensor(0.0, device=device)
        step_gram     = torch.tensor(0.0, device=device)
        step_tv       = torch.tensor(0.0, device=device)
        step_dino_l   = torch.tensor(0.0, device=device)
        step_content  = torch.tensor(0.0, device=device)
        step_preserve = torch.tensor(0.0, device=device)

        for inst in batch:
            # Per-instance vocabulary: each clip has its own keyword list
            inst_spec       = inst['gpt_spec']
            inst_cat_kws    = inst_spec.get('consts',     [])
            inst_unary_kws  = inst_spec.get('unary_kws',  [])
            inst_binary_kws = inst_spec.get('binary_kws', [])

            if config.attack_mode == 'relation':
                frame     = _load_frame(inst).to(device)
                subj_mask = inst['subj_mask'].to(device)
                obj_mask  = inst['obj_mask'].to(device)
                subj_bbox = inst['subj_bbox']
                obj_bbox  = inst['obj_bbox']

                # RELATION: match the ESCA planner's per-relation 2-way contrast
                # [rel, "not rel"] (esca.py binary grounding), NOT multi-way over
                # RELATION_WORDS. Suppressing the relation logit below "not rel" is
                # what deletes the (subject, rel, object) edge at deployment.
                binary_kws  = [config.target_relation, f"not {config.target_relation}"]
                correct_idx = 0

                ph, pw    = compute_patch_size(subj_bbox, config.coverage_ratio)
                cx, cy    = anchor_center(anchor, subj_bbox, ph, pw)
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
                    p_orig = _get_clean_probs(vine_model, enc, frame, inst['subj_mask'],
                                              subj_bbox, kws, device, config.clip_image_size)
                    step_preserve = step_preserve + l_preserve_kl(p_orig, p_pat)

            else:
                frame = _load_frame(inst).to(device)
                mask  = inst['mask'].to(device)
                bbox  = inst['bbox']

                if config.attack_mode == 'category':
                    enc         = vine_model.clip_cate_model
                    kws         = inst_cat_kws
                    target_str  = config.target_category
                else:
                    # ATTRIBUTE: ground the SAME way the ESCA planner does — an independent
                    # 2-way contrast [attr, "not attr"] per attribute (esca.py:585), NOT
                    # multi-way over the whole ATTRIBUTE_WORDS vocab. Suppressing the
                    # attribute logit below "not attr" is exactly what flips it at deployment.
                    enc         = vine_model.clip_unary_model
                    kws         = [config.target_attribute, f"not {config.target_attribute}"]
                    target_str  = config.target_attribute

                correct_idx = next(
                    (i for i, c in enumerate(kws) if target_str.lower() in c.lower()), None
                )
                if correct_idx is None:
                    continue

                ph, pw  = compute_patch_size(bbox, config.coverage_ratio)
                cx, cy  = anchor_center(anchor, bbox, ph, pw)
                patched = paste_patch(
                    frame, patch.pixels_hwc,
                    cx, cy, ph, pw,
                    offset_range=config.offset_range,
                )

                # Category head classifies on the FULL frame (background whitened, no
                # crop) — its validated eval forward. The ATTRIBUTE (unary) head instead
                # matches the MODEL's forward (llava_clip_model_v3:205-210):
                # extract_single_object(bbox_mask)+crop_to_bbox == a RAW bbox crop, since
                # outside-bbox pixels are cropped away. The ESCA planner grounds
                # attributes through that cropped bbox forward, so optimising the
                # full-frame view instead INVERTS the gradient (the patch raises the
                # attribute score at deployment). proc (full frame) still feeds preserve.
                proc = extract_single_object_torch(patched, mask, alpha=vine_model.config.alpha)
                if config.attack_mode == 'attribute':
                    proc_attack = crop_image_contain_bboxes_torch(patched, [bbox])
                else:
                    proc_attack = proc

                scores = _clip_scores(vine_model, enc, proc_attack, kws, device, config.clip_image_size)
                attack_terms = [l_attack_cw(scores, correct_idx, config.kappa)]
                # Fix-B EOT: also flip SGClip across jittered detector-like crops, so
                # the patch transfers to ESCA's live GroundingDINO boxes (not just the
                # full-frame view). Each crop is differentiable through the patch.
                # Attribute crops are RAW (whiten=False) to match the model's bbox crop.
                for _k in range(config.eot_crops):
                    obj_k = _eot_crop_object(patched, mask, bbox, vine_model.config.alpha,
                                             config.eot_scale_jitter,
                                             whiten=(config.attack_mode != 'attribute'))
                    if obj_k is None:
                        continue
                    s_k = _clip_scores(vine_model, enc, obj_k, kws, device, config.clip_image_size)
                    attack_terms.append(l_attack_cw(s_k, correct_idx, config.kappa))
                step_attack = step_attack + torch.stack(attack_terms).mean()

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
                    p_orig = _get_clean_probs(vine_model, penc, frame, inst['mask'],
                                              bbox, pkws, device, config.clip_image_size)
                    step_preserve = step_preserve + l_preserve_kl(p_orig, p_pat)

            # Gram + DINO naturalness (need VGG16 / DINOv2 loaded)
            if config.use_chameleon:
                patch_px = patch.pixels_hwc  # [H, W, 3]
                step_gram   = step_gram   + l_chameleon_gram(patch_px, reference_grams, vgg, vgg_mean, vgg_std)
                step_dino_l = step_dino_l + l_chameleon_dino(
                    patched, _patch_bbox_from_anchor(cx, cy, ph, pw, patched.shape),
                    dino_model, dino_preprocess_torch,
                )

            # TV smoothness + appearance/content anchor — network-free, so they run
            # independent of use_chameleon (object-init camouflage = content [+ TV],
            # without the VGG/DINO stack).
            if config.lambda_tv > 0:
                step_tv = step_tv + l_chameleon_tv(patch.pixels_hwc)
            if config.lambda_content > 0 and content_ref is not None:
                step_content = step_content + l_chameleon_content(patch.pixels_hwc, content_ref)

        n = len(batch)
        loss = total_loss(
            attack_loss   = step_attack   / n,
            gram_loss     = step_gram     / n,
            tv_loss       = step_tv       / n,
            dino_loss     = step_dino_l   / n,
            preserve_loss = step_preserve / n,
            content_loss  = (step_content / n) if config.lambda_content > 0 else None,
            lambda_c      = config.lambda_c,
            lambda_tv     = config.lambda_tv,
            lambda_dino   = config.lambda_dino,
            lambda_p      = config.lambda_p,
            lambda_content= config.lambda_content,
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
                f"content={step_content.item()/n:.4f} | "
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
    Keep only instances where VINE's top-1 clean prediction matches the target,
    using the SAME gates as eval so training and evaluation target the same objects.

    - category attack:  predicted category contains target_category
    - attribute attack: predicted category contains target_category AND
                        predicted attribute contains target_attribute
    - relation attack:  subject's predicted category contains target_category AND
                        predicted relation contains target_relation

    The category gate matters: without it, an attribute attack on "stand" keeps any
    object VINE calls "standing" (e.g. a table), not just standing PERSONS, which
    both wastes the patch and mismatches eval.
    """
    from functions.attack.preprocess import (
        extract_single_object_torch,
        extract_object_subject_torch,
        crop_image_contain_bboxes_torch,
    )

    def _head_top1_ok(enc, proc_img, kws, target, min_conf: float = 0.0,
                      exact: bool = False) -> bool:
        """
        True if `target` is empty (no gate), or VINE's top-1 over `kws` matches
        `target` — EXACT word when `exact` (categories: 'pot' != 'potato'),
        else substring (attribute keywords: target 'open' ⊂ predicted 'opened')
        — AND prob >= min_conf.
        """
        if not target:
            return True
        if not kws:
            return False
        with torch.no_grad():
            scores = _clip_scores(vine_model, enc, proc_img, kws, device, config.clip_image_size)
            probs  = torch.softmax(scores, dim=-1)
        i = int(probs.argmax().item())
        ok = (target.lower() == kws[i].lower()) if exact else (target.lower() in kws[i].lower())
        if not ok:
            return False
        return float(probs[i].item()) >= min_conf

    kept = []
    n_err = 0
    for inst in instances:
        frame  = _load_frame(inst).to(device)
        spec   = inst['gpt_spec']
        consts = spec.get('consts', [])

        try:
            if config.attack_mode in ('category', 'attribute'):
                mask = inst['mask'].to(device)
                # category/attribute classify on the full frame (background whitened)
                proc = extract_single_object_torch(frame, mask, alpha=vine_model.config.alpha)

                if config.attack_mode == 'category':
                    ok = _head_top1_ok(vine_model.clip_cate_model, proc, consts,
                                       config.target_category, min_conf=config.min_target_conf,
                                       exact=True)
                else:
                    # attribute: require the target CATEGORY (full-frame view, its
                    # validated forward) AND the target ATTRIBUTE grounded through the
                    # model's unary forward — a RAW bbox crop (llava_clip_model_v3:205-210).
                    # This keeps exactly the instances the ESCA planner reads as having
                    # the attribute, so filter, loss and deployment share ONE forward.
                    proc_attr = crop_image_contain_bboxes_torch(frame, [inst['bbox']])
                    ok = (
                        _head_top1_ok(vine_model.clip_cate_model,  proc, consts,
                                      config.target_category, min_conf=config.min_target_conf,
                                      exact=True)
                        and
                        _head_top1_ok(vine_model.clip_unary_model, proc_attr,
                                      [config.target_attribute, f"not {config.target_attribute}"],
                                      config.target_attribute)
                    )

            else:  # relation: require subject's category AND the target relation
                subj_mask = inst['subj_mask'].to(device)
                obj_mask  = inst['obj_mask'].to(device)
                subj_bbox = inst['subj_bbox']
                obj_bbox  = inst['obj_bbox']

                # subject category gate — full-frame whitened subject, as eval classifies it
                proc_subj = extract_single_object_torch(frame, subj_mask, alpha=vine_model.config.alpha)
                if not _head_top1_ok(vine_model.clip_cate_model, proc_subj, consts,
                                     config.target_category, min_conf=config.min_target_conf,
                                     exact=True):
                    continue

                # relation gate — subject+object crop, as eval classifies the relation
                proc = extract_object_subject_torch(
                    frame, subj_mask, obj_mask,
                    alpha=vine_model.config.alpha, white_alpha=vine_model.config.white_alpha,
                )
                crop = crop_image_contain_bboxes_torch(proc, [subj_bbox, obj_bbox])
                ok = _head_top1_ok(vine_model.clip_binary_model, crop,
                                   [config.target_relation, f"not {config.target_relation}"],
                                   config.target_relation)

            if ok:
                kept.append(inst)

        except Exception as e:
            n_err += 1
            if n_err <= 3:
                print(f"  [filter] instance dropped by error: {type(e).__name__}: {e}")
            continue

    if n_err:
        print(f"  [filter] {n_err} instance(s) dropped due to errors (first 3 shown). "
              f"If ALL instances were dropped this is likely a code bug, not VINE.")
    return kept


def _keep_largest_target_per_clip(instances: list, config: 'AttackConfig') -> list:
    """
    Keep only the instances belonging to the LARGEST target object in each clip.

    `instances` is already filtered to the target category/predicate by clean VINE
    prediction. Because the masks are over-segmented (dozens of objects per clip,
    many noise fragments that VINE may also classify as the target), we pick the
    single largest-area target object per clip — the actual subject — and drop the
    rest. All frames of that object are retained for EoT. Mirrors the eval-side
    `largest_object_id` so training and evaluation attack the SAME object.
    """
    from collections import defaultdict

    is_rel   = config.attack_mode == 'relation'
    id_key   = 'subj_obj_id' if is_rel else 'obj_id'
    mask_key = 'subj_mask'   if is_rel else 'mask'

    by_clip: dict = defaultdict(list)
    for inst in instances:
        by_clip[inst['data_id']].append(inst)

    kept = []
    for insts in by_clip.values():
        area: dict = {}
        for inst in insts:
            a = int(inst[mask_key].sum().item())
            oid = inst[id_key]
            if a > area.get(oid, 0):
                area[oid] = a
        if not area:
            continue
        best = max(area, key=area.get)
        kept.extend(inst for inst in insts if inst[id_key] == best)
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
