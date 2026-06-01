import torch
import torch.nn.functional as F
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# L_attack : CW margin loss (untargeted)
# ──────────────────────────────────────────────────────────────────────────────

def l_attack_cw(
    scores: torch.Tensor,
    correct_idx: int,
    kappa: float = 0.0,
) -> torch.Tensor:
    """
    Carlini-Wagner untargeted margin loss.

    Drives score_correct below max_{j ≠ correct} score_j by margin κ.
    Returns zero once the attack has already succeeded — no wasted gradient budget.

    Args:
        scores:      [N] or [1, N] logit tensor from a single CLIP head
        correct_idx: index of the ground-truth label in the score vector
        kappa:       confidence margin (default 0 — flip the top-1 prediction)

    Returns:
        scalar tensor
    """
    scores = scores.view(-1)
    correct_score = scores[correct_idx]

    # Best score among all wrong labels
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[correct_idx] = False
    wrong_scores = scores[mask]
    if wrong_scores.numel() == 0:
        # Only one class in vocabulary — nothing to flip to; no gradient signal
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    best_wrong = wrong_scores.max()

    return F.relu(correct_score - best_wrong + kappa)


# ──────────────────────────────────────────────────────────────────────────────
# L_chameleon : VGG16 Gram style loss + total variation + DINOv2 scene consistency
# ──────────────────────────────────────────────────────────────────────────────

def _gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """
    Compute normalised Gram matrix from a [1, C, H, W] feature map.
    Returns [C, C] tensor.
    """
    _, C, H, W = feature_map.shape
    f = feature_map.view(C, H * W)
    return torch.mm(f, f.t()) / (C * H * W)


def l_chameleon_gram(
    patch_pixels: torch.Tensor,
    reference_grams: List[torch.Tensor],
    vgg: torch.nn.Module,
    vgg_mean: torch.Tensor,
    vgg_std: torch.Tensor,
) -> torch.Tensor:
    """
    Gram matrix style loss between the patch and the precomputed reference Gram matrices.

    Args:
        patch_pixels:    [3, H, W] or [H, W, 3] float tensor, values [0, 255]
        reference_grams: list of [C, C] tensors, one per VGG layer (relu1_2, relu2_2, relu3_3)
        vgg:             VGG16 feature extractor (torchvision, eval mode, no_grad for reference;
                         grad DOES flow through this call for the patch)
        vgg_mean:        [1, 3, 1, 1] ImageNet mean tensor (on same device as patch)
        vgg_std:         [1, 3, 1, 1] ImageNet std tensor

    Returns:
        scalar tensor
    """
    if patch_pixels.dim() == 3 and patch_pixels.shape[-1] == 3:
        patch_pixels = patch_pixels.permute(2, 0, 1)   # [3, H, W]

    x = patch_pixels.unsqueeze(0) / 255.0              # [1, 3, H, W] in [0, 1]
    x = (x - vgg_mean) / vgg_std

    # VGG16 has 2 MaxPool layers before relu3_3 — input must be at least 32×32
    if x.shape[2] < 32 or x.shape[3] < 32:
        x = F.interpolate(
            x,
            size=(max(32, x.shape[2]), max(32, x.shape[3])),
            mode='bilinear', align_corners=False,
        )

    feats = _vgg_features(vgg, x)                      # list of [1, C, H, W]

    loss = torch.tensor(0.0, device=patch_pixels.device)
    for feat, ref_gram in zip(feats, reference_grams):
        g = _gram_matrix(feat)
        loss = loss + F.mse_loss(g, ref_gram.to(patch_pixels.device))
    return loss


def _vgg_features(vgg: torch.nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    """
    Extract feature maps at VGG16 relu1_2, relu2_2, relu3_3.
    Assumes vgg.features is the standard torchvision VGG16 features module.
    Layer indices: relu1_2 = 3, relu2_2 = 8, relu3_3 = 15
    """
    TARGET_LAYERS = {3, 8, 15}
    feats = []
    h = x
    for i, layer in enumerate(vgg.features):
        h = layer(h)
        if i in TARGET_LAYERS:
            feats.append(h)
        if len(feats) == len(TARGET_LAYERS):
            break
    return feats


def l_chameleon_tv(patch_pixels: torch.Tensor) -> torch.Tensor:
    """
    Total variation regularisation — penalises noisy / high-frequency patch content.

    Args:
        patch_pixels: [3, H, W] or [H, W, 3] float tensor

    Returns:
        scalar tensor
    """
    if patch_pixels.dim() == 3 and patch_pixels.shape[-1] == 3:
        patch_pixels = patch_pixels.permute(2, 0, 1)   # [3, H, W]

    diff_h = patch_pixels[:, 1:, :] - patch_pixels[:, :-1, :]
    diff_w = patch_pixels[:, :, 1:] - patch_pixels[:, :, :-1]
    return diff_h.pow(2).mean() + diff_w.pow(2).mean()


def l_chameleon_dino(
    patched_frame: torch.Tensor,
    patch_bbox: dict,
    dino_model: torch.nn.Module,
    dino_preprocess_fn,
) -> torch.Tensor:
    """
    DINOv2 scene consistency loss.

    Minimises L2 distance between the CLS token of the patch region and
    the CLS token of the full frame. Over training on diverse frames the patch
    converges toward semantic neutrality — it blends into any scene context.

    Args:
        patched_frame:       [H, W, 3] float tensor, RGB [0, 255]
        patch_bbox:          {x1, y1, x2, y2} — region occupied by the patch
        dino_model:          DINOv2 model with .forward() returning last_hidden_state
        dino_preprocess_fn:  callable: Tensor[H,W,3] → Tensor[1,3,224,224]

    Returns:
        scalar tensor
    """
    # Full-frame CLS (no grad — we only want the patch to move, not the frame)
    with torch.no_grad():
        frame_input = dino_preprocess_fn(patched_frame)
        frame_out   = dino_model(pixel_values=frame_input)
        f_scene     = frame_out.last_hidden_state[:, 0, :]   # [1, D]

    # Patch-region CLS (grad flows through patch pixels)
    x1, y1 = int(patch_bbox['x1']), int(patch_bbox['y1'])
    x2, y2 = int(patch_bbox['x2']), int(patch_bbox['y2'])
    H_f, W_f = patched_frame.shape[:2]
    x1 = max(0, min(x1, W_f - 1)); x2 = max(x1 + 1, min(x2, W_f))
    y1 = max(0, min(y1, H_f - 1)); y2 = max(y1 + 1, min(y2, H_f))
    patch_region = patched_frame[y1:y2, x1:x2]              # [ph, pw, 3]

    patch_input = dino_preprocess_fn(patch_region)
    patch_out   = dino_model(pixel_values=patch_input)
    f_patch     = patch_out.last_hidden_state[:, 0, :]       # [1, D]

    return F.mse_loss(f_patch, f_scene.detach())


# ──────────────────────────────────────────────────────────────────────────────
# L_preserve : KL divergence on non-attacked heads
# ──────────────────────────────────────────────────────────────────────────────

def l_preserve_kl(
    orig_probs: torch.Tensor,
    patched_probs: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence loss between original and patched probability distributions.

    Used to keep non-attacked CLIP heads unchanged.
    KL(P_original || P_patched) — penalises when the patched distribution
    deviates from the original.

    Args:
        orig_probs:   [N] or [1, N] probability tensor (softmax output, no grad)
        patched_probs:[N] or [1, N] probability tensor (with grad)

    Returns:
        scalar tensor
    """
    p = orig_probs.view(1, -1).detach()
    q = patched_probs.view(1, -1)
    # F.kl_div expects log-probabilities for input, probabilities for target
    return F.kl_div(q.log(), p, reduction='batchmean')


# ──────────────────────────────────────────────────────────────────────────────
# Combined loss
# ──────────────────────────────────────────────────────────────────────────────

def total_loss(
    attack_loss: torch.Tensor,
    gram_loss: torch.Tensor,
    tv_loss: torch.Tensor,
    dino_loss: torch.Tensor,
    preserve_loss: torch.Tensor,
    lambda_c: float = 1.0,
    lambda_tv: float = 1e-4,
    lambda_dino: float = 0.5,
    lambda_p: float = 0.5,
) -> torch.Tensor:
    """
    Objective minimised over the patch pixels w only (VINE is frozen), averaged
    over an EoT mini-batch of diversity-sampled target instances with a random
    paste offset:

        L_total = L_attack
                + λ_c · ( L_gram + λ_tv · L_tv + λ_dino · L_dino )    [= L_chameleon]
                + λ_p · L_preserve

    Term by term (all evaluated on the patched target object):

      L_attack   — CW untargeted margin (l_attack_cw) on the ATTACKED head's CLIP
                   logits over that clip's keyword vocabulary. correct_idx is the
                   TARGET label we want to suppress; the loss = ReLU(s_target −
                   max_{j≠target} s_j + κ). It pushes the target label below the
                   best competitor by margin κ and is 0 once already flipped.
                   Attacked head: category→clip_cate_model/`consts`,
                   attribute→clip_unary_model/`unary_kws`,
                   relation→clip_binary_model/`binary_kws`.

      L_preserve — Σ KL(P_clean ‖ P_patched) over the NON-attacked heads of the
                   same object. Keeps the other scene-graph predictions unchanged,
                   giving a surgical (isolated) attack rather than global damage.

      L_chameleon — naturalness regulariser on the patch so it reads as a benign
                   object rather than noise:
                     L_gram  : VGG16 Gram-matrix style distance to real object crops,
                     L_tv    : total variation (spatial smoothness),
                     L_dino  : DINOv2 CLS-token distance between the patch region and
                               the full frame (semantic blending into the scene).

    Defaults: λ_c=1.0, λ_tv=1e-4, λ_dino=0.5, λ_p=0.5, κ=0 (flip the top-1 label).
    """
    chameleon = gram_loss + lambda_tv * tv_loss + lambda_dino * dino_loss
    return attack_loss + lambda_c * chameleon + lambda_p * preserve_loss
