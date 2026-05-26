"""
Precompute and cache reference Gram matrices for the Gram-based chameleon loss.

The reference is the average of Gram matrices computed over N crops of the target
category from the ESCA training set. It is fixed for the entire training run.
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Optional
import torchvision.models as tv_models
import torchvision.transforms.functional as TF

# VGG16 ImageNet normalisation constants
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD  = [0.229, 0.224, 0.225]

# VGG16 feature layer indices for relu1_2, relu2_2, relu3_3
VGG_TARGET_LAYERS = {3, 8, 15}


def _gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """[1, C, H, W] → [C, C] normalised Gram matrix."""
    _, C, H, W = feat.shape
    f = feat.view(C, H * W)
    return torch.mm(f, f.t()) / (C * H * W)


def _extract_vgg_grams(
    vgg: torch.nn.Module,
    img_hwc: torch.Tensor,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Run a single [H, W, 3] float tensor (RGB, [0,255]) through VGG16 and
    return Gram matrices at relu1_2, relu2_2, relu3_3.
    """
    x = img_hwc.permute(2, 0, 1).unsqueeze(0).to(device) / 255.0  # [1,3,H,W]
    # Resize to 224×224 (VGG16 standard input size)
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

    mean = torch.tensor(VGG_MEAN, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(VGG_STD,  device=device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std

    grams = []
    h = x
    for i, layer in enumerate(vgg.features):
        h = layer(h)
        if i in VGG_TARGET_LAYERS:
            grams.append(_gram_matrix(h).cpu())
        if len(grams) == len(VGG_TARGET_LAYERS):
            break
    return grams


def compute_reference_grams(
    instance_list: list,
    n_instances: int,
    device: torch.device,
    cache_path: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Compute the average Gram matrices over N object crops.

    Args:
        instance_list: flat list of instance dicts (from EscaAttackDataset.build_instance_list)
                       each dict must have 'frame', 'mask', 'bbox' keys
        n_instances:   maximum number of crops to average (N=500 default)
        device:        torch device
        cache_path:    if given, save/load the result from this .pt file

    Returns:
        List of 3 Gram tensors: [C1×C1, C2×C2, C3×C3] on CPU
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached reference Gram from {cache_path}")
        return torch.load(cache_path, map_location='cpu')

    from functions.attack.preprocess import extract_single_object_torch, crop_image_contain_bboxes_torch

    vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
    vgg.eval().to(device)
    for p in vgg.parameters():
        p.requires_grad_(False)

    samples = instance_list[:n_instances]
    print(f"Computing reference Gram over {len(samples)} instances...")

    # Accumulate Gram matrices
    gram_accum: Optional[List[torch.Tensor]] = None

    with torch.no_grad():
        for i, inst in enumerate(samples):
            frame = inst['frame'].to(device)
            mask  = inst['mask'].to(device)
            bbox  = inst['bbox']

            # Same preprocessing as the attack forward pass
            obj_img = extract_single_object_torch(frame, mask)
            crop    = crop_image_contain_bboxes_torch(obj_img, [bbox])

            if crop.numel() == 0:
                continue

            grams = _extract_vgg_grams(vgg, crop, device)

            if gram_accum is None:
                gram_accum = [g.clone() for g in grams]
            else:
                for j in range(len(gram_accum)):
                    gram_accum[j] += grams[j]

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(samples)}")

    if gram_accum is None:
        raise RuntimeError("No valid crops found to compute reference Gram matrices.")

    n_valid = min(n_instances, len(samples))
    reference = [g / n_valid for g in gram_accum]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(reference, cache_path)
        print(f"Saved reference Gram to {cache_path}")

    return reference


def build_vgg_normalisation_tensors(device: torch.device):
    """
    Return (mean, std) tensors for VGG16 ImageNet normalisation.
    Shape: [1, 3, 1, 1] each, on the given device.
    """
    mean = torch.tensor(VGG_MEAN, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(VGG_STD,  device=device).view(1, 3, 1, 1)
    return mean, std
