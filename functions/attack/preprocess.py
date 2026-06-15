import torch
import torch.nn.functional as F

# CLIP normalization constants — identical for all CLIP variants (ViT-B/32, ViT-L/14, ViT-L/14-336)
CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# DINOv2 normalization constants (ImageNet)
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD  = [0.229, 0.224, 0.225]


def clip_scores(
    vine_model,
    clip_encoder,
    img_hwc: torch.Tensor,
    keywords,
    device,
    clip_image_size: int,
) -> torch.Tensor:
    """
    Differentiable CLIP forward for one image — the single train-side mirror of
    VineModel's eval scoring. Returns a [N_keywords] score tensor
    (cosine similarity × logit_scale); gradients flow back through img_hwc.

    img_hwc must already be the head-appropriate image, matching eval:
      - category/attribute: extract_single_object_torch output (FULL frame)
      - relation:           extract_object_subject_torch → crop to union of bboxes

    All preprocessing (BGR swap, resize-shortest-edge + center-crop, normalize)
    and tokenisation (pad to max_length=75) match the eval path exactly, so a
    falling training loss transfers to flip rate. Used by train.py and the anchor
    warmups; keep this the ONLY train-side forward to prevent drift.
    """
    pv       = clip_preprocess_torch(img_hwc, clip_image_size)    # [1,3,H,W]
    img_feat = clip_image_features(clip_encoder, pv)              # [1, D]

    text_tokens = vine_model.clip_tokenizer(
        keywords, return_tensors='pt', max_length=75, truncation=True, padding='max_length'
    ).to(device)
    text_feat = clip_text_features(clip_encoder, **text_tokens)   # [N, D]

    img_n  = img_feat  / img_feat.norm(p=2, dim=-1, keepdim=True)
    text_n = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
    logit_scale = clip_encoder.logit_scale.exp() if hasattr(clip_encoder, 'logit_scale') else 1.0
    return (img_n @ text_n.T).squeeze(0) * logit_scale            # [N]


def clip_image_features(clip_model, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Extract CLIP projected image features as a [B, D] tensor.

    Transformers >=5.x changed get_image_features() to return BaseModelOutputWithPooling
    instead of a plain tensor. This helper bypasses that by going through vision_model
    and visual_projection directly, which is stable across versions.
    """
    vision_out = clip_model.vision_model(pixel_values=pixel_values)
    return clip_model.visual_projection(vision_out.pooler_output)


def clip_text_features(clip_model, **text_inputs) -> torch.Tensor:
    """
    Extract CLIP projected text features as a [N, D] tensor.

    Same version-compatibility wrapper as clip_image_features.
    """
    text_out = clip_model.text_model(**text_inputs)
    return clip_model.text_projection(text_out.pooler_output)


def extract_single_object_torch(frame: torch.Tensor, mask: torch.Tensor, alpha: float = 0.8) -> torch.Tensor:
    """
    Differentiable PyTorch reimplementation of model_utils.extract_single_object.

    Object pixels are left unchanged.
    Background pixels are blended: pixel_out = pixel * (1 - alpha) + 255 * alpha.

    Args:
        frame: [H, W, 3] float tensor, RGB, values in [0, 255]
        mask:  [H, W] bool/float tensor — True/1 where the object is
        alpha: blend strength toward white for background (default 0.8)

    Returns:
        [H, W, 3] float tensor
    """
    frame = frame.float()
    H, W = frame.shape[0], frame.shape[1]

    mask_2d = mask.bool()
    if mask_2d.dim() == 3:
        mask_2d = mask_2d.squeeze(-1)
    if mask_2d.shape[0] != H or mask_2d.shape[1] != W:
        mask_2d = F.interpolate(
            mask_2d.float().unsqueeze(0).unsqueeze(0),
            size=(H, W), mode='nearest'
        ).squeeze(0).squeeze(0).bool()

    mask_3d = mask_2d.unsqueeze(-1)              # [H, W, 1]
    bg = mask_3d.logical_not()

    white = torch.full_like(frame, 255.0)
    masked_white = torch.where(bg, white, frame)
    return frame * (1.0 - alpha) + masked_white * alpha


def extract_object_subject_torch(
    frame: torch.Tensor,
    red_mask: torch.Tensor,
    blue_mask: torch.Tensor,
    alpha: float = 0.5,
    white_alpha: float = 0.8,
) -> torch.Tensor:
    """
    Differentiable PyTorch reimplementation of model_utils.extract_object_subject.

    Subject (red_mask): R channel boosted — r_out = clip(r + (255 - r) * alpha, 0, 255)
    Object  (blue_mask): B channel boosted — b_out = clip(b + (255 - b) * alpha, 0, 255)
    Background: blended 80% toward white.

    Assumes RGB channel order (channel 0 = R, channel 2 = B).

    Args:
        frame:      [H, W, 3] float tensor, RGB, values in [0, 255]
        red_mask:   [H, W] or [H, W, 1] bool/float tensor — subject
        blue_mask:  [H, W] or [H, W, 1] bool/float tensor — object
        alpha:      color boost strength (default 0.5)
        white_alpha: background blend strength toward white (default 0.8)

    Returns:
        [H, W, 3] float tensor, RGB
    """
    frame = frame.float()
    H, W = frame.shape[0], frame.shape[1]

    def _squeeze_to_2d(m: torch.Tensor) -> torch.Tensor:
        m = m.bool()
        if m.dim() == 3:
            m = m.squeeze(-1)
        if m.shape[0] != H or m.shape[1] != W:
            m = F.interpolate(
                m.float().unsqueeze(0).unsqueeze(0),
                size=(H, W), mode='nearest'
            ).squeeze(0).squeeze(0).bool()
        return m  # [H, W]

    red_2d  = _squeeze_to_2d(red_mask)
    blue_2d = _squeeze_to_2d(blue_mask)
    bg_2d   = ~(red_2d | blue_2d)              # [H, W]

    r = frame[..., 0]
    g = frame[..., 1]
    b = frame[..., 2]

    # Boost red channel on subject region
    r = torch.where(red_2d,  torch.clamp(r + (255.0 - r) * alpha, 0.0, 255.0), r)
    # Boost blue channel on object region
    b = torch.where(blue_2d, torch.clamp(b + (255.0 - b) * alpha, 0.0, 255.0), b)

    out = torch.stack([r, g, b], dim=-1)        # [H, W, 3]

    # Blend background toward white
    white      = torch.full_like(out, 255.0)
    bg_blended = out * (1.0 - white_alpha) + white * white_alpha
    bg_3d      = bg_2d.unsqueeze(-1)
    return torch.where(bg_3d, bg_blended, out)


def crop_image_contain_bboxes_torch(img: torch.Tensor, bbox_list: list) -> torch.Tensor:
    """
    Differentiable PyTorch reimplementation of model_utils.crop_image_contain_bboxes.

    Crops img to the union bounding box of all provided bboxes.
    Tensor slicing preserves gradients.

    Args:
        img:       [H, W, 3] float tensor
        bbox_list: list of dicts {x1,y1,x2,y2} or lists/tuples [x1,y1,x2,y2]

    Returns:
        [h, w, 3] float tensor
    """
    all_x1, all_y1, all_x2, all_y2 = [], [], [], []

    for bbox in bbox_list:
        if isinstance(bbox, dict):
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        else:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        x1, x2 = int(min(x1, x2)), int(max(x1, x2))
        y1, y2 = int(min(y1, y2)), int(max(y1, y2))

        all_x1.append(x1); all_x2.append(x2)
        all_y1.append(y1); all_y2.append(y2)

    H_img, W_img = img.shape[:2]

    cx1, cx2 = min(all_x1), max(all_x2)
    cy1, cy2 = min(all_y1), max(all_y2)

    # Clamp to image bounds and guarantee at least 1×1
    cx1 = max(0, min(cx1, W_img - 1))
    cy1 = max(0, min(cy1, H_img - 1))
    cx2 = max(cx1 + 1, min(cx2, W_img))
    cy2 = max(cy1 + 1, min(cy2, H_img))

    return img[cy1:cy2, cx1:cx2]


def clip_preprocess_torch(img_hwc: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """
    Differentiable CLIP image preprocessing. Replaces CLIPProcessor for gradient flow.

    This MUST match VineModel's eval path (VineModel._extract_image_features →
    CLIPProcessor) op-for-op, or a patch trained through it will not transfer to
    eval. Verified to reproduce the eval image features to cosine 1.0000.

    The eval path applies, in order:
      1. RGB is kept as-is. (Historically eval did cv2.cvtColor(BGR2RGB) on an
         already-RGB frame — a channel REVERSAL that fed CLIP BGR; that swap was a
         bug, now removed from both eval and here. VINE/SGClip expects RGB.)
      2. CLIPImageProcessor: resize the SHORTEST edge to image_size (bicubic,
         antialiased), center-crop image_size × image_size, rescale 1/255,
         normalize with CLIP mean/std. NOTE: this is resize-shortest-edge +
         center-crop, NOT a squash to a square — getting this wrong shifts the
         object scale and silently degrades transfer.

    Args:
        img_hwc:    [H, W, 3] float tensor, RGB, values in [0, 255]
        image_size: target spatial size (224 for ViT-B/32, 336 for ViT-L/14-336)

    Returns:
        [1, 3, image_size, image_size] float tensor, ready for clip vision_model
    """
    x = img_hwc.permute(2, 0, 1).unsqueeze(0).float()   # [1, 3, H, W], RGB (kept RGB)

    # Resize shortest edge → image_size, preserving aspect ratio (step 2a)
    h, w = x.shape[-2], x.shape[-1]
    scale = image_size / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    x = F.interpolate(x, size=(new_h, new_w), mode='bicubic',
                      align_corners=False, antialias=True)

    # Center crop image_size × image_size (step 2b)
    top  = (new_h - image_size) // 2
    left = (new_w - image_size) // 2
    x = x[..., top:top + image_size, left:left + image_size]

    x = x / 255.0
    mean = torch.tensor(CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(CLIP_STD,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def dino_preprocess_torch(img_hwc: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """
    Differentiable DINOv2 image preprocessing.

    Args:
        img_hwc:    [H, W, 3] float tensor, RGB, values in [0, 255]
        image_size: target spatial size (default 224)

    Returns:
        [1, 3, image_size, image_size] float tensor, ready for dino_model(pixel_values=...)
    """
    x = img_hwc.permute(2, 0, 1).unsqueeze(0).float()   # [1, 3, H, W]
    x = F.interpolate(x, size=(image_size, image_size), mode='bilinear', align_corners=False)
    x = x / 255.0
    mean = torch.tensor(DINO_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(DINO_STD,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std
