"""Gemma 4 image geometry, implemented once and shared by the collator, the teacher cache and
the grafted VLM.

What ``Gemma4ImageProcessor`` actually does (transformers >= 5.4):

1. Aspect-preserving resize to the *largest* ``(H, W)`` with ``H % 48 == W % 48 == 0`` and
   ``(H/16) * (W/16) <= max_soft_tokens * 9`` (bicubic, antialiased). The image is *resized*
   to multiples of 48, it is never padded spatially.
2. Rescale to ``[0, 1]``; no mean/std normalisation (the model maps to ``[-1, 1]`` itself).
3. Patchify into 16x16 patches, flattened row-major, channel-last within a patch.
4. Pad the patch *sequence* to ``max_soft_tokens * 9`` with position ``(-1, -1)``.

Spatial padding only appears when we stack differently-sized images into a dense student
batch; it is always bottom/right and in whole 48-px cells.
"""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as tvF

from onboard_vlm.interfaces import (
    PATCH_SIZE,
    POOLING_KERNEL,
    SUPPORTED_TOKEN_BUDGETS,
    TOKEN_STRIDE,
    TransformParams,
)

try:  # use the reference implementation when available so we can never drift from HF
    from transformers.models.gemma4.image_processing_gemma4 import (
        get_aspect_ratio_preserving_size as _hf_target_size,
    )
except ImportError:  # pragma: no cover - transformers < 5.4
    _hf_target_size = None


def target_size(
    height: int,
    width: int,
    token_budget: int,
    patch_size: int = PATCH_SIZE,
    pooling_kernel: int = POOLING_KERNEL,
) -> tuple[int, int]:
    """Gemma 4 output size for a ``height x width`` image at ``token_budget`` soft tokens."""
    if token_budget not in SUPPORTED_TOKEN_BUDGETS:
        raise ValueError(f"token_budget must be one of {SUPPORTED_TOKEN_BUDGETS}, got {token_budget}")
    max_patches = token_budget * pooling_kernel**2
    if _hf_target_size is not None:
        return tuple(_hf_target_size(height, width, patch_size, max_patches, pooling_kernel))  # type: ignore[return-value]
    # Fallback copy of the reference algorithm.
    side = pooling_kernel * patch_size
    factor = math.sqrt(max_patches * patch_size**2 / (height * width))
    th = int(math.floor(factor * height / side)) * side
    tw = int(math.floor(factor * width / side)) * side
    max_side = (max_patches // pooling_kernel**2) * side
    if th == 0 and tw == 0:
        raise ValueError("degenerate 0x0 target size")
    if th == 0:
        th, tw = side, min(int(math.floor(width / height)) * side, max_side)
    elif tw == 0:
        tw, th = side, min(int(math.floor(height / width)) * side, max_side)
    return th, tw


# --------------------------------------------------------------------------------------
# Transform sampling / application (the ONLY place images are geometrically transformed)
# --------------------------------------------------------------------------------------
def sample_transform(
    src_hw: tuple[int, int],
    token_budget: int,
    *,
    train: bool,
    rng: random.Random | None = None,
    crop_scale: tuple[float, float] = (0.5, 1.0),
    crop_ratio: tuple[float, float] = (3 / 4, 4 / 3),
    hflip_prob: float = 0.0,
    degradations: Mapping[str, Any] | None = None,
) -> TransformParams:
    """Sample (train) or construct (eval) the geometric transform of one image.

    Training uses an Inception-style random-resized crop (area fraction ``crop_scale``,
    log-uniform aspect jitter ``crop_ratio`` relative to the source aspect) and optional
    horizontal flip, plus optional capture degradations (``data.degrade``, train only). Evaluation
    uses the full, clean image, exactly like ``Gemma4ImageProcessor``.
    """
    h, w = src_hw
    crop_box = None
    hflip = False
    if train:
        rng = rng or random.Random()
        for _ in range(10):
            area = h * w * rng.uniform(*crop_scale)
            log_r = rng.uniform(math.log(crop_ratio[0]), math.log(crop_ratio[1]))
            aspect = (w / h) * math.exp(log_r)
            cw = int(round(math.sqrt(area * aspect)))
            ch = int(round(math.sqrt(area / aspect)))
            if 0 < cw <= w and 0 < ch <= h:
                top = rng.randint(0, h - ch)
                left = rng.randint(0, w - cw)
                crop_box = (top, left, ch, cw)
                break
        hflip = rng.random() < hflip_prob
    eff_h, eff_w = (crop_box[2], crop_box[3]) if crop_box else (h, w)
    thw = target_size(eff_h, eff_w, token_budget)
    ops = ()
    if train and degradations:
        from onboard_vlm.data.degrade import sample_degradations

        ops = sample_degradations(degradations, rng, thw)  # type: ignore[arg-type]
    return TransformParams(src_hw=(h, w), crop_box=crop_box, hflip=hflip, target_hw=thw, token_budget=token_budget, degradations=ops)


def apply_transform(image: torch.Tensor, params: TransformParams) -> torch.Tensor:
    """Apply ``params`` to ``image``.

    Args:
        image: ``UInt8Tensor[3, H_src, W_src]`` (or ``[T, 3, H_src, W_src]`` for video frames,
            which all receive the same transform).
    Returns:
        ``UInt8Tensor[..., 3, H, W]`` with ``(H, W) == params.target_hw``. Resizing is done on
        uint8 with bicubic + antialias, matching ``Gemma4ImageProcessor`` bit-for-bit.
    """
    if image.dtype != torch.uint8:
        raise TypeError(f"expected uint8 image, got {image.dtype}")
    if tuple(image.shape[-2:]) != tuple(params.src_hw):
        raise ValueError(f"image size {tuple(image.shape[-2:])} != params.src_hw {params.src_hw}")
    if params.crop_box is not None:
        t, l, ch, cw = params.crop_box
        image = image[..., t : t + ch, l : l + cw]
    if params.hflip:
        image = image.flip(-1)
    if tuple(image.shape[-2:]) != tuple(params.target_hw):
        image = tvF.resize(
            image,
            size=list(params.target_hw),
            interpolation=tvF.InterpolationMode.BICUBIC,
            antialias=True,
        )
    if params.degradations:
        from onboard_vlm.data.degrade import apply_degradations

        image = apply_degradations(image, params.degradations)
    return image.contiguous()


def to_unit_float(image_u8: torch.Tensor) -> torch.Tensor:
    """uint8 -> float32 in [0, 1] (Gemma's ``rescale_factor = 1/255``)."""
    return image_u8.to(torch.float32) * (1.0 / 255.0)


# --------------------------------------------------------------------------------------
# Teacher (Gemma) view: patch sequences
# --------------------------------------------------------------------------------------
def patch_positions(grid_h: int, grid_w: int, device: torch.device | None = None) -> torch.Tensor:
    """``LongTensor[grid_h * grid_w, 2]`` of ``(x, y)`` in row-major order (Gemma convention)."""
    ys, xs = torch.meshgrid(
        torch.arange(grid_h, device=device), torch.arange(grid_w, device=device), indexing="ij"
    )
    return torch.stack([xs, ys], dim=-1).reshape(-1, 2)


def patchify(image: torch.Tensor, patch_size: int = PATCH_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """Gemma/SigLIP2 patchification.

    Args:
        image: ``FloatTensor[3, H, W]``.
    Returns:
        patches ``[H/p * W/p, p*p*3]`` (row-major patches, channel-last inside a patch) and
        positions ``LongTensor[H/p * W/p, 2]`` as ``(x, y)``.
    """
    c, h, w = image.shape
    gh, gw = h // patch_size, w // patch_size
    x = image.reshape(c, gh, patch_size, gw, patch_size).permute(1, 3, 2, 4, 0)
    return x.reshape(gh * gw, -1), patch_positions(gh, gw, image.device)


def build_teacher_patches(
    images: Sequence[torch.Tensor], max_patches: int | None = None, patch_size: int = PATCH_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patchify + pad a list of ``[3, H_i, W_i]`` float images to ``[B, P, p*p*3]`` / ``[B, P, 2]``."""
    patches, positions = zip(*(patchify(im, patch_size) for im in images))
    p_max = max_patches or max(p.shape[0] for p in patches)
    pv = images[0].new_zeros((len(images), p_max, patches[0].shape[1]))
    pos = torch.full((len(images), p_max, 2), -1, dtype=torch.long, device=images[0].device)
    for i, (p, q) in enumerate(zip(patches, positions)):
        if p.shape[0] > p_max:
            raise ValueError(f"image {i} has {p.shape[0]} patches > max_patches={p_max}")
        pv[i, : p.shape[0]] = p
        pos[i, : q.shape[0]] = q
    return pv, pos


def unpatchify_batch(
    pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor, patch_size: int = PATCH_SIZE
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Inverse of :func:`build_teacher_patches` (used to feed Gemma-format inputs to students).

    Args:
        pixel_values: ``FloatTensor[B, P, p*p*3]``.
        pixel_position_ids: ``LongTensor[B, P, 2]`` with ``-1`` padding.
    Returns:
        dense ``FloatTensor[B, 3, H_max, W_max]`` (zero-padded), ``valid_token_mask[B, H_max/48,
        W_max/48]`` and per-image ``(h_i, w_i)`` in pixels.
    """
    b, _, d = pixel_values.shape
    valid = (pixel_position_ids >= 0).all(-1)
    sizes: list[tuple[int, int]] = []
    for i in range(b):
        pos = pixel_position_ids[i][valid[i]]
        gw = int(pos[:, 0].max()) + 1
        gh = int(pos[:, 1].max()) + 1
        if gh * gw != int(valid[i].sum()):
            raise ValueError("patches of image %d do not form a full rectangle" % i)
        sizes.append((gh * patch_size, gw * patch_size))
    h_max = max(s[0] for s in sizes)
    w_max = max(s[1] for s in sizes)
    out = pixel_values.new_zeros((b, 3, h_max, w_max))
    mask = torch.zeros((b, h_max // TOKEN_STRIDE, w_max // TOKEN_STRIDE), dtype=torch.bool, device=out.device)
    for i, (h, w) in enumerate(sizes):
        gh, gw = h // patch_size, w // patch_size
        pv = pixel_values[i][valid[i]]
        pos = pixel_position_ids[i][valid[i]]
        grid = pv.new_zeros((gh, gw, d))
        grid[pos[:, 1], pos[:, 0]] = pv  # scatter by explicit positions (order-agnostic)
        img = grid.reshape(gh, gw, patch_size, patch_size, 3).permute(4, 0, 2, 1, 3).reshape(3, h, w)
        out[i, :, :h, :w] = img
        mask[i, : h // TOKEN_STRIDE, : w // TOKEN_STRIDE] = True
    return out, mask, sizes


# --------------------------------------------------------------------------------------
# Student (dense) view
# --------------------------------------------------------------------------------------
def pad_dense(images: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Stack ``[3, h_i, w_i]`` images (multiples of 48) into a zero-padded dense batch.

    Returns:
        ``images[B, 3, H, W]``, ``valid_token_mask[B, H/48, W/48]``, ``image_token_lengths``.
    """
    for im in images:
        if im.shape[-2] % TOKEN_STRIDE or im.shape[-1] % TOKEN_STRIDE:
            raise ValueError(f"image size {tuple(im.shape[-2:])} is not a multiple of {TOKEN_STRIDE}")
    h_max = max(im.shape[-2] for im in images)
    w_max = max(im.shape[-1] for im in images)
    out = images[0].new_zeros((len(images), 3, h_max, w_max))
    mask = torch.zeros((len(images), h_max // TOKEN_STRIDE, w_max // TOKEN_STRIDE), dtype=torch.bool, device=out.device)
    lengths = []
    for i, im in enumerate(images):
        h, w = im.shape[-2:]
        out[i, :, :h, :w] = im
        mask[i, : h // TOKEN_STRIDE, : w // TOKEN_STRIDE] = True
        lengths.append((h // TOKEN_STRIDE) * (w // TOKEN_STRIDE))
    return out, mask, lengths


def flatten_valid(grid: torch.Tensor, valid_token_mask: torch.Tensor) -> torch.Tensor:
    """``[B, Ht, Wt, D]`` -> ``[N, D]`` in the packed Gemma order (batch-major, row-major)."""
    if grid.shape[:3] != valid_token_mask.shape:
        raise ValueError(f"grid {tuple(grid.shape[:3])} vs mask {tuple(valid_token_mask.shape)}")
    return grid[valid_token_mask]


def exact_avg_pool(grid: torch.Tensor, factor: int) -> torch.Tensor:
    """Average-pool a channels-first map ``[B, C, H, W]`` by an exact integer factor (no resampling)."""
    if factor == 1:
        return grid
    if grid.shape[-2] % factor or grid.shape[-1] % factor:
        raise ValueError(f"spatial size {tuple(grid.shape[-2:])} not divisible by {factor}")
    return F.avg_pool2d(grid, kernel_size=factor, stride=factor)
