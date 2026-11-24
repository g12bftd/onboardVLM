"""Deterministic image degradations modelling smart-glasses capture (WearVQA's failure modes:
blur, low light, rotation, occlusion, cut-off, unmagnified/small subjects) plus JPEG artefacts.

Degradations are part of the *single* per-image transform: they are sampled once, applied to the
uint8 image after the Gemma resize, and only then branched into the teacher and student views.
The teacher therefore labels exactly the degraded pixels the student sees (consistent teaching,
Beyer et al. 2022), which transfers the teacher's robustness instead of teaching invariance.
All randomness lives in the sampled parameters, so cached teacher features replay exactly.

Cut-off and unmagnified are produced by the geometric crop (``crop_scale``) and by sampling
low token budgets; the photometric ops are below.
"""

from __future__ import annotations

import io
import math
import random
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as tvF

# op name -> default probability (used when a config lists an op without a probability)
DEFAULT_PROBS = {"motion_blur": 0.1, "defocus_blur": 0.1, "low_light": 0.1, "rotate": 0.1, "occlusion": 0.05, "jpeg": 0.1}


def sample_degradations(cfg: Mapping[str, Any] | None, rng: random.Random, hw: tuple[int, int]) -> tuple[tuple[str, tuple], ...]:
    """Sample a tuple of ``(op, params)`` from ``cfg = {op: probability}`` (or ``{op: {p: .., ...}}``)."""
    if not cfg:
        return ()
    h, w = hw
    ops: list[tuple[str, tuple]] = []
    for op, spec in cfg.items():
        p = spec if isinstance(spec, (int, float)) else float((spec or {}).get("p", DEFAULT_PROBS.get(op, 0.1)))
        if op not in DEFAULT_PROBS:
            raise ValueError(f"unknown degradation {op!r}; available: {sorted(DEFAULT_PROBS)}")
        if rng.random() >= p:
            continue
        if op == "motion_blur":
            ops.append((op, (rng.choice([5, 7, 9, 11, 15]), round(rng.uniform(0, 180), 1))))
        elif op == "defocus_blur":
            ops.append((op, (round(rng.uniform(1.0, 4.0), 2),)))
        elif op == "low_light":  # gamma darkening, gain, signal-dependent noise (seeded)
            ops.append((op, (round(rng.uniform(1.5, 3.0), 2), round(rng.uniform(0.3, 0.7), 2), round(rng.uniform(0.01, 0.05), 3), rng.randrange(2**31))))
        elif op == "rotate":  # head tilt; 90-degree sensor rotations are common in VizWiz / WearVQA
            ops.append((op, (rng.choice([round(rng.uniform(-25, 25), 1), 90.0, -90.0, 180.0]),)))
        elif op == "occlusion":  # finger / strap over the lens: an ellipse near a border
            rh, rw = rng.uniform(0.15, 0.4) * h, rng.uniform(0.15, 0.4) * w
            cy = rng.choice([0.0, h]) if rng.random() < 0.5 else rng.uniform(0, h)
            cx = rng.uniform(0, w) if cy in (0.0, h) else rng.choice([0.0, w])
            ops.append((op, (round(cy, 1), round(cx, 1), round(rh, 1), round(rw, 1), rng.randrange(20, 90))))
        elif op == "jpeg":
            ops.append((op, (rng.randrange(15, 60),)))
    return tuple(ops)


def _motion_kernel(length: int, angle: float) -> torch.Tensor:
    k = torch.zeros(length, length)
    c = (length - 1) / 2
    t = math.radians(angle)
    for i in range(length * 4):
        r = (i / (length * 4 - 1) - 0.5) * (length - 1)
        x, y = round(c + r * math.cos(t)), round(c + r * math.sin(t))
        k[y, x] = 1.0
    return k / k.sum()


def _disk_kernel(radius: float) -> torch.Tensor:
    r = int(math.ceil(radius))
    yy, xx = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij")
    k = ((yy**2 + xx**2) <= radius**2).float()
    return k / k.sum()


def _filter(img: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    x = img.float()[None]
    pad = k.shape[-1] // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    y = F.conv2d(x, k[None, None].expand(3, 1, -1, -1), groups=3)
    return y[0].round().clamp(0, 255).to(torch.uint8)


def apply_degradations(img: torch.Tensor, ops: tuple[tuple[str, tuple], ...]) -> torch.Tensor:
    """Apply sampled ops to ``UInt8[3, H, W]`` (or ``[T, 3, H, W]``); output keeps size and dtype."""
    if not ops:
        return img
    if img.dim() == 4:
        return torch.stack([apply_degradations(f, ops) for f in img])
    for op, a in ops:
        if op == "motion_blur":
            img = _filter(img, _motion_kernel(int(a[0]), float(a[1])))
        elif op == "defocus_blur":
            img = _filter(img, _disk_kernel(float(a[0])))
        elif op == "low_light":
            gamma, gain, sigma, seed = a
            x = (img.float() / 255.0).pow(gamma) * gain
            g = torch.Generator().manual_seed(int(seed))
            x = x + torch.randn(x.shape, generator=g) * sigma * x.clamp(min=1e-3).sqrt()
            img = (x.clamp(0, 1) * 255.0).round().to(torch.uint8)
        elif op == "rotate":
            img = tvF.rotate(img, float(a[0]), interpolation=tvF.InterpolationMode.BILINEAR, expand=False, fill=0)
        elif op == "occlusion":
            cy, cx, rh, rw, level = a
            h, w = img.shape[-2:]
            yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
            m = (((yy - cy) / max(rh, 1)) ** 2 + ((xx - cx) / max(rw, 1)) ** 2) <= 1.0
            img = img.clone()
            img[:, m] = int(level)
        elif op == "jpeg":
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(img.permute(1, 2, 0).numpy()).save(buf, format="JPEG", quality=int(a[0]))
            from onboard_vlm.data.lmdb_io import decode_image

            img = decode_image(buf.getvalue())
        else:
            raise ValueError(f"unknown degradation {op!r}")
    return img.contiguous()


def freeze(obj: Any) -> Any:
    """JSON lists -> nested tuples (hashable, comparable with sampled params)."""
    return tuple(freeze(o) for o in obj) if isinstance(obj, (list, tuple)) else obj
