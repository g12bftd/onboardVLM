"""``timm`` CNN / hybrid students that emit one token per 48-px Gemma cell.

Mapping channels to 768 is easy: a 1x1 conv (linear) or MLP head does it. The hard part is the
spatial grid. A per-token head cannot change how many tokens there are or where they sit, and
standard backbones stride by 32 (or 16/64), which does not tile Gemma's 48-px cells. Each
``alignment`` mode solves this differently:

* ``input_rescale`` (default; the backbone is used unmodified with its pretrained weights): the input
  is resized by ``r / 48`` (r = the backbone's final stride, e.g. 2/3 for stride 32). Every 48-px cell
  then becomes an ``r``-px cell, and the backbone's native stride-``r`` output lands *exactly* on the
  Gemma token grid (cell ``(i, j)`` <-> original pixels ``[48i, 48i + 48)``).
* ``stem_stride3``: the first stride-2 layer becomes stride 3 (3 x 2^4 = 48); full-resolution input.
* ``tap16_conv3``: stride-16 stage + a learned 3x3/stride-3 conv; full-resolution input.
* ``resample``: for fixed-resolution models (EfficientFormerV2, LeViT, EfficientViT-M ...), whose
  internals hard-code the token count and fail at any other input size. Each image is resized to the
  model's native square input, and a learned *position-query* attention head (one query per target
  48-px cell, keyed by the backbone's feature positions) produces the variable-length token sequence.
  The resize distorts the aspect ratio and limits spatial resolution, but it works for any backbone.

Every mode is verified at construction on non-square probes. CNNs are not batch invariant (neighbours'
zero padding leaks through the receptive field), so the grafted VLM runs them per image; ``resample``
crops each image before resizing and is batch invariant.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from onboard_vlm.geometry import exact_avg_pool, flatten_valid
from onboard_vlm.interfaces import TOKEN_STRIDE, StudentVisionOutput, VisionStudent
from onboard_vlm.registry import register_student
from onboard_vlm.students.common import InputNorm, upsample_mask
from onboard_vlm.utils.logging import describe_module, get_logger

log = get_logger(__name__)
_PROBE = 96
ALIGNMENTS = ("input_rescale", "stem_stride3", "tap16_conv3", "resample")


def _first_downsample_modules(model: nn.Module, probe: int = _PROBE) -> list[tuple[str, nn.Module]]:
    """Modules (Conv2d / pooling) that consume the full-resolution input and emit 1/2 resolution."""
    hits: list[tuple[str, nn.Module]] = []
    hooks = []

    def make_hook(name: str, mod: nn.Module):
        def hook(_m, inputs, output):
            x = inputs[0] if inputs else None
            if isinstance(x, torch.Tensor) and isinstance(output, torch.Tensor) and x.dim() == 4 and output.dim() == 4:
                if tuple(x.shape[-2:]) == (probe, probe) and tuple(output.shape[-2:]) == (probe // 2, probe // 2):
                    hits.append((name, mod))

        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            hooks.append(mod.register_forward_hook(make_hook(name, mod)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, probe, probe))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return hits


def set_first_stride_to_3(model: nn.Module) -> list[str]:
    """In-place stride surgery; returns the names of the modified modules."""
    hits = _first_downsample_modules(model)
    if not hits:
        raise RuntimeError("no stride-2 first downsampling layer (patchify stem?); use alignment=input_rescale or tap16_conv3")
    for _, mod in hits:
        mod.stride = (3, 3) if isinstance(mod, nn.Conv2d) else 3
    return [n for n, _ in hits]


def fourier_features(xy: torch.Tensor, num_freq: int) -> torch.Tensor:
    """``[..., 2]`` coordinates in [0, 1] -> ``[..., 4 * num_freq]`` sin/cos features."""
    freqs = 2.0 ** torch.arange(num_freq, device=xy.device, dtype=xy.dtype) * math.pi
    ang = xy[..., None] * freqs  # [..., 2, F]
    return torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)


def _grid_centers(h: int, w: int, device: torch.device) -> torch.Tensor:
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # row-major, (x, y)


class PositionalResampler(nn.Module):
    """Cross-attention head: one query per target cell (at its normalised image position) attends to
    the backbone's feature map (keyed by its own normalised positions). Produces any number of
    tokens in row-major order, so it can map a fixed ``h x w`` grid onto the image's ``gh x gw`` grid."""

    def __init__(self, in_dim: int, out_dim: int, dim: int = 256, heads: int = 4, depth: int = 2, num_freq: int = 16):
        super().__init__()
        self.num_freq = num_freq
        self.kv_proj = nn.Linear(in_dim, dim)
        self.pos = nn.Linear(4 * num_freq, dim)
        self.query = nn.Parameter(torch.zeros(dim))
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "norm_q": nn.LayerNorm(dim), "norm_kv": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, heads, batch_first=True),
                "norm_mlp": nn.LayerNorm(dim), "mlp": nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)),
            }))
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))

    def forward(self, feat: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        """feat ``[C, h, w]`` of one image -> ``[gh * gw, out_dim]``."""
        c, h, w = feat.shape
        kv_pos = self.pos(fourier_features(_grid_centers(h, w, feat.device), self.num_freq))
        kv = (self.kv_proj(feat.flatten(1).T) + kv_pos)[None]
        q = (self.query + self.pos(fourier_features(_grid_centers(*grid_hw, feat.device), self.num_freq)))[None]
        for layer in self.layers:
            q = q + layer["attn"](layer["norm_q"](q), layer["norm_kv"](kv), layer["norm_kv"](kv), need_weights=False)[0]
            q = q + layer["mlp"](layer["norm_mlp"](q))
        return self.out(q[0])


@register_student("timm_spatial")
class TimmSpatialStudent(VisionStudent):
    """Any ``timm`` CNN / hybrid as a Gemma-token student.

    Args:
        model_name: ``timm`` model name, optionally with a pretrained tag.
        out_dim: output token width (768).
        pretrained: load ``timm`` pretrained weights.
        alignment: one of :data:`ALIGNMENTS` (see module docstring).
        head: ``"linear"`` (1x1 conv) or ``"mlp"`` (1x1 -> GELU -> 1x1); unused for ``resample``.
        input_norm: ``"imagenet"`` = the checkpoint's own mean/std (timm ``pretrained_cfg``), or ``"gemma"``.
        fixed_input_size: ``resample`` only; ``(H, W)`` of the backbone input (default: pretrained_cfg input_size).
        resampler: ``resample`` only; kwargs of :class:`PositionalResampler` (``dim``, ``heads``, ``depth``, ``num_freq``).
        model_kwargs: extra kwargs for ``timm.create_model`` (e.g. ``drop_path_rate``).

    ``layer_dims`` / ``return_layers`` index backbone stages (0 = earliest); ``resample`` exposes none.
    """

    def __init__(
        self,
        model_name: str = "mobilenetv4_conv_medium",
        out_dim: int = 768,
        pretrained: bool = True,
        alignment: str = "input_rescale",
        head: str = "linear",
        input_norm: str = "imagenet",
        fixed_input_size: Sequence[int] | None = None,
        resampler: dict | None = None,
        model_kwargs: dict | None = None,
        freeze_backbone: bool = False,
    ):
        """``freeze_backbone``: keep the pretrained backbone frozen (incl. BatchNorm statistics) and train
        only the alignment / projection head - the "off-the-shelf encoder + projector" baseline."""
        super().__init__()
        import timm

        if alignment not in ALIGNMENTS:
            raise ValueError(f"alignment must be one of {ALIGNMENTS}, got {alignment!r}")
        self.model_name, self.alignment, self.out_dim = model_name, alignment, out_dim
        self.batch_invariant = alignment == "resample"
        kw = dict(model_kwargs or {})
        backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, **kw)
        reductions = list(backbone.feature_info.reduction())
        channels = list(backbone.feature_info.channels())
        self.modified_modules: list[str] = []
        self.downsample: nn.Module = nn.Identity()
        self.scale = 1.0
        if alignment == "input_rescale":
            self.scale = reductions[-1] / TOKEN_STRIDE  # e.g. 32 / 48 = 2/3
            final_ch = channels[-1]
        elif alignment == "stem_stride3":
            self.modified_modules = set_first_stride_to_3(backbone)
            reductions = [r // 2 * 3 for r in reductions]
            final_ch = channels[-1]
        elif alignment == "tap16_conv3":
            if 16 not in reductions:
                raise ValueError(f"{model_name} has no stride-16 stage (reductions={reductions})")
            keep = reductions.index(16) + 1
            backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=tuple(range(keep)), **kw)
            reductions, channels = reductions[:keep], channels[:keep]
            final_ch = max(channels[-1] * 2, 256)
            self.downsample = nn.Sequential(nn.Conv2d(channels[-1], final_ch, 3, stride=3, bias=False), nn.BatchNorm2d(final_ch), nn.GELU())
        else:  # resample
            cfg = getattr(backbone, "pretrained_cfg", None) or {}
            self.fixed_input_size = tuple(fixed_input_size or cfg.get("input_size", (3, 224, 224))[-2:])
            final_ch = channels[-1]
        self.backbone = backbone
        self.channels = channels
        self.stage_strides = reductions
        self.input_norm = InputNorm(input_norm, *self._pretrained_mean_std(backbone, input_norm))
        if alignment == "resample":
            self.head = PositionalResampler(final_ch, out_dim, **(resampler or {}))
            self.layer_dims: dict[int, int] = {}
        else:
            self.head = nn.Conv2d(final_ch, out_dim, 1) if head == "linear" else nn.Sequential(nn.Conv2d(final_ch, 2 * out_dim, 1), nn.GELU(), nn.Conv2d(2 * out_dim, out_dim, 1))
            if head not in ("linear", "mlp"):
                raise ValueError(f"unknown head {head!r}")
            # stage s (stride r_s on the backbone input) is pooled by r_final / r_s onto the token grid
            self.layer_dims = {i: c for i, (c, r) in enumerate(zip(channels, reductions)) if self._pool_factor(r) is not None}
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self._verify_alignment()
        log.info("     student timm_spatial %s [%s%s] -> %s", model_name, alignment,
                 f", input x{self.scale:.3g}" if alignment == "input_rescale" else "", describe_module(self))

    @staticmethod
    def _pretrained_mean_std(backbone: nn.Module, mode: str):
        cfg = getattr(backbone, "pretrained_cfg", None) or {}
        if mode == "imagenet" and "mean" in cfg and "std" in cfg:
            return tuple(cfg["mean"]), tuple(cfg["std"])
        return None, None

    def _pool_factor(self, r: int) -> int | None:
        """Integer pooling factor that maps a stride-``r`` backbone map onto the token grid."""
        token_stride = TOKEN_STRIDE * self.scale  # token cell size in backbone-input pixels
        f = token_stride / r
        return int(round(f)) if f >= 1 and abs(f - round(f)) < 1e-6 else None

    def _nchw(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Some timm models (e.g. MambaOut) return NHWC feature maps."""
        out = []
        for f, c in zip(feats, self.channels):
            out.append(f.permute(0, 3, 1, 2) if f.shape[1] != c and f.shape[-1] == c else f)
        return out

    def _backbone_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.scale != 1.0:
            # Downscale with an area (box) filter: for factor 2/3 each output pixel averages a fixed
            # pattern of input pixels with period 3, so 48-px cells map exactly onto 32-px cells with no
            # boundary special cases (shift-equivariant, tested). Upscaling (stride-64 backbones) is bilinear.
            size = (round(x.shape[-2] * self.scale), round(x.shape[-1] * self.scale))
            x = F.interpolate(x, size=size, mode="area") if self.scale < 1 else F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        feats = self._nchw(self.backbone(x))
        return self.downsample(feats[-1]), feats

    def train(self, mode: bool = True) -> "TimmSpatialStudent":
        super().train(mode)
        if getattr(self, "freeze_backbone", False):
            self.backbone.eval()  # frozen BatchNorm statistics
        return self

    @torch.no_grad()
    def _verify_alignment(self) -> None:
        was_training = self.training
        self.eval()
        try:
            for h, w in ((96, 144), (240, 48), (48, 336)):
                m = torch.ones(1, h // TOKEN_STRIDE, w // TOKEN_STRIDE, dtype=torch.bool)
                out = self.forward(torch.zeros(1, 3, h, w), m, return_layers=sorted(self.layer_dims))
                if out.tokens.shape[0] != m.sum():
                    raise RuntimeError(f"{self.model_name}/{self.alignment}: {out.tokens.shape[0]} tokens for a {tuple(m.shape[1:])} grid")
        finally:
            self.train(was_training)

    def forward(self, images: torch.Tensor, valid_token_mask: torch.Tensor, return_layers: Sequence[int] = ()) -> StudentVisionOutput:
        """images ``[B, 3, H, W]`` in [0, 1]; valid_token_mask ``[B, H/48, W/48]``."""
        unknown = set(return_layers) - set(self.layer_dims)
        if unknown:
            raise KeyError(f"stages {sorted(unknown)} not available; valid: {sorted(self.layer_dims)}")
        lengths = valid_token_mask.flatten(1).sum(1).tolist()
        x = self.input_norm(images, upsample_mask(valid_token_mask, TOKEN_STRIDE)[:, None])
        if self.alignment == "resample":
            return self._forward_resample(x, valid_token_mask, lengths)
        y, feats = self._backbone_maps(x)
        if tuple(y.shape[-2:]) != tuple(valid_token_mask.shape[-2:]):
            raise RuntimeError(f"spatial drift: map {tuple(y.shape[-2:])} vs token grid {tuple(valid_token_mask.shape[-2:])}")
        tokens = flatten_valid(self.head(y).permute(0, 2, 3, 1), valid_token_mask)
        inter = {}
        for i in return_layers:
            f = exact_avg_pool(feats[i], self._pool_factor(self.stage_strides[i]))
            inter[i] = flatten_valid(f.permute(0, 2, 3, 1), valid_token_mask)
        return StudentVisionOutput(tokens=tokens, intermediate_tokens=inter, image_token_lengths=lengths)

    def _forward_resample(self, x: torch.Tensor, valid_token_mask: torch.Tensor, lengths: list[int]) -> StudentVisionOutput:
        crops, grids = [], []
        for i in range(x.shape[0]):
            gh, gw = int(valid_token_mask[i].any(1).sum()), int(valid_token_mask[i].any(0).sum())
            crop = x[i : i + 1, :, : gh * TOKEN_STRIDE, : gw * TOKEN_STRIDE]
            crops.append(F.interpolate(crop, size=self.fixed_input_size, mode="bilinear", align_corners=False, antialias=True))
            grids.append((gh, gw))
        feat = self._nchw(self.backbone(torch.cat(crops)))[-1]
        tokens = torch.cat([self.head(feat[i], grids[i]) for i in range(len(grids))])
        return StudentVisionOutput(tokens=tokens, intermediate_tokens={}, image_token_lengths=lengths)
