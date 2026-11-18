"""``timm`` CNN / hybrid students with an exact 48x output stride.

Standard ``timm`` backbones downsample by 32. Resampling a stride-32 map onto the 48-px grid
would blur and shift features, so instead we change the network's stride:

* ``alignment="stem_stride3"`` (default): every conv/pool that maps the full-resolution input to
  1/2 resolution (all parallel branches of the first downsampling block, found by probing) is
  switched to stride 3, giving ``3 * 2**4 = 48`` at the last stage. Pretrained weights are kept;
  only the sampling grid of the first layer changes. Stage strides become 3/6/12/24/48.
* ``alignment="tap16_conv3"``: the backbone is truncated at its stride-16 stage and a learned
  ``3x3, stride 3`` conv (+BN+GELU) produces the stride-48 map (``16 * 3 = 48``).

Both are verified at construction time on non-square probes: the output grid must be exactly
``(H/48, W/48)``. A ``1x1`` conv projects to ``out_dim``. Intermediate stage features at stride
``s`` (with ``48 % s == 0``) are average-pooled by the exact factor ``48 / s``.

CNNs are not batch invariant (zero padding of neighbouring images leaks through the receptive
field), so ``batch_invariant = False``; the grafted VLM runs them per image at inference and the
training loader can bucket images by size (``data.samplers.SizeBucketBatchSampler``).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from onboard_vlm.geometry import exact_avg_pool, flatten_valid
from onboard_vlm.interfaces import TOKEN_STRIDE, StudentVisionOutput, VisionStudent
from onboard_vlm.registry import register_student
from onboard_vlm.students.common import InputNorm, upsample_mask

_PROBE = 96  # divisible by 2, 3, 16 and 48


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
        raise RuntimeError("could not locate the first stride-2 downsampling layer")
    for _, mod in hits:
        mod.stride = (3, 3) if isinstance(mod, nn.Conv2d) else 3
    return [n for n, _ in hits]


@register_student("timm_spatial")
class TimmSpatialStudent(VisionStudent):
    """Unified adapter for ``timm`` CNN / hybrid backbones (MobileNetV4, FastViT, EfficientViT, TinyViT, ...).

    Args:
        model_name: any ``timm`` model supporting ``features_only=True``.
        out_dim: ``D_target``.
        pretrained: load ``timm`` pretrained weights (needs HF Hub access).
        alignment: ``"stem_stride3"`` or ``"tap16_conv3"`` (see module docstring).
        head: ``"linear"`` (1x1 conv) or ``"mlp"`` (1x1 -> GELU -> 1x1).
        input_norm: ``"imagenet"`` for timm-pretrained weights, ``"gemma"`` otherwise.
        model_kwargs: extra kwargs for ``timm.create_model`` (e.g. ``drop_path_rate``).

    ``layer_dims`` / ``return_layers`` index the backbone's feature stages (0 = earliest).
    """

    batch_invariant = False

    def __init__(
        self,
        model_name: str = "mobilenetv4_conv_medium",
        out_dim: int = 768,
        pretrained: bool = True,
        alignment: str = "stem_stride3",
        head: str = "linear",
        input_norm: str = "imagenet",
        model_kwargs: dict | None = None,
    ):
        super().__init__()
        import timm

        self.model_name = model_name
        self.alignment = alignment
        self.out_dim = out_dim
        kw = dict(model_kwargs or {})
        backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, **kw)
        reductions = list(backbone.feature_info.reduction())
        channels = list(backbone.feature_info.channels())

        if alignment == "stem_stride3":
            self.modified_modules = set_first_stride_to_3(backbone)
            reductions = [r // 2 * 3 for r in reductions]  # first /2 became /3
            self.downsample: nn.Module = nn.Identity()
            final_ch = channels[-1]
        elif alignment == "tap16_conv3":
            if 16 not in reductions:
                raise ValueError(f"{model_name} has no stride-16 stage (reductions={reductions})")
            keep = reductions.index(16) + 1
            backbone = timm.create_model(
                model_name, pretrained=pretrained, features_only=True, out_indices=tuple(range(keep)), **kw
            )
            reductions, channels = reductions[:keep], channels[:keep]
            c16 = channels[-1]
            final_ch = max(c16 * 2, 256)
            self.downsample = nn.Sequential(
                nn.Conv2d(c16, final_ch, kernel_size=3, stride=3, padding=0, bias=False),
                nn.BatchNorm2d(final_ch),
                nn.GELU(),
            )
            self.modified_modules = []
        else:
            raise ValueError(f"unknown alignment {alignment!r}")

        self.backbone = backbone
        self.stage_strides = reductions
        self.layer_dims = {i: c for i, (c, r) in enumerate(zip(channels, reductions)) if TOKEN_STRIDE % r == 0}
        self.input_norm = InputNorm(input_norm, *self._pretrained_mean_std(backbone, input_norm))
        if head == "linear":
            self.head = nn.Conv2d(final_ch, out_dim, 1)
        elif head == "mlp":
            self.head = nn.Sequential(nn.Conv2d(final_ch, 2 * out_dim, 1), nn.GELU(), nn.Conv2d(2 * out_dim, out_dim, 1))
        else:
            raise ValueError(f"unknown head {head!r}")
        self._verify_alignment()

    @staticmethod
    def _pretrained_mean_std(backbone: nn.Module, mode: str) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
        cfg = getattr(backbone, "pretrained_cfg", None) or {}
        if mode == "imagenet" and "mean" in cfg and "std" in cfg:
            return tuple(cfg["mean"]), tuple(cfg["std"])
        return None, None

    def _final_map(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        feats = self.backbone(x)
        return self.downsample(feats[-1]), feats

    @torch.no_grad()
    def _verify_alignment(self) -> None:
        was_training = self.training
        self.eval()
        try:
            for h, w in ((96, 144), (240, 48), (48, 336)):
                y, feats = self._final_map(torch.zeros(1, 3, h, w))
                if tuple(y.shape[-2:]) != (h // TOKEN_STRIDE, w // TOKEN_STRIDE):
                    raise RuntimeError(
                        f"{self.model_name}/{self.alignment}: output grid {tuple(y.shape[-2:])} != "
                        f"{(h // TOKEN_STRIDE, w // TOKEN_STRIDE)} for input {(h, w)}; try the other alignment"
                    )
                for f, r in zip(feats, self.stage_strides):
                    if tuple(f.shape[-2:]) != (h // r, w // r):
                        raise RuntimeError(f"stage stride bookkeeping is wrong: {tuple(f.shape[-2:])} vs stride {r}")
        finally:
            self.train(was_training)

    def forward(
        self,
        images: torch.Tensor,
        valid_token_mask: torch.Tensor,
        return_layers: Sequence[int] = (),
    ) -> StudentVisionOutput:
        """images ``[B, 3, H, W]`` in [0, 1]; valid_token_mask ``[B, H/48, W/48]``."""
        unknown = set(return_layers) - set(self.layer_dims)
        if unknown:
            raise KeyError(f"stages {sorted(unknown)} not available; valid: {sorted(self.layer_dims)}")
        pixel_mask = upsample_mask(valid_token_mask, TOKEN_STRIDE)[:, None]
        x = self.input_norm(images, pixel_mask)
        y, feats = self._final_map(x)
        if tuple(y.shape[-2:]) != tuple(valid_token_mask.shape[-2:]):
            raise RuntimeError(f"spatial drift: map {tuple(y.shape[-2:])} vs token grid {tuple(valid_token_mask.shape[-2:])}")
        tokens = flatten_valid(self.head(y).permute(0, 2, 3, 1), valid_token_mask)
        inter = {}
        for i in return_layers:
            f = exact_avg_pool(feats[i], TOKEN_STRIDE // self.stage_strides[i])
            inter[i] = flatten_valid(f.permute(0, 2, 3, 1), valid_token_mask)
        lengths = valid_token_mask.flatten(1).sum(1).tolist()
        return StudentVisionOutput(tokens=tokens, intermediate_tokens=inter, image_token_lengths=lengths)
