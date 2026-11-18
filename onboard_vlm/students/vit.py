"""ViT students that emit Gemma-4-aligned soft tokens.

``TokenViT`` is the shared implementation; the registered students only differ in defaults
and trunk:

* ``CompactViT``      16-px patches + 3x3 average pooling (Gemma's own tokenisation), 2D-RoPE.
* ``CoarsePatchViT``  48-px patches straight from the stem (9x fewer tokens in every layer).
* ``MultiplexedViT``  (see ``multiplexed_vit.py``) MiniViT-style weight-multiplexed trunk.

All variants mask padded patches out of attention, so an image's tokens do not depend on the
other images in the batch (``batch_invariant = True``).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from onboard_vlm.geometry import exact_avg_pool, flatten_valid, patch_positions
from onboard_vlm.interfaces import TOKEN_STRIDE, StudentVisionOutput, VisionStudent
from onboard_vlm.registry import register_student
from onboard_vlm.students.common import (
    Block,
    InputNorm,
    RoPE2D,
    drop_path_schedule,
    trunc_normal_init,
    upsample_mask,
)


class ConvStem(nn.Module):
    """Overlapping conv stem with strides (2, 2, 2, 2, 3) = 48 that keeps zero-padding semantics.

    After every stage the activation is multiplied by the validity mask at that resolution,
    so padded regions look exactly like conv zero-padding and the stem stays batch invariant.
    """

    STRIDES = (2, 2, 2, 2, 3)

    def __init__(self, out_dim: int, widths: Sequence[int] = (24, 48, 96, 192, 384)):
        super().__init__()
        chans = [3, *widths]
        layers = []
        for i, s in enumerate(self.STRIDES):
            last = i == len(self.STRIDES) - 1
            layers.append(
                nn.Sequential(
                    nn.Conv2d(chans[i], chans[i + 1] if not last else out_dim, 3, stride=s, padding=1),
                    nn.Identity() if last else nn.BatchNorm2d(chans[i + 1]),
                    nn.Identity() if last else nn.GELU(),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, valid_token_mask: torch.Tensor) -> torch.Tensor:
        stride = 1
        for layer, s in zip(self.layers, self.STRIDES):
            stride *= s
            x = layer(x)
            m = upsample_mask(valid_token_mask, TOKEN_STRIDE // stride)[:, None]
            x = x * m.to(x.dtype)
        return x


class TokenViT(VisionStudent):
    """Patch-embed -> Transformer trunk (2D-RoPE) -> k x k average pool -> projection to ``out_dim``.

    Args:
        out_dim: ``D_target`` (the deployment VLM's vision hidden size, e.g. 768 for E2B/E4B).
        embed_dim, depth, num_heads, mlp_ratio: trunk size.
        patch_size: 16 (pooled by 3) or 48 (no pooling); ``patch_size * pool_kernel == 48``.
        stem: ``"linear"`` (non-overlapping patch conv) or ``"conv"`` (overlapping, 48-px only).
        head: ``"linear"`` or ``"mlp"`` projection from ``embed_dim`` to ``out_dim``.
        input_norm: see :class:`InputNorm`.
        grad_checkpointing: activation checkpointing on trunk blocks.
    """

    batch_invariant = True

    def __init__(
        self,
        out_dim: int = 768,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        stem: str = "linear",
        head: str = "linear",
        qk_norm: bool = True,
        rope_theta: float = 100.0,
        drop_path: float = 0.0,
        input_norm: str = "gemma",
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        if TOKEN_STRIDE % patch_size:
            raise ValueError(f"patch_size must divide {TOKEN_STRIDE}, got {patch_size}")
        self.patch_size = patch_size
        self.pool_kernel = TOKEN_STRIDE // patch_size
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.grad_checkpointing = grad_checkpointing
        self.input_norm = InputNorm(input_norm)
        if stem == "linear":
            self.stem = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        elif stem == "conv":
            if patch_size != TOKEN_STRIDE:
                raise ValueError("conv stem is only defined for patch_size=48")
            self.stem = ConvStem(embed_dim)
        else:
            raise ValueError(f"unknown stem {stem!r}")
        self.rope = RoPE2D(embed_dim // num_heads, rope_theta)
        self.blocks = self._build_trunk(embed_dim, depth, num_heads, mlp_ratio, qk_norm, drop_path)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        if head == "linear":
            self.head = nn.Linear(embed_dim, out_dim)
        elif head == "mlp":
            self.head = nn.Sequential(nn.Linear(embed_dim, 2 * out_dim), nn.GELU(approximate="tanh"), nn.Linear(2 * out_dim, out_dim))
        else:
            raise ValueError(f"unknown head {head!r}")
        self.layer_dims = {i: embed_dim for i in range(depth)}
        trunc_normal_init(self)

    # Overridden by MultiplexedViT.
    def _build_trunk(self, dim: int, depth: int, heads: int, mlp_ratio: float, qk_norm: bool, drop_path: float) -> nn.Module:
        dpr = drop_path_schedule(drop_path, depth)
        return nn.ModuleList([Block(dim, heads, mlp_ratio, qk_norm, dpr[i]) for i in range(depth)])

    def _run_trunk(self, x: torch.Tensor, key_mask: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], grid_hw: tuple[int, int], return_layers: set[int]) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        feats: dict[int, torch.Tensor] = {}
        for i, blk in enumerate(self.blocks):
            if self.grad_checkpointing and self.training:
                x = checkpoint(blk, x, key_mask, rope, use_reentrant=False)
            else:
                x = blk(x, key_mask, rope)
            if i in return_layers:
                feats[i] = x
        return x, feats

    def _to_tokens(self, x: torch.Tensor, grid_hw: tuple[int, int], valid_token_mask: torch.Tensor) -> torch.Tensor:
        """``[B, Np, D]`` patch tokens -> ``[N, D]`` pooled, unpadded soft tokens."""
        b, _, d = x.shape
        g = x.transpose(1, 2).reshape(b, d, *grid_hw)
        g = exact_avg_pool(g, self.pool_kernel).permute(0, 2, 3, 1)
        return flatten_valid(g, valid_token_mask)

    def forward(
        self,
        images: torch.Tensor,
        valid_token_mask: torch.Tensor,
        return_layers: Sequence[int] = (),
    ) -> StudentVisionOutput:
        """images ``[B, 3, H, W]`` in [0, 1]; valid_token_mask ``[B, H/48, W/48]``."""
        unknown = set(return_layers) - set(self.layer_dims)
        if unknown:
            raise KeyError(f"layers {sorted(unknown)} not available; valid: 0..{self.depth - 1}")
        pixel_mask = upsample_mask(valid_token_mask, TOKEN_STRIDE)[:, None]
        x = self.input_norm(images, pixel_mask)
        x = self.stem(x, valid_token_mask) if isinstance(self.stem, ConvStem) else self.stem(x)
        b, d, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, Np, D]
        key_mask = upsample_mask(valid_token_mask, self.pool_kernel).reshape(b, -1)
        rope = self.rope(patch_positions(gh, gw, x.device))
        x, feats = self._run_trunk(x, key_mask, rope, (gh, gw), set(return_layers))
        x = self.norm(x)
        tokens = self.head(self._to_tokens(x, (gh, gw), valid_token_mask))
        inter = {i: self._to_tokens(f, (gh, gw), valid_token_mask) for i, f in feats.items()}
        lengths = valid_token_mask.flatten(1).sum(1).tolist()
        return StudentVisionOutput(tokens=tokens, intermediate_tokens=inter, image_token_lengths=lengths)


@register_student("compact_vit")
class CompactViT(TokenViT):
    """16-px patches + 3x3 pooling + 2D-RoPE: a shrunk replica of Gemma 4's own tokenisation."""

    def __init__(self, **kwargs):
        kwargs.setdefault("patch_size", 16)
        if kwargs["patch_size"] != 16:
            raise ValueError("CompactViT uses patch_size=16; use CoarsePatchViT for 48-px patches")
        super().__init__(**kwargs)


@register_student("coarse_patch_vit")
class CoarsePatchViT(TokenViT):
    """48-px patches straight from the stem: one trunk token per soft token (9x fewer than CompactViT)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("patch_size", TOKEN_STRIDE)
        if kwargs["patch_size"] != TOKEN_STRIDE:
            raise ValueError("CoarsePatchViT uses patch_size=48")
        super().__init__(**kwargs)
