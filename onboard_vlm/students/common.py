"""Shared building blocks for ViT-style students: input normalisation, axial 2D-RoPE,
masked attention (with optional MiniViT head mixing) and pre-norm blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InputNorm(nn.Module):
    """Maps ``[0, 1]`` pixels to the student's expected input range.

    Normalisation lives *inside* the student so that the training path (dense ``[0, 1]`` images
    from the collator) and the grafted inference path (``[0, 1]`` patches from the HF processor)
    are identical. Padded pixels are re-zeroed after normalisation so that zero-padding semantics
    are preserved for convolutional stems.

    Args:
        mode: ``"gemma"`` (``2x - 1`` like ``Gemma4VisionPatchEmbedder``), ``"imagenet"`` or ``"none"``.
    """

    def __init__(self, mode: str = "gemma", mean: tuple[float, ...] | None = None, std: tuple[float, ...] | None = None):
        super().__init__()
        if mode == "gemma":
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        elif mode == "imagenet":
            mean, std = mean or IMAGENET_MEAN, std or IMAGENET_STD
        elif mode == "none":
            mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        else:
            raise ValueError(f"unknown input_norm {mode!r}")
        self.mode = mode
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor, pixel_mask: torch.Tensor | None = None) -> torch.Tensor:
        """images ``[B, 3, H, W]`` in [0, 1]; pixel_mask ``[B, 1, H, W]`` (True = valid)."""
        x = (images - self.mean.to(images.dtype)) / self.std.to(images.dtype)
        if pixel_mask is not None:
            x = x * pixel_mask.to(x.dtype)
        return x


def upsample_mask(valid_token_mask: torch.Tensor, factor: int) -> torch.Tensor:
    """``[B, Ht, Wt]`` -> ``[B, Ht*factor, Wt*factor]`` by exact repetition."""
    if factor == 1:
        return valid_token_mask
    return valid_token_mask.repeat_interleave(factor, dim=1).repeat_interleave(factor, dim=2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class RoPE2D(nn.Module):
    """Axial 2D rotary embedding (same scheme as ``Gemma4VisionRotaryEmbedding``).

    The head dimension is split in two halves: the first half is rotated by the patch's x
    coordinate, the second by its y coordinate, each with standard ``rotate_half`` RoPE.
    """

    def __init__(self, head_dim: int, theta: float = 100.0):
        super().__init__()
        if head_dim % 4:
            raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}")
        half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, 2, dtype=torch.float32) / half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions ``LongTensor[N, 2]`` (x, y) -> cos, sin ``[N, head_dim]``."""
        freqs = positions[..., None].float() * self.inv_freq  # [N, 2, half/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [N, 2, half]
        emb = emb.flatten(-2)  # [N, head_dim]  (x-half then y-half)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x ``[B, H, N, Dh]``; cos/sin ``[N, Dh]``; rotates the x- and y-halves independently."""
    half = x.shape[-1] // 2
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    xs, ys = x[..., :half], x[..., half:]
    xs = xs * cos[:, :half] + _rotate_half(xs) * sin[:, :half]
    ys = ys * cos[:, half:] + _rotate_half(ys) * sin[:, half:]
    return torch.cat([xs, ys], dim=-1)


class Attention(nn.Module):
    """Multi-head self-attention with 2D-RoPE, QK-norm and a key padding mask.

    ``head_mix`` (``[H, H]``) implements the MiniViT attention transformation
    ``A'_g = sum_h F[g, h] A_h`` *after* softmax without materialising attention maps: we compute
    ``A_h @ V_all`` for every head with one SDPA call (value dim ``H * Dh``) and contract with ``F``.
    """

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        head_mix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x ``[B, N, D]``; key_mask ``[B, N]`` (True = valid key)."""
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = apply_rope_2d(self.q_norm(q), *rope)
        k = apply_rope_2d(self.k_norm(k), *rope)
        attn_mask = key_mask[:, None, None, :]
        if head_mix is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # [B, H, N, Dh]
        else:
            v_all = v.permute(0, 2, 1, 3).reshape(b, 1, n, d).expand(b, self.num_heads, n, d)
            per_head = F.scaled_dot_product_attention(q, k, v_all, attn_mask=attn_mask)  # [B, H, N, H*Dh]
            per_head = per_head.view(b, self.num_heads, n, self.num_heads, self.head_dim)
            # out[b, g] = sum_h F[g, h] * (A_h @ V_g)
            out = torch.einsum("gh,bhngd->bgnd", head_mix.to(per_head.dtype), per_head)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(1 - self.p)
        return x * keep / (1 - self.p)


class Block(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qk_norm: bool = True, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qk_norm)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], **_: object) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), key_mask, rope))
        return x + self.drop_path(self.mlp(self.norm2(x)))


def trunc_normal_init(module: nn.Module, std: float = 0.02) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def drop_path_schedule(rate: float, depth: int) -> list[float]:
    return [rate * i / max(depth - 1, 1) for i in range(depth)]


def num_params(module: nn.Module) -> int:
    """Parameter count with shared tensors counted once (``parameters()`` de-duplicates)."""
    return sum(p.numel() for p in module.parameters())


__all__ = [
    "Attention",
    "Block",
    "DropPath",
    "InputNorm",
    "Mlp",
    "RMSNorm",
    "RoPE2D",
    "apply_rope_2d",
    "drop_path_schedule",
    "num_params",
    "trunc_normal_init",
    "upsample_mask",
]
