"""Weight-multiplexed ViT (MiniViT, Zhang et al., CVPR 2022).

The trunk has ``num_shared_blocks`` Transformer blocks, each *stored once* and invoked
``steps_per_block`` consecutive times. Every invocation ("step") owns only lightweight
parameters, held in a separate ``nn.ModuleList``:

* its own LayerNorms (as in MiniViT, normalisation is never shared),
* ``"head_mix"``  an ``[H, H]`` linear transformation of the attention maps across heads,
  identity-initialised (MiniViT's attention transformation),
* ``"dwconv"``    a depthwise 3x3 conv over the 2D token grid inside the MLP hidden layer,
  zero-initialised residual (MiniViT's MLP transformation),
* ``"lowrank"``   a rank-``r`` residual adapter on the MLP branch (zero-initialised up-projection),
* ``"diag"``      per-channel scales on the attention and MLP branch outputs (init 1).

No parameter tensor is ever cloned: ``sum(p.numel() for p in model.parameters())`` counts every
shared block once, and ``named_parameters(remove_duplicate=False)`` yields the same total.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from onboard_vlm.registry import register_student
from onboard_vlm.students.common import Attention, DropPath, Mlp, drop_path_schedule
from onboard_vlm.students.vit import TokenViT

STEP_TRANSFORM_PRESETS: dict[str, tuple[str, ...]] = {
    "minivit": ("head_mix", "dwconv"),
    "lowrank": ("lowrank",),
    "diag": ("diag",),
    "none": (),
}
_VALID = {"head_mix", "dwconv", "lowrank", "diag"}


class SharedBlock(nn.Module):
    """The heavy, shared weights of one multiplexed block (attention + MLP; no norms)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qk_norm: bool):
        super().__init__()
        self.attn = Attention(dim, num_heads, qk_norm)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))


class StepTransform(nn.Module):
    """Per-step parameters of a multiplexed block (cheap; one instance per step)."""

    def __init__(self, dim: int, num_heads: int, hidden: int, components: Sequence[str], rank: int, drop_path: float):
        super().__init__()
        bad = set(components) - _VALID
        if bad:
            raise ValueError(f"unknown step transforms {sorted(bad)}; valid: {sorted(_VALID)}")
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path)
        self.head_mix = nn.Parameter(torch.eye(num_heads)) if "head_mix" in components else None
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden) if "dwconv" in components else None
        if "lowrank" in components:
            self.lr_down = nn.Linear(dim, rank, bias=False)
            self.lr_up = nn.Linear(rank, dim, bias=False)
        else:
            self.lr_down = self.lr_up = None
        if "diag" in components:
            self.gamma_attn = nn.Parameter(torch.ones(dim))
            self.gamma_mlp = nn.Parameter(torch.ones(dim))
        else:
            self.gamma_attn = self.gamma_mlp = None
        self.reset_to_identity()

    @torch.no_grad()
    def reset_to_identity(self) -> None:
        """Initialise so that the step is exactly plain weight sharing (identity transforms)."""
        if self.head_mix is not None:
            self.head_mix.copy_(torch.eye(self.head_mix.shape[0]))
        if self.dwconv is not None:
            nn.init.zeros_(self.dwconv.weight)
            nn.init.zeros_(self.dwconv.bias)
        if self.lr_up is not None:
            nn.init.trunc_normal_(self.lr_down.weight, std=0.02)
            nn.init.zeros_(self.lr_up.weight)
        if self.gamma_attn is not None:
            self.gamma_attn.fill_(1.0)
            self.gamma_mlp.fill_(1.0)

    def forward(
        self,
        shared: SharedBlock,
        x: torch.Tensor,
        key_mask: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        grid_hw: tuple[int, int],
    ) -> torch.Tensor:
        """One application of the shared block. x ``[B, N, D]`` with ``N = gh * gw``."""
        a = shared.attn(self.norm1(x), key_mask, rope, head_mix=self.head_mix)
        if self.gamma_attn is not None:
            a = a * self.gamma_attn
        x = x + self.drop_path(a)

        h = self.norm2(x)
        u = shared.mlp.act(shared.mlp.fc1(h))  # [B, N, hidden]
        if self.dwconv is not None:
            b, n, c = u.shape
            m = key_mask[..., None].to(u.dtype)
            g = (u * m).transpose(1, 2).reshape(b, c, *grid_hw)  # padded tokens act as zero padding
            u = u + self.dwconv(g).flatten(2).transpose(1, 2)
        m_out = shared.mlp.fc2(u)
        if self.lr_down is not None:
            m_out = m_out + self.lr_up(F.gelu(self.lr_down(h), approximate="tanh"))
        if self.gamma_mlp is not None:
            m_out = m_out * self.gamma_mlp
        return x + self.drop_path(m_out)


class MultiplexedTrunk(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qk_norm: bool,
        num_shared_blocks: int,
        steps_per_block: int,
        components: Sequence[str],
        rank: int,
        drop_path: float,
    ):
        super().__init__()
        self.steps_per_block = steps_per_block
        self.shared = nn.ModuleList([SharedBlock(dim, num_heads, mlp_ratio, qk_norm) for _ in range(num_shared_blocks)])
        dpr = drop_path_schedule(drop_path, num_shared_blocks * steps_per_block)
        hidden = int(dim * mlp_ratio)
        self.steps = nn.ModuleList(
            [StepTransform(dim, num_heads, hidden, components, rank, dpr[i]) for i in range(num_shared_blocks * steps_per_block)]
        )

    def shared_for_step(self, step_idx: int) -> SharedBlock:
        return self.shared[step_idx // self.steps_per_block]


@register_student("multiplexed_vit")
class MultiplexedViT(TokenViT):
    """MiniViT-style student. ``depth = num_shared_blocks * steps_per_block`` effective layers.

    Args:
        num_shared_blocks: distinct (stored) Transformer blocks.
        steps_per_block: consecutive re-uses of each shared block.
        step_transform: preset name (``minivit``, ``lowrank``, ``diag``, ``none``) or a list of
            components from ``{head_mix, dwconv, lowrank, diag}``.
        lowrank_rank: rank of the ``lowrank`` adapter.
        **kwargs: forwarded to :class:`TokenViT` (``depth`` is derived and must not be given).
    """

    def __init__(
        self,
        num_shared_blocks: int = 4,
        steps_per_block: int = 3,
        step_transform: str | Sequence[str] = "minivit",
        lowrank_rank: int = 16,
        **kwargs,
    ):
        if "depth" in kwargs:
            raise ValueError("MultiplexedViT derives depth = num_shared_blocks * steps_per_block")
        components = STEP_TRANSFORM_PRESETS[step_transform] if isinstance(step_transform, str) else tuple(step_transform)
        self._mux = dict(num_shared_blocks=num_shared_blocks, steps_per_block=steps_per_block, components=components, rank=lowrank_rank)
        super().__init__(depth=num_shared_blocks * steps_per_block, **kwargs)
        for step in self.blocks.steps:  # the base-class weight init must not perturb identity transforms
            step.reset_to_identity()

    def _build_trunk(self, dim, depth, heads, mlp_ratio, qk_norm, drop_path):  # type: ignore[override]
        return MultiplexedTrunk(dim, heads, mlp_ratio, qk_norm, drop_path=drop_path, **self._mux)

    def _run_trunk(self, x, key_mask, rope, grid_hw, return_layers):  # type: ignore[override]
        feats: dict[int, torch.Tensor] = {}
        trunk: MultiplexedTrunk = self.blocks  # type: ignore[assignment]
        for i, step in enumerate(trunk.steps):
            shared = trunk.shared_for_step(i)  # the SAME module object for consecutive steps
            if self.grad_checkpointing and self.training:
                x = checkpoint(step, shared, x, key_mask, rope, grid_hw, use_reentrant=False)
            else:
                x = step(shared, x, key_mask, rope, grid_hw)
            if i in return_layers:
                feats[i] = x
        return x, feats
