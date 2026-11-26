"""The Gemma 4 vision encoder itself, made smaller: depth and/or width reduced, nothing else changed.

The student is ``transformers``' ``Gemma4VisionModel`` (unchanged code) built from a copy of the
*teacher's own* ``vision_config``, in which only ``num_hidden_layers``, ``hidden_size``,
``intermediate_size`` and the head counts are changed. Everything else (patch embedder, 2-D RoPE and its
base, the learned position table and its size, clipped linears, RMSNorm epsilon, GELU-tanh, the pooler and
its ``sqrt(hidden)`` scaling) is the teacher's. At full size with ``init: teacher`` it *is* the teacher,
weight for weight (tested), which anchors every shrinking experiment at fidelity 1.

Initialisation (``init``):

* ``random``: ``transformers``' own initialiser, seeded by ``init_seed``;
* ``teacher``: weights inherited from the teacher.
  * Depth: a subset of the teacher's layers (``layer_selection``: ``uniform``, evenly spaced as in DistilBERT,
    Sanh et al. 2019; ``first``; ``last``; or an explicit list).
  * Width: weight selection (Xu et al., ICLR 2024, "Initializing Models with Larger Ones"). The same subset of
    residual channels is kept in every tensor of every layer (``width_selection``: ``uniform`` or ``first``),
    and whole attention heads and MLP units are kept, never parts of a head.
  * The output head embeds the kept channels back at their teacher positions, scaled by
    ``sqrt(768 / hidden)`` to undo the pooler's ``sqrt(hidden)``.

Weight multiplexing (``share_every = k``, MiniViT, Zhang et al., CVPR 2022): every ``k`` consecutive layers share one
set of attention and MLP weights (stored once); each layer keeps its own RMSNorms (as MiniViT keeps LayerNorms
unshared) and one per-layer transformation of its output, identity-initialised (``share_transform``: ``linear``, a
``hidden x hidden`` linear layer, or ``dwconv``, a 3x3 depthwise convolution over the patch grid, MiniViT's choice for
the MLP). With ``init: teacher``, a group's shared weights are those of the teacher layer its first layer inherits.

``source_layers[j]`` is the teacher layer that student layer ``j`` corresponds to (inherited with ``init:
teacher``, proportional depth otherwise). ``layer_map: auto`` uses it for the intermediate-layer losses.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch
from torch import nn

from onboard_vlm.registry import register_student
from onboard_vlm.students.gemma4_vit import Gemma4ViTStudent
from onboard_vlm.utils.logging import describe_module, get_logger

log = get_logger(__name__)


def select(n_from: int, n_to: int, how: str | Sequence[int]) -> list[int]:
    """``n_to`` of ``range(n_from)``: ``uniform`` (evenly spaced, both ends), ``first``, ``last`` or explicit."""
    if not isinstance(how, str):
        idx = [int(i) for i in how]
        if len(idx) != n_to or len(set(idx)) != n_to or not all(0 <= i < n_from for i in idx):
            raise ValueError(f"explicit selection {idx} must be {n_to} distinct indices in [0, {n_from})")
        return sorted(idx)
    if not 1 <= n_to <= n_from:
        raise ValueError(f"cannot select {n_to} of {n_from}")
    if how == "first":
        return list(range(n_to))
    if how == "last":
        return list(range(n_from - n_to, n_from))
    if how == "uniform":
        idx = np.round(np.linspace(0, n_from - 1, n_to)).astype(int).tolist()
        if len(set(idx)) != n_to:  # cannot happen for n_to <= n_from, kept as a guard
            raise ValueError(f"uniform selection of {n_to} of {n_from} is not injective")
        return idx
    raise ValueError(f"unknown selection {how!r} (uniform | first | last | explicit list)")


def teacher_vision_config(model_id: str, revision: str | None, tiny: dict | None):
    if tiny is not None:
        from onboard_vlm.utils.tiny import tiny_gemma4_config

        return copy.deepcopy(tiny_gemma4_config(**tiny).vision_config)
    from transformers import AutoConfig

    from onboard_vlm.teachers.gemma4 import resolve_revision

    return copy.deepcopy(AutoConfig.from_pretrained(model_id, revision=resolve_revision(model_id, revision)).vision_config)


@register_student("gemma4_shrink")
class Gemma4ShrinkStudent(Gemma4ViTStudent):
    """Gemma 4's vision encoder with fewer layers and / or a narrower residual stream (see module docstring).

    Args:
        num_hidden_layers: depth (teacher 16). ``None`` = the teacher's.
        hidden_size: width, a multiple of the teacher's head dim (64). ``None`` = the teacher's.
        intermediate_size: MLP width; ``None`` keeps the teacher's ratio (4x).
        init: ``random`` | ``teacher``.
        layer_selection / width_selection: which teacher layers / channels to inherit (``init: teacher``).
        init_seed: seed of the random initialisation (``init: random``, and the head when it is new).
        out_dim: output width (768: the Gemma 4 E2B / E4B vision width).
        model_id / revision / tiny: the teacher whose config (and weights) are used.
        load_teacher_weights: ``False`` when re-loading an export (weights come from the checkpoint).
    """

    def __init__(
        self,
        num_hidden_layers: int | None = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        init: str = "teacher",
        layer_selection: str | Sequence[int] = "uniform",
        width_selection: str = "uniform",
        init_seed: int = 0,
        out_dim: int = 768,
        model_id: str = "google/gemma-4-E2B-it",
        revision: str | None = None,
        tiny: dict | None = None,
        load_teacher_weights: bool = True,
        attn_implementation: str = "sdpa",
        grad_checkpointing: bool = False,
        share_every: int = 1,
        share_transform: str = "linear",
        **_: object,
    ):
        nn.Module.__init__(self)
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        if init not in ("random", "teacher"):
            raise ValueError(f"init must be random or teacher, got {init!r}")
        if int(share_every) < 1 or share_transform not in ("linear", "dwconv", "none"):
            raise ValueError(f"share_every >= 1 and share_transform in linear | dwconv | none, got {share_every}, {share_transform!r}")
        tcfg = teacher_vision_config(model_id, revision, tiny)
        T, H, D, hd = tcfg.num_hidden_layers, tcfg.hidden_size, tcfg.num_attention_heads, tcfg.head_dim
        if H != D * hd:
            raise ValueError(f"teacher hidden {H} != heads {D} x head_dim {hd}")
        L = num_hidden_layers or T
        h = hidden_size or H
        if h % hd or not hd <= h <= H or not 1 <= L <= T:
            raise ValueError(f"need 1 <= layers <= {T} and width a multiple of {hd} in [{hd}, {H}]; got {L} x {h}")
        heads = h // hd
        inter = intermediate_size or tcfg.intermediate_size * h // H
        cfg = copy.deepcopy(tcfg)  # the teacher's config: only the sizes below change
        cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = L, h, inter
        cfg.num_attention_heads = cfg.num_key_value_heads = heads
        cfg._attn_implementation = attn_implementation
        self.config = cfg
        self.out_dim = out_dim
        self.max_patches_per_axis = cfg.position_embedding_size
        self.layer_dims = {i: h for i in range(L)}
        self.init_mode = init

        with torch.random.fork_rng(devices=[]):  # the student's own seed, independent of the data order
            torch.manual_seed(init_seed)
            self.encoder = Gemma4VisionModel(cfg)
            self.head: nn.Module = nn.Linear(h, out_dim, bias=False) if h != out_dim else nn.Identity()
        self.share_every = int(share_every)
        self.share_transform = share_transform
        self.transforms = nn.ModuleList()
        if self.share_every > 1:
            self._multiplex(L, h)

        if init == "teacher":
            if out_dim != H:
                raise ValueError(f"init=teacher embeds the kept channels into the teacher's {H}-d token space; got out_dim={out_dim}")
            self.source_layers = select(T, L, layer_selection)
            self._sel = {"chans": select(H, h, width_selection), "heads": select(D, heads, width_selection),
                         "units": select(tcfg.intermediate_size, inter, width_selection), "head_dim": hd}
            if load_teacher_weights:
                from onboard_vlm.teachers.gemma4 import Gemma4Teacher

                teacher = Gemma4Teacher(model_id, revision, load="vision", dtype="float32", tiny=tiny)
                self.inherit_from(teacher.vision_tower.state_dict())
                del teacher
            if isinstance(self.head, nn.Linear):
                with torch.no_grad():
                    self.head.weight.zero_()
                    self.head.weight[self._sel["chans"], torch.arange(h)] = float(np.sqrt(out_dim / h))
        else:
            self.source_layers = [int(round((j + 1) * T / L)) - 1 for j in range(L)]  # proportional depth
        if grad_checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        share = f", {L // self.share_every + bool(L % self.share_every)} weight sets shared by {self.share_every} layers + {share_transform}" \
            if self.share_every > 1 else ""
        log.info("     student gemma4_shrink: %d of %d layers x %d of %d wide (heads %d, MLP %d)%s, init=%s%s -> %s", L, T, h, H, heads,
                 inter, share, init, f" (layers {self.source_layers}, width {width_selection})" if init == "teacher" else f" (seed {init_seed})",
                 describe_module(self))

    def _leader(self, j: int) -> int:
        """The layer whose attention / MLP weights layer ``j`` uses (itself when nothing is shared)."""
        return j - j % self.share_every

    def _multiplex(self, n_layers: int, hidden: int) -> None:
        layers = self.encoder.encoder.layers
        for j in range(n_layers):
            g = self._leader(j)
            if g != j:  # the same module objects: stored once, counted once, updated once
                layers[j].self_attn = layers[g].self_attn
                layers[j].mlp = layers[g].mlp
        if self.share_transform != "none":
            make = LinearTransform if self.share_transform == "linear" else GridDWConvTransform
            self.transforms = nn.ModuleList(make(hidden) for _ in range(n_layers))
            for j, layer in enumerate(layers):
                layer.register_forward_hook(lambda _m, _a, kw, out, j=j: self.transforms[j](out, kw.get("position_ids")), with_kwargs=True)

    def unique_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def inherit_from(self, src: dict[str, torch.Tensor]) -> None:
        """Copy the selected layers / channels / heads / MLP units of a teacher vision tower state dict."""
        hd = self._sel["head_dim"]
        c = torch.tensor(self._sel["chans"])
        rows = torch.tensor([i * hd + k for i in self._sel["heads"] for k in range(hd)])  # whole heads
        u = torch.tensor(self._sel["units"])
        new = {}
        for name, p in self.encoder.state_dict().items():
            key = name
            if name.startswith("encoder.layers."):
                parts = name.split(".")
                j = int(parts[2])
                shared = parts[3] in ("self_attn", "mlp")  # weights of the group's first layer (weight multiplexing)
                parts[2] = str(self.source_layers[self._leader(j) if shared else j])
                key = ".".join(parts)
            if key not in src:
                raise KeyError(f"student parameter {name} has no teacher counterpart {key}")
            t = src[key]
            if name.endswith(("q_proj.linear.weight", "k_proj.linear.weight", "v_proj.linear.weight")):
                t = t[rows][:, c]
            elif name.endswith("o_proj.linear.weight"):
                t = t[c][:, rows]
            elif name.endswith(("gate_proj.linear.weight", "up_proj.linear.weight")):
                t = t[u][:, c]
            elif name.endswith("down_proj.linear.weight"):
                t = t[c][:, u]
            elif name == "patch_embedder.input_proj.weight":
                t = t[c]
            elif name == "patch_embedder.position_embedding_table":
                t = t[..., c]
            elif t.ndim == 1 and t.shape[0] != p.shape[0]:  # RMSNorm scales of the residual stream
                t = t[c]
            if t.shape != p.shape:
                raise ValueError(f"{name}: selected teacher tensor {tuple(t.shape)} != student {tuple(p.shape)}")
            new[name] = t.to(p.dtype)
        self.encoder.load_state_dict(new, strict=True)


class LinearTransform(nn.Module):
    """Per-layer ``hidden x hidden`` linear map of a shared layer's output, identity-initialised."""

    def __init__(self, hidden: int):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(hidden))
            self.lin.bias.zero_()

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.lin(x)


class GridDWConvTransform(nn.Module):
    """Per-layer 3x3 depthwise convolution over each image's patch grid (MiniViT's MLP transformation),
    identity-initialised. Patches are placed on their ``(x, y)`` grid from ``position_ids``; padding stays as is."""

    def __init__(self, hidden: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(hidden, hidden, kernel, padding=kernel // 2, groups=hidden)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, 0, kernel // 2, kernel // 2] = 1.0
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        if position_ids is None:
            raise ValueError("GridDWConvTransform needs the patch positions (position_ids)")
        b, n, d = x.shape
        valid = (position_ids >= 0).all(-1)
        pos = position_ids.clamp(min=0)
        w, h = int(pos[..., 0].max()) + 1, int(pos[..., 1].max()) + 1
        idx = (pos[..., 1] * w + pos[..., 0]).long()[..., None].expand(-1, -1, d)
        grid = x.new_zeros(b, h * w, d).scatter_add_(1, idx, x * valid[..., None].to(x.dtype))
        y = self.conv(grid.transpose(1, 2).reshape(b, d, h, w)).reshape(b, d, h * w).transpose(1, 2)
        return torch.where(valid[..., None], torch.gather(y, 1, idx), x)
