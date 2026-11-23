"""A *smaller Gemma 4 vision encoder*: the exact ``transformers`` ``Gemma4VisionModel`` code
(learned 2D position table, axial 2D-RoPE, QK/V-norms, sandwich RMSNorms, gated-GELU MLP,
position-based 3x3 pooling with ``sqrt(D)`` scaling, optional clipped linears) with every size
knob exposed to YAML, plus optional initialisation from the exact E2B/E4B teacher weights.

Initialisation (``init: teacher``) follows *weight selection* (Xu et al., "Initializing Models with
Larger Ones", ICLR 2024): pick a subset of teacher layers, and inside every kept tensor select a
consistent subset of residual dimensions, attention heads (whole 64-d heads, so RoPE / QK-norm
semantics are preserved) and MLP channels. If the student keeps the teacher's width and all
layers, the copy is exact and the student reproduces the teacher bit-for-bit (tested).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn

from onboard_vlm.geometry import build_teacher_patches
from onboard_vlm.interfaces import PATCH_SIZE, TOKEN_STRIDE, StudentVisionOutput, VisionStudent
from onboard_vlm.registry import register_student
from onboard_vlm.teachers.hub import restore_rope_fp32

TEACHER_HIDDEN = 768
TEACHER_HEAD_DIM = 64


def _no_pool_pooler(config: Any) -> nn.Module:
    """``Gemma4VisionPooler`` for ``pooling_kernel_size=1``.

    transformers' pooler returns the *padding* mask (True = pad) when no pooling happens but the
    *valid* mask (True = token) after pooling, so ``Gemma4VisionModel`` keeps zero tokens without
    pooling. Gemma 4 itself always pools 3x3; this only matters for 48-px-patch students.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPooler

    class NoPoolPooler(Gemma4VisionPooler):
        def forward(self, hidden_states, pixel_position_ids, padding_positions, output_length=None):
            out, _ = super().forward(hidden_states, pixel_position_ids, padding_positions, output_length)
            return out, ~padding_positions

    return NoPoolPooler(config)


def select_indices(n: int, k: int, mode: str | Sequence[int], scores: torch.Tensor | None = None) -> torch.Tensor:
    """``k`` of ``n`` indices (sorted): ``uniform`` (evenly spaced, incl. both ends), ``first``,
    ``last``, ``magnitude`` (top-k of ``scores``) or an explicit list."""
    if not isinstance(mode, str):
        idx = torch.as_tensor(list(mode), dtype=torch.long)
        if len(idx) != k or idx.min() < 0 or idx.max() >= n or len(set(idx.tolist())) != k:
            raise ValueError(f"explicit selection {list(mode)} must hold {k} distinct indices in [0, {n})")
        return idx.sort().values
    if k > n:
        raise ValueError(f"cannot select {k} of {n}")
    if mode == "uniform":
        return torch.linspace(0, n - 1, k).round().long() if k > 1 else torch.tensor([n - 1])
    if mode == "first":
        return torch.arange(k)
    if mode == "last":
        return torch.arange(n - k, n)
    if mode == "magnitude":
        if scores is None:
            raise ValueError("magnitude selection needs scores")
        return scores.topk(k).indices.sort().values
    raise ValueError(f"unknown selection mode {mode!r}")


def inherit_teacher_weights(
    student: nn.Module,
    teacher_sd: dict[str, torch.Tensor],
    layer_selection: str | Sequence[int] = "uniform",
    width_selection: str = "uniform",
    head_selection: str = "uniform",
    mlp_selection: str = "uniform",
    inherit_clip_ranges: bool = True,
) -> dict[str, Any]:
    """Copy (a selection of) teacher weights into a smaller ``Gemma4VisionModel`` in place."""
    cfg = student.config
    t_layers = 1 + max(int(k.split(".")[2]) for k in teacher_sd if k.startswith("encoder.layers."))
    t_hidden = teacher_sd["patch_embedder.input_proj.weight"].shape[0]
    hd = teacher_sd["encoder.layers.0.self_attn.q_norm.weight"].shape[0]  # 64 for E2B/E4B
    t_heads = teacher_sd["encoder.layers.0.self_attn.q_proj.linear.weight"].shape[0] // hd
    t_inter = teacher_sd["encoder.layers.0.mlp.gate_proj.linear.weight"].shape[0]
    if cfg.head_dim != hd:
        raise ValueError(f"teacher init keeps whole heads: head_dim must equal the teacher's ({hd}), got {cfg.head_dim}")
    if cfg.num_key_value_heads != cfg.num_attention_heads:
        raise ValueError("teacher init needs num_key_value_heads == num_attention_heads")
    layers = select_indices(t_layers, cfg.num_hidden_layers, layer_selection)
    if width_selection == "magnitude":  # residual dims ranked by mean |RMSNorm gain| over all layers
        gains = torch.stack([teacher_sd[f"encoder.layers.{i}.input_layernorm.weight"].float().abs() for i in range(t_layers)]).mean(0)
        hid = select_indices(t_hidden, cfg.hidden_size, "magnitude", gains)
    else:
        hid = select_indices(t_hidden, cfg.hidden_size, width_selection)
    n_pos = cfg.position_embedding_size
    new: dict[str, torch.Tensor] = {
        "patch_embedder.input_proj.weight": teacher_sd["patch_embedder.input_proj.weight"][hid],
        "patch_embedder.position_embedding_table": teacher_sd["patch_embedder.position_embedding_table"][:, :n_pos][:, :, hid],
    }
    for s, t in enumerate(layers.tolist()):
        tp, sp = f"encoder.layers.{t}.", f"encoder.layers.{s}."
        w_o = teacher_sd[tp + "self_attn.o_proj.linear.weight"].float()
        head_scores = w_o.view(t_hidden, t_heads, hd).norm(dim=(0, 2))
        heads = select_indices(t_heads, cfg.num_attention_heads, head_selection, head_scores)
        rows = (heads[:, None] * hd + torch.arange(hd)[None]).reshape(-1)
        mlp_scores = teacher_sd[tp + "mlp.down_proj.linear.weight"].float().norm(dim=0) * teacher_sd[tp + "mlp.up_proj.linear.weight"].float().norm(dim=1)
        inter = select_indices(t_inter, cfg.intermediate_size, mlp_selection, mlp_scores)
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            new[sp + n + ".weight"] = teacher_sd[tp + n + ".weight"][hid]
        for n in ("q_proj", "k_proj", "v_proj"):
            new[sp + f"self_attn.{n}.linear.weight"] = teacher_sd[tp + f"self_attn.{n}.linear.weight"][rows][:, hid]
        new[sp + "self_attn.o_proj.linear.weight"] = teacher_sd[tp + "self_attn.o_proj.linear.weight"][hid][:, rows]
        for n in ("q_norm", "k_norm"):
            new[sp + f"self_attn.{n}.weight"] = teacher_sd[tp + f"self_attn.{n}.weight"]
        for n in ("gate_proj", "up_proj"):
            new[sp + f"mlp.{n}.linear.weight"] = teacher_sd[tp + f"mlp.{n}.linear.weight"][inter][:, hid]
        new[sp + "mlp.down_proj.linear.weight"] = teacher_sd[tp + "mlp.down_proj.linear.weight"][hid][:, inter]
        if cfg.use_clipped_linears and inherit_clip_ranges:
            for k, v in teacher_sd.items():
                if k.startswith(tp) and k.rsplit(".", 1)[-1] in ("input_min", "input_max", "output_min", "output_max"):
                    new[sp + k[len(tp) :]] = v
    own = student.state_dict()
    missing = [k for k in own if k not in new and not k.endswith(("input_min", "input_max", "output_min", "output_max"))]
    if missing:
        raise KeyError(f"teacher init does not cover {missing[:5]}")
    with torch.no_grad():
        for k, v in new.items():
            if own[k].shape != v.shape:
                raise ValueError(f"{k}: student {tuple(own[k].shape)} vs selected {tuple(v.shape)}")
            own[k].copy_(v.to(own[k].dtype))
    return {"layers": layers.tolist(), "hidden_idx": hid}


@register_student("gemma4_vit")
class Gemma4ViTStudent(VisionStudent):
    """Configurable Gemma 4-architecture student.

    Args:
        out_dim: ``D_target`` (768 = the teacher width, required to graft into E2B/E4B).
        hidden_size, num_hidden_layers, intermediate_size: trunk size (``intermediate_size=None`` -> 4x hidden).
        num_attention_heads: ``None`` -> ``hidden_size // head_dim``.
        head_dim: 64 (teacher) keeps RoPE / QK-norm geometry; required for ``init: teacher``.
        position_embedding_size: learned positions per axis (teacher: 10240). 256 covers up to
            4,096 px per side (every 16:9 / 4:3 frame at <= 1120 tokens) at 1/40 of the parameters.
        use_clipped_linears: Gemma's activation clamps (teacher: on); with teacher init the clamp
            ranges are inherited when ``inherit_clip_ranges``.
        init: ``"random"`` or ``"teacher"`` (weight selection from ``teacher_id``'s exact weights).
        layer_selection / width_selection / head_selection / mlp_selection: see :func:`select_indices`.
        output_proj: ``"auto"`` (linear iff ``hidden_size != out_dim``), ``"linear"`` or ``"none"``.
        patch_size: 16 (Gemma's own patches, 3x3 pooled; consumes the teacher's patch sequence and
            supports ``init: teacher``) or 48 (one patch per soft token, no pooling: ~9x fewer tokens
            in every layer, random init only). Glasses-class budgets usually need 48.
        grad_checkpointing: activation checkpointing in the encoder.
    """

    batch_invariant = True

    def __init__(
        self,
        out_dim: int = TEACHER_HIDDEN,
        hidden_size: int = 384,
        num_hidden_layers: int = 8,
        num_attention_heads: int | None = None,
        head_dim: int = TEACHER_HEAD_DIM,
        intermediate_size: int | None = None,
        position_embedding_size: int = 256,
        use_clipped_linears: bool = False,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 100.0,
        attention_dropout: float = 0.0,
        init: str = "random",
        teacher_id: str = "google/gemma-4-E2B-it",
        teacher_state_dict: dict[str, torch.Tensor] | None = None,
        layer_selection: str | Sequence[int] = "uniform",
        width_selection: str = "uniform",
        head_selection: str = "uniform",
        mlp_selection: str = "uniform",
        inherit_clip_ranges: bool = True,
        output_proj: str = "auto",
        attn_implementation: str = "sdpa",
        grad_checkpointing: bool = False,
        patch_size: int = PATCH_SIZE,
    ):
        super().__init__()
        if patch_size not in (PATCH_SIZE, TOKEN_STRIDE):
            raise ValueError(f"patch_size must be {PATCH_SIZE} or {TOKEN_STRIDE}, got {patch_size}")
        if patch_size != PATCH_SIZE and init == "teacher":
            raise ValueError("init: teacher needs patch_size=16 (the teacher's patch embedding)")
        self.patch_size = patch_size
        self.pool = TOKEN_STRIDE // patch_size
        self.accepts_teacher_patches = patch_size == PATCH_SIZE
        from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        heads = num_attention_heads or hidden_size // head_dim
        if heads * head_dim != hidden_size and init == "teacher":
            raise ValueError("teacher init needs hidden_size == num_attention_heads * head_dim")
        cfg = Gemma4VisionConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size or 4 * hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=heads,
            num_key_value_heads=heads,
            head_dim=head_dim,
            patch_size=patch_size,
            pooling_kernel_size=TOKEN_STRIDE // patch_size,
            position_embedding_size=position_embedding_size,
            use_clipped_linears=use_clipped_linears,
            rms_norm_eps=rms_norm_eps,
            attention_dropout=attention_dropout,
            standardize=False,
        )
        cfg.rope_parameters = {**(cfg.rope_parameters or {}), "rope_theta": rope_theta}
        cfg._attn_implementation = attn_implementation
        self.encoder = Gemma4VisionModel._from_config(cfg, dtype=torch.float32)
        if self.pool == 1:
            self.encoder.pooler = _no_pool_pooler(cfg)
        self.config = cfg
        self.out_dim = out_dim
        self.layer_dims = {i: hidden_size for i in range(num_hidden_layers)}
        need_proj = output_proj == "linear" or (output_proj == "auto" and hidden_size != out_dim)
        self.head: nn.Module = nn.Linear(hidden_size, out_dim, bias=False) if need_proj else nn.Identity()
        if not need_proj and hidden_size != out_dim:
            raise ValueError(f"output_proj='none' requires hidden_size == out_dim ({hidden_size} vs {out_dim})")
        self.init_info: dict[str, Any] = {"init": init}
        if init == "teacher":
            sd = teacher_state_dict if teacher_state_dict is not None else self._exact_teacher_sd(teacher_id)
            info = inherit_teacher_weights(self.encoder, sd, layer_selection, width_selection, head_selection, mlp_selection, inherit_clip_ranges)
            self.init_info.update(layers=info["layers"], teacher_id=teacher_id)
            if isinstance(self.head, nn.Linear):  # map selected dims back to their teacher positions
                with torch.no_grad():
                    self.head.weight.zero_()
                    t_hidden = sd["patch_embedder.input_proj.weight"].shape[0]
                    if out_dim == t_hidden:
                        scale = math.sqrt(t_hidden / hidden_size)  # undo the pooler's sqrt(hidden) scaling
                        self.head.weight[info["hidden_idx"], torch.arange(hidden_size)] = scale
                    else:
                        nn.init.normal_(self.head.weight, std=hidden_size**-0.5)
        elif init != "random":
            raise ValueError(f"init must be 'random' or 'teacher', got {init!r}")
        if grad_checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    @staticmethod
    def _exact_teacher_sd(teacher_id: str) -> dict[str, torch.Tensor]:
        from onboard_vlm.teachers.hub import load_exact_vision_tower

        tower, _, _, _ = load_exact_vision_tower(teacher_id)
        return {k: v.float() for k, v in tower.state_dict().items()}

    def _apply(self, fn, *args, **kwargs):  # keep RoPE frequencies in fp32 under .to(bf16)/.half()
        out = super()._apply(fn, *args, **kwargs)
        restore_rope_fp32(self)
        return out

    def to_gemma_inputs(self, images: torch.Tensor, valid_token_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense padded batch -> Gemma patch sequence (each image's own rectangle only)."""
        crops = []
        for i in range(images.shape[0]):
            gh = int(valid_token_mask[i].any(1).sum())
            gw = int(valid_token_mask[i].any(0).sum())
            crops.append(images[i, :, : gh * TOKEN_STRIDE, : gw * TOKEN_STRIDE])
        return build_teacher_patches(crops, patch_size=self.patch_size)

    def forward(
        self,
        images: torch.Tensor | None = None,
        valid_token_mask: torch.Tensor | None = None,
        return_layers: Sequence[int] = (),
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_position_ids: torch.Tensor | None = None,
    ) -> StudentVisionOutput:
        """Dense input (``images [B, 3, H, W]`` in [0, 1] + ``valid_token_mask [B, H/48, W/48]``) or,
        for 16-px students, Gemma-format patches (``pixel_values [B, P, 768]``, ``pixel_position_ids``)."""
        unknown = set(return_layers) - set(self.layer_dims)
        if unknown:
            raise KeyError(f"layers {sorted(unknown)} not available; valid 0..{len(self.layer_dims) - 1}")
        if pixel_values is not None:
            if not self.accepts_teacher_patches:
                raise ValueError("48-px students take dense images, not 16-px Gemma patches")
            return self.forward_gemma(pixel_values, pixel_position_ids, return_layers)
        pv, pos = self.to_gemma_inputs(images, valid_token_mask)
        return self.forward_gemma(pv, pos, return_layers)

    def forward_gemma(self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor, return_layers: Sequence[int] = ()) -> StudentVisionOutput:
        """Forward on Gemma-format patches (``[B, P, 768]``, ``[B, P, 2]``), as the teacher receives them."""
        from onboard_vlm.teachers.gemma4 import pool_patch_states

        pixel_values = pixel_values.to(next(self.encoder.parameters()).dtype)
        captured: dict[int, torch.Tensor] = {}
        hooks = [
            self.encoder.encoder.layers[i].register_forward_hook(lambda _m, _i, o, i=i: captured.__setitem__(i, o[0] if isinstance(o, tuple) else o))
            for i in return_layers
        ]
        try:
            out = self.encoder(pixel_values=pixel_values, pixel_position_ids=pixel_position_ids).last_hidden_state
        finally:
            for h in hooks:
                h.remove()
        n_tok = ((pixel_position_ids >= 0).all(-1).sum(-1) // self.pool**2).tolist()
        inter = {i: pool_patch_states(h, pixel_position_ids, self.pool)[0] for i, h in captured.items()}
        return StudentVisionOutput(tokens=self.head(out), intermediate_tokens=inter, image_token_lengths=n_tok)
