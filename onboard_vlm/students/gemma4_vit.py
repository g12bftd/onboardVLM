"""A smaller Gemma 4 vision encoder: the original ``transformers`` ``Gemma4VisionModel`` (unchanged
code) built from a smaller ``Gemma4VisionConfig``. Depth, width, heads, MLP size and position-table
size come from YAML; weights are freshly initialised by transformers' own initialiser.

The student consumes exactly the teacher's input (Gemma patch sequence) and returns unpadded
``[N, hidden]`` soft tokens; a linear projection maps them to the 768-d Gemma 4 token space when
``hidden_size != 768`` so the student can replace the vision tower of E2B / E4B.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from onboard_vlm.geometry import build_teacher_patches
from onboard_vlm.interfaces import PATCH_SIZE, POOLING_KERNEL, TOKEN_STRIDE, StudentVisionOutput, VisionStudent
from onboard_vlm.registry import register_student
from onboard_vlm.utils.logging import describe_module, get_logger

log = get_logger(__name__)


@register_student("gemma4_vit")
class Gemma4ViTStudent(VisionStudent):
    """Gemma 4-architecture student (teacher: 16 layers, hidden 768, 12 heads x 64, MLP 3072).

    Args:
        out_dim: output token width (768 = Gemma 4 E2B / E4B vision width).
        hidden_size, num_hidden_layers, intermediate_size (``None`` -> 4 x hidden): trunk size.
        num_attention_heads: ``None`` -> ``hidden_size // head_dim``.
        head_dim: per-head width (teacher: 64).
        position_embedding_size: learned positions per axis (teacher: 10240; 256 x 16 px = 4096 px).
        use_clipped_linears: Gemma's activation clamps (teacher: true; inert at random init).
        attn_implementation: ``sdpa`` / ``eager``.
        grad_checkpointing: activation checkpointing inside the encoder.
    """

    batch_invariant = True
    accepts_teacher_patches = True

    def __init__(
        self,
        out_dim: int = 768,
        hidden_size: int = 384,
        num_hidden_layers: int = 8,
        num_attention_heads: int | None = None,
        head_dim: int = 64,
        intermediate_size: int | None = None,
        position_embedding_size: int = 256,
        use_clipped_linears: bool = False,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 100.0,
        attention_dropout: float = 0.0,
        attn_implementation: str = "sdpa",
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        heads = num_attention_heads or hidden_size // head_dim
        cfg = Gemma4VisionConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size or 4 * hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=heads,
            num_key_value_heads=heads,
            head_dim=head_dim,
            patch_size=PATCH_SIZE,
            pooling_kernel_size=POOLING_KERNEL,
            position_embedding_size=position_embedding_size,
            use_clipped_linears=use_clipped_linears,
            rms_norm_eps=rms_norm_eps,
            attention_dropout=attention_dropout,
        )
        cfg.rope_parameters = {**(cfg.rope_parameters or {}), "rope_theta": rope_theta}
        cfg._attn_implementation = attn_implementation
        self.config = cfg
        self.max_patches_per_axis = position_embedding_size  # checked against the input domain by the tasks
        self.encoder = Gemma4VisionModel(cfg)
        self.out_dim = out_dim
        self.layer_dims = {i: hidden_size for i in range(num_hidden_layers)}
        self.head: nn.Module = nn.Linear(hidden_size, out_dim, bias=False) if hidden_size != out_dim else nn.Identity()
        if grad_checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        log.info("     student gemma4_vit: %d layers x %d (heads %d x %d, MLP %d) -> %s", num_hidden_layers, hidden_size, heads, head_dim, cfg.intermediate_size, describe_module(self))

    def forward(
        self,
        images: torch.Tensor | None = None,
        valid_token_mask: torch.Tensor | None = None,
        return_layers: Sequence[int] = (),
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_position_ids: torch.Tensor | None = None,
    ) -> StudentVisionOutput:
        """Gemma-format patches (``pixel_values [B, P, 768]``, ``pixel_position_ids [B, P, 2]``), as the
        teacher receives them, or a dense batch (``images [B, 3, H, W]`` in [0, 1] + ``valid_token_mask``)."""
        from onboard_vlm.teachers.gemma4 import pool_patch_states

        unknown = set(return_layers) - set(self.layer_dims)
        if unknown:
            raise KeyError(f"layers {sorted(unknown)} not available; valid 0..{len(self.layer_dims) - 1}")
        if pixel_values is None:
            crops = []
            for i in range(images.shape[0]):
                gh, gw = int(valid_token_mask[i].any(1).sum()), int(valid_token_mask[i].any(0).sum())
                crops.append(images[i, :, : gh * TOKEN_STRIDE, : gw * TOKEN_STRIDE])
            pixel_values, pixel_position_ids = build_teacher_patches(crops)
        if int(pixel_position_ids.max()) >= self.max_patches_per_axis:
            raise ValueError(f"input has {int(pixel_position_ids.max()) + 1} patches on one axis; this student's position table "
                             f"covers {self.max_patches_per_axis} (outside its declared input domain)")
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
        n_tok = ((pixel_position_ids >= 0).all(-1).sum(-1) // POOLING_KERNEL**2).tolist()
        inter = {i: pool_patch_states(h, pixel_position_ids)[0] for i, h in captured.items()}
        return StudentVisionOutput(tokens=self.head(out), intermediate_tokens=inter, image_token_lengths=n_tok)


@register_student("gemma4_truncated")
class Gemma4TruncatedTeacher(Gemma4ViTStudent):
    """Baseline: the original Gemma 4 vision tower cut to its first ``num_layers`` layers, followed by
    its own pooler. No new parameters; usable as-is (zero-training baseline) or as an initialisation.

    Args:
        num_layers: layers kept (teacher: 16).
        model_id / revision / tiny: the teacher checkpoint (``tiny`` = random tiny Gemma 4 for tests).
        load_weights: copy the teacher weights (``False`` when re-loaded from an export).
        trainable: leave the parameters trainable (when used as an initialisation).
    """

    def __init__(self, num_layers: int = 8, out_dim: int = 768, model_id: str = "google/gemma-4-E2B-it", revision: str | None = None,
                 tiny: dict | None = None, load_weights: bool = True, trainable: bool = False, attn_implementation: str = "sdpa", **_: object):
        nn.Module.__init__(self)
        import copy

        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        if tiny is not None:
            from onboard_vlm.utils.tiny import tiny_gemma4_config

            vcfg = copy.deepcopy(tiny_gemma4_config(**tiny).vision_config)
        else:
            from transformers import AutoConfig

            from onboard_vlm.teachers.gemma4 import resolve_revision

            vcfg = copy.deepcopy(AutoConfig.from_pretrained(model_id, revision=resolve_revision(model_id, revision)).vision_config)
        total = vcfg.num_hidden_layers
        if not 1 <= num_layers <= total:
            raise ValueError(f"num_layers must be in [1, {total}], got {num_layers}")
        if out_dim != vcfg.hidden_size:
            raise ValueError(f"a truncated teacher emits {vcfg.hidden_size}-d tokens, got out_dim={out_dim}")
        vcfg.num_hidden_layers = num_layers
        vcfg._attn_implementation = attn_implementation
        self.config = vcfg
        self.encoder = Gemma4VisionModel(vcfg)
        self.out_dim = vcfg.hidden_size
        self.layer_dims = {i: vcfg.hidden_size for i in range(num_layers)}
        self.max_patches_per_axis = vcfg.position_embedding_size
        self.head = nn.Identity()
        if load_weights:
            from onboard_vlm.teachers.gemma4 import Gemma4Teacher

            teacher = Gemma4Teacher(model_id, revision, load="vision", dtype="float32", tiny=tiny)
            keep = {k: v for k, v in teacher.vision_tower.state_dict().items()
                    if not (k.startswith("encoder.layers.") and int(k.split(".")[2]) >= num_layers)}
            self.encoder.load_state_dict(keep, strict=True)
            del teacher
        self.requires_grad_(trainable)
        log.info("     truncated teacher: first %d of %d layers -> %s", num_layers, total, describe_module(self))
