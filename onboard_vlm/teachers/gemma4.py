"""The Gemma 4 E2B / E4B vision-encoder teacher, loaded byte-for-byte from the official checkpoints.

* ``encode_vision`` runs the exact ``Gemma4VisionModel`` (167.4M params, shared by E2B and E4B) and
  returns its unpadded ``[N, 768]`` soft tokens (post pooling, pre ``embed_vision``), plus
  intermediate encoder-layer outputs captured with forward hooks. Encoder layers operate on 16-px
  patches (9x more tokens than the output), so intermediate states are average-pooled 3x3 by patch
  position (the teacher's own pooling rule, without its ``sqrt(D)`` output scaling).
* ``project_to_lm`` applies the checkpoint's frozen ``embed_vision`` (scale-free RMSNorm + linear
  projection to 1536 (E2B) / 2560 (E4B)): exactly what the language model consumes.
* ``forward_vlm`` (``load="vlm"``) runs the full ``Gemma4ForConditionalGeneration``.

With ``compute_dtype="bfloat16"`` (default) the outputs are bitwise identical to the official
``get_image_features`` (``scripts/verify_gemma4_teacher.py --full_check``).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
from torch import nn

from onboard_vlm.interfaces import POOLING_KERNEL, TeacherInputs, TeacherVisionOutput, VisionTeacher
from onboard_vlm.registry import register_teacher
from onboard_vlm.teachers import hub
from onboard_vlm.teachers.hub import UnsupportedTeacherError

log = logging.getLogger(__name__)


class EncoderFreeTeacherError(UnsupportedTeacherError):
    """Raised for configs without a vision tower (e.g. the encoder-free ``gemma4_unified`` models)."""


def pool_patch_states(
    hidden: torch.Tensor, pixel_position_ids: torch.Tensor, k: int = POOLING_KERNEL
) -> tuple[torch.Tensor, list[int]]:
    """Average-pool patch-level states to soft tokens by patch position, then unpad.

    Matches ``Gemma4VisionPooler._avg_pool_by_positions`` (kernel index
    ``x // k + (W_patches // k) * (y // k)``) minus the ``sqrt(D)`` scaling.

    Args:
        hidden: ``[B, P, D]`` encoder-layer output (padded patch sequence).
        pixel_position_ids: ``[B, P, 2]`` (x, y), ``-1`` = padding.
    Returns:
        packed ``[N, D]`` tokens and per-image token counts.
    """
    b, p, d = hidden.shape
    valid = (pixel_position_ids >= 0).all(-1)  # [B, P]
    pos = pixel_position_ids.clamp(min=0)
    gw = pos[..., 0].amax(dim=-1, keepdim=True) + 1  # patches per row, per image
    idx = (pos[..., 0] // k) + (gw // k) * (pos[..., 1] // k)  # [B, P]
    n_tok = valid.sum(-1) // (k * k)  # [B]
    length = p // (k * k)
    out = hidden.new_zeros((b, length, d), dtype=torch.float32)
    src = hidden.float() * valid[..., None]
    out.scatter_add_(1, idx[..., None].expand(-1, -1, d), src)
    out = (out / (k * k)).to(hidden.dtype)
    tok_mask = torch.arange(length, device=hidden.device)[None] < n_tok[:, None]
    return out[tok_mask], n_tok.tolist()


def _random_init(config: Any, dtype: torch.dtype):
    """Randomly initialised tower + embedder with the real module classes (tests / dry runs only)."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder, Gemma4VisionModel

    if getattr(config, "vision_config", None) is None:
        raise EncoderFreeTeacherError("config has no vision_config (encoder-free model); it cannot be a vision teacher")
    tower = Gemma4VisionModel._from_config(config.vision_config, dtype=dtype).eval()
    embedder = Gemma4MultimodalEmbedder(config.vision_config, config.text_config).to(dtype)
    return tower, embedder


@register_teacher("gemma4")
class Gemma4Teacher(VisionTeacher, nn.Module):
    """Frozen Gemma 4 E2B / E4B vision-encoder teacher.

    Args:
        model_id: one of ``google/gemma-4-E2B-it`` (default), ``google/gemma-4-E4B-it``,
            ``google/gemma-4-E2B``, ``google/gemma-4-E4B``. The vision tower is identical for all four;
            the choice selects ``embed_vision`` (LM-space losses) and the VLM for ``load="vlm"``.
        revision: must equal the pinned commit (``None`` = pinned).
        load: ``"vision"`` (vision tower + embed_vision only, ~320 MB via range reads; Stage 1) or
            ``"vlm"`` (full model; Stage-2 logits).
        compute_dtype: ``"bfloat16"`` (bitwise-identical to the deployed VLM) or ``"float32"``
            (lossless upcast of the same bf16 weights, higher-precision arithmetic).
        verify: re-hash every loaded tensor against the bundled manifest (raises on any difference).
        tiny / config: random tiny model with the real module classes (tests / dry runs only).
    """

    def __init__(
        self,
        model_id: str | None = "google/gemma-4-E2B-it",
        revision: str | None = None,
        load: str = "vision",
        compute_dtype: str | torch.dtype = "bfloat16",
        verify: bool = True,
        cache_dir: str | None = None,
        attn_implementation: str = "sdpa",
        tiny: dict[str, Any] | None = None,
        config: Any | None = None,
        dtype: str | None = None,  # deprecated alias of compute_dtype (kept for old configs)
    ):
        nn.Module.__init__(self)
        if load not in ("vision", "vlm"):
            raise ValueError(f"load must be 'vision' or 'vlm', got {load!r}")
        compute = dtype or compute_dtype
        compute = compute if isinstance(compute, torch.dtype) else getattr(torch, compute)
        self.vlm: nn.Module | None = None
        self.spec: hub.TeacherSpec | None = None

        if tiny is not None or config is not None:
            from transformers import Gemma4Config

            from onboard_vlm.utils.tiny import tiny_gemma4_config

            cfg = tiny_gemma4_config(**tiny) if tiny is not None else (Gemma4Config(**config) if isinstance(config, dict) else config)
            if load == "vlm":
                from transformers import Gemma4ForConditionalGeneration

                if getattr(cfg, "vision_config", None) is None:
                    raise EncoderFreeTeacherError("config has no vision_config")
                self.vlm = Gemma4ForConditionalGeneration._from_config(cfg, dtype=compute).eval()
                self.vision_tower, embedder = self.vlm.model.vision_tower, self.vlm.model.embed_vision
            else:
                self.vision_tower, embedder = _random_init(cfg, compute)
            self.config, self.model_id = cfg, None
        else:
            if model_id is None:
                raise ValueError("model_id is required (or tiny=... for tests)")
            self.spec = hub.resolve_teacher(model_id, revision)
            self.model_id = self.spec.model_id
            if load == "vision":
                from transformers import AutoConfig

                tower, embed_w, _, _ = hub.load_exact_vision_tower(
                    model_id, self.spec.revision, cache_dir, compute_dtype=torch.bfloat16, verify=verify, attn_implementation=attn_implementation
                )
                path = hub.fetch_vision_weights(model_id, self.spec.revision, cache_dir, verify=False)
                self.config = AutoConfig.from_pretrained(str(path))
                from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder

                embedder = Gemma4MultimodalEmbedder(self.config.vision_config, self.config.text_config).to(torch.bfloat16)
                embedder.embedding_projection.weight.data.copy_(embed_w)
                self.vision_tower = tower
            else:
                from transformers import Gemma4ForConditionalGeneration

                self.vlm = Gemma4ForConditionalGeneration.from_pretrained(
                    model_id, revision=self.spec.revision, dtype=torch.bfloat16, attn_implementation=attn_implementation
                ).eval()
                self.config = self.vlm.config
                self.vision_tower, embedder = self.vlm.model.vision_tower, self.vlm.model.embed_vision
            if verify:
                hub.verify_module_weights(self.vision_tower, embedder.embedding_projection.weight, self.model_id)
            if compute != torch.bfloat16:
                hub.cast_keep_rope_fp32(self.vlm if self.vlm is not None else self.vision_tower, compute)
                embedder.to(compute)
        self.embed_vision = embedder
        self.embed_projection = embedder.embedding_projection

        vc = self.config.vision_config
        self.hidden_size = int(vc.hidden_size)
        self.num_layers = int(vc.num_hidden_layers)
        self.patch_size = int(vc.patch_size)
        self.pooling_kernel_size = int(vc.pooling_kernel_size)
        self.lm_hidden_size = int(self.config.text_config.hidden_size)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "Gemma4Teacher":  # always frozen / eval
        return super().train(False)

    def _apply(self, fn, *args, **kwargs):  # .to(dtype) must never round the RoPE frequencies
        out = super()._apply(fn, *args, **kwargs)
        hub.restore_rope_fp32(self)
        return out

    @property
    def text_hidden_size(self) -> int:
        return self.lm_hidden_size

    @torch.no_grad()
    def encode_vision(self, inputs: TeacherInputs, layers: Sequence[int] = ()) -> TeacherVisionOutput:
        """Teacher soft tokens.

        Args:
            inputs: Gemma-format patches (``pixel_values [B, P, 768]``, ``pixel_position_ids [B, P, 2]``).
            layers: encoder layer indices (0-based) whose outputs to return, pooled to ``[N, D]``.
        Returns:
            :class:`TeacherVisionOutput` with ``tokens [N, 768]``.
        """
        bad = [i for i in layers if not 0 <= i < self.num_layers]
        if bad:
            raise IndexError(f"teacher layers {bad} out of range [0, {self.num_layers})")
        captured: dict[int, torch.Tensor] = {}
        handles = [
            self.vision_tower.encoder.layers[i].register_forward_hook(
                lambda _m, _in, out, i=i: captured.__setitem__(i, out[0] if isinstance(out, tuple) else out)
            )
            for i in layers
        ]
        try:
            dtype = next(self.vision_tower.parameters()).dtype
            out = self.vision_tower(pixel_values=inputs.pixel_values.to(dtype), pixel_position_ids=inputs.pixel_position_ids)
        finally:
            for h in handles:
                h.remove()
        inter = {}
        for i, h in captured.items():
            pooled, lengths = pool_patch_states(h, inputs.pixel_position_ids, self.pooling_kernel_size)
            if lengths != list(inputs.image_token_lengths):
                raise RuntimeError(f"pooled lengths {lengths} != {inputs.image_token_lengths}")
            inter[i] = pooled
        return TeacherVisionOutput(tokens=out.last_hidden_state, intermediate_tokens=inter, image_token_lengths=list(inputs.image_token_lengths))

    def project_to_lm(self, tokens: torch.Tensor) -> torch.Tensor:
        """Frozen ``embed_vision`` exactly as the VLM runs it (checkpoint dtype): ``[N, 768] -> [N, 1536 | 2560]``."""
        w = self.embed_projection.weight
        return self.embed_vision(tokens.to(w.dtype))

    def project_to_lm_fp32(self, tokens: torch.Tensor) -> torch.Tensor:
        """Same map in fp32 (for losses; differentiable w.r.t. ``tokens``, weights stay frozen)."""
        x = tokens.float()
        x = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.embed_vision.eps, -0.5)  # as Gemma4RMSNorm
        return torch.nn.functional.linear(x, self.embed_projection.weight.float())

    @torch.no_grad()
    def forward_vlm(self, **vlm_inputs: torch.Tensor) -> torch.Tensor:
        if self.vlm is None:
            raise RuntimeError("teacher was loaded with load='vision'; use load='vlm' for logits")
        return self.vlm(**vlm_inputs).logits
