"""Gemma 4 E2B / E4B teacher: the original Hugging Face model, loaded with ``from_pretrained``.

* ``load="vision"`` (feature / embedding distillation): the full original
  ``Gemma4ForConditionalGeneration`` is loaded, its ``model.vision_tower`` and
  ``model.embed_vision`` are kept, and the language / audio parts are released.
* ``load="vlm"`` (logit distillation): the whole original model is kept.

``encode_vision`` returns the vision tower's unpadded ``[N, 768]`` soft tokens (exactly
``get_image_features(...).last_hidden_state``) plus intermediate encoder-layer outputs captured
with forward hooks. Encoder layers run on 16-px patches (9x more tokens than the output), so
intermediate states are average-pooled 3x3 by patch position, which is the tower's own pooling
rule without its ``sqrt(D)`` scaling.

The vision tower is shared by all four checkpoints: on 2026-11-23 the 658 vision-tower tensors were
byte-identical in E2B, E4B, base and ``-it``. The teacher choice only changes ``embed_vision``
(768->1536 for E2B, 768->2560 for E4B) and, for logit distillation, the language model.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Sequence

import torch
from torch import nn

from onboard_vlm.interfaces import POOLING_KERNEL, TeacherInputs, TeacherVisionOutput, VisionTeacher
from onboard_vlm.registry import register_teacher
from onboard_vlm.utils.logging import describe_module, get_logger, stage

log = get_logger(__name__)

# Supported teachers and their default revisions (commits at the time of writing, for
# reproducibility). Override ``revision`` in the teacher YAML (e.g. ``main``) to change it.
SUPPORTED_TEACHERS: dict[str, str] = {
    "google/gemma-4-E2B-it": "4adf15748de9e1a0097b6274cee9c15836c41c37",
    "google/gemma-4-E4B-it": "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
    "google/gemma-4-E2B": "433813a669639fd7eaa83155828e9000ed357d53",
    "google/gemma-4-E4B": "11b8250ef59eb3ebc5dae7b21144d7b84f2c06dd",
}


class UnsupportedTeacherError(ValueError):
    """Only the Gemma 4 E2B / E4B checkpoints are supported teachers."""


class EncoderFreeTeacherError(UnsupportedTeacherError):
    """Raised for configs without a vision tower (e.g. the encoder-free ``gemma4_unified`` models)."""


def resolve_revision(model_id: str, revision: str | None = None) -> str:
    if model_id not in SUPPORTED_TEACHERS:
        raise UnsupportedTeacherError(f"{model_id!r} is not supported; use one of {sorted(SUPPORTED_TEACHERS)}")
    return revision or SUPPORTED_TEACHERS[model_id]


def load_gemma4_vlm(model_id: str, revision: str | None = None, dtype: str | torch.dtype = "bfloat16", attn_implementation: str = "sdpa", **kw: Any):
    """``Gemma4ForConditionalGeneration.from_pretrained`` of a supported checkpoint, with logging."""
    from transformers import Gemma4ForConditionalGeneration

    rev = resolve_revision(model_id, revision)
    torch_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
    with stage(f"Loading original Gemma 4 weights {model_id}@{rev[:7]} ({torch_dtype}, attn={attn_implementation})", log):
        t0 = time.perf_counter()
        vlm = Gemma4ForConditionalGeneration.from_pretrained(
            model_id, revision=rev, dtype=torch_dtype, attn_implementation=attn_implementation, **kw
        ).eval()
        log.info("     loaded %s in %.1fs", describe_module(vlm), time.perf_counter() - t0)
    return vlm


def pool_patch_states(hidden: torch.Tensor, pixel_position_ids: torch.Tensor, k: int = POOLING_KERNEL) -> tuple[torch.Tensor, list[int]]:
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
    valid = (pixel_position_ids >= 0).all(-1)
    pos = pixel_position_ids.clamp(min=0)
    gw = pos[..., 0].amax(dim=-1, keepdim=True) + 1
    idx = (pos[..., 0] // k) + (gw // k) * (pos[..., 1] // k)
    n_tok = valid.sum(-1) // (k * k)
    length = p // (k * k)
    out = hidden.new_zeros((b, length, d), dtype=torch.float32)
    out.scatter_add_(1, idx[..., None].expand(-1, -1, d), hidden.float() * valid[..., None])
    out = (out / (k * k)).to(hidden.dtype)
    tok_mask = torch.arange(length, device=hidden.device)[None] < n_tok[:, None]
    return out[tok_mask], n_tok.tolist()


@register_teacher("gemma4")
class Gemma4Teacher(VisionTeacher, nn.Module):
    """Frozen original Gemma 4 E2B / E4B teacher.

    Args:
        model_id: ``google/gemma-4-E2B-it`` (default), ``-E4B-it``, ``-E2B`` or ``-E4B``.
        revision: Hub revision; ``None`` = the pinned default in :data:`SUPPORTED_TEACHERS`.
        load: ``"vision"`` (keep vision tower + embed_vision) or ``"vlm"`` (keep everything).
        dtype: weight dtype passed to ``from_pretrained`` (checkpoints are bf16).
        attn_implementation: passed to ``from_pretrained``.
        tiny: build a tiny randomly initialised model with the same classes (tests / dry runs only).
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-E2B-it",
        revision: str | None = None,
        load: str = "vision",
        dtype: str | torch.dtype = "bfloat16",
        attn_implementation: str = "sdpa",
        tiny: dict[str, Any] | None = None,
        config: Any | None = None,
        key: str | None = None,
    ):
        """``key`` is the config's short id (used for output paths only)."""
        nn.Module.__init__(self)
        if load not in ("vision", "vlm"):
            raise ValueError(f"load must be 'vision' or 'vlm', got {load!r}")
        torch_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        if tiny is not None or config is not None:
            from transformers import Gemma4Config, Gemma4ForConditionalGeneration

            from onboard_vlm.utils.tiny import tiny_gemma4_config

            cfg = tiny_gemma4_config(**tiny) if tiny is not None else (Gemma4Config(**config) if isinstance(config, dict) else config)
            if getattr(cfg, "vision_config", None) is None:
                raise EncoderFreeTeacherError("config has no vision_config (encoder-free model); it cannot be a vision teacher")
            log.info("===> Building tiny random Gemma 4 teacher (tests / dry runs)")
            vlm = Gemma4ForConditionalGeneration(cfg).to(torch_dtype).eval()
            self.model_id, self.revision = None, None
        else:
            self.revision = resolve_revision(model_id, revision)
            self.model_id = model_id
            vlm = load_gemma4_vlm(model_id, self.revision, torch_dtype, attn_implementation)
        self.config = vlm.config
        self.vision_tower = vlm.model.vision_tower
        self.embed_vision = vlm.model.embed_vision
        if load == "vlm":
            self.vlm: nn.Module | None = vlm
        else:
            self.vlm = None
            del vlm  # keep only the vision tower + embed_vision; free the language model
            gc.collect()
        vc = self.config.vision_config
        self.hidden_size = int(vc.hidden_size)
        self.num_layers = int(vc.num_hidden_layers)
        self.pooling_kernel_size = int(vc.pooling_kernel_size)
        self.lm_hidden_size = int(self.config.text_config.hidden_size)
        self.requires_grad_(False)
        self.eval()
        log.info("     teacher vision tower: %s", describe_module(self.vision_tower))
        log.info("     teacher embed_vision: %d -> %d (frozen)%s", self.hidden_size, self.lm_hidden_size,
                 "" if load == "vlm" else " | language model released (load=vision)")

    def train(self, mode: bool = True) -> "Gemma4Teacher":  # always frozen / eval
        return super().train(False)

    @torch.no_grad()
    def encode_vision(self, inputs: TeacherInputs, layers: Sequence[int] = ()) -> TeacherVisionOutput:
        """Teacher soft tokens ``[N, 768]`` (+ intermediate layers pooled to the 48-px grid)."""
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
        """The original frozen ``embed_vision``: ``[N, 768] -> [N, 1536 | 2560]`` (differentiable w.r.t. ``tokens``)."""
        return self.embed_vision(tokens.to(self.embed_vision.embedding_projection.weight.dtype))

    @torch.no_grad()
    def forward_vlm(self, **vlm_inputs: torch.Tensor) -> torch.Tensor:
        if self.vlm is None:
            raise RuntimeError("teacher was loaded with load='vision'; use load='vlm' for logits")
        return self.vlm(**vlm_inputs).logits
