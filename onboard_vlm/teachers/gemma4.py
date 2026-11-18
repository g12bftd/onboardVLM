"""Gemma 4 teacher.

* ``encode_vision`` runs ``Gemma4VisionModel`` and returns its unpadded ``[N, D]`` soft tokens
  (post pooling, post optional standardisation, *pre* ``embed_vision``), plus intermediate
  encoder-layer outputs captured with forward hooks. Encoder layers operate on 16-px patches
  (``9x`` more tokens than the output), so intermediate states are average-pooled 3x3 by patch
  position (the teacher's own pooling rule, without its ``sqrt(D)`` output scaling) to obtain
  ``[N, D]`` tensors aligned with the final tokens.
* ``forward_vlm`` runs the full ``Gemma4ForConditionalGeneration`` (Stage-2 logit targets).

``load="vision"`` downloads only the safetensors shards that contain vision-tower weights, so
even the 31B teacher can be used for Stage-1 on a single GPU.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Sequence

import torch
from torch import nn

from onboard_vlm.interfaces import POOLING_KERNEL, TeacherInputs, TeacherVisionOutput, VisionTeacher
from onboard_vlm.registry import register_teacher

log = logging.getLogger(__name__)

# Variants that ship without a vision encoder. The authoritative check is
# ``config.vision_config is None``; the name pattern only lets us fail before downloading.
ENCODER_FREE_PATTERNS = (re.compile(r"gemma-4-12b", re.IGNORECASE),)


class EncoderFreeTeacherError(ValueError):
    """Raised when a teacher without a vision tower is requested for vision distillation."""


def _dtype(d: str | torch.dtype) -> torch.dtype:
    return d if isinstance(d, torch.dtype) else getattr(torch, d)


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


def _load_vision_tower_only(model_id: str, config: Any, dtype: torch.dtype, **hub_kwargs: Any) -> nn.Module | None:
    """Load ``Gemma4VisionModel`` from only the shards that hold vision weights; None on failure."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

    try:
        try:
            index = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json", **hub_kwargs)))
            weight_map: dict[str, str] = index["weight_map"]
        except Exception:  # single-file checkpoint
            weight_map = None  # type: ignore[assignment]
        if weight_map is None:
            shard_keys = {"model.safetensors": None}
        else:
            keys = [k for k in weight_map if "vision_tower." in k]
            shard_keys = {}
            for k in keys:
                shard_keys.setdefault(weight_map[k], []).append(k)
        state: dict[str, torch.Tensor] = {}
        for shard, keys in shard_keys.items():
            with safe_open(hf_hub_download(model_id, shard, **hub_kwargs), framework="pt") as f:
                for k in keys if keys is not None else [k for k in f.keys() if "vision_tower." in k]:
                    state[k.split("vision_tower.", 1)[1]] = f.get_tensor(k)
        model = Gemma4VisionModel(config.vision_config)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            log.warning("vision-only load mismatch (missing=%s unexpected=%s); falling back", missing[:5], unexpected[:5])
            return None
        return model.to(dtype)
    except Exception as e:  # pragma: no cover - network / layout dependent
        log.warning("vision-only load failed (%s); falling back to full VLM load", e)
        return None


@register_teacher("gemma4")
class Gemma4Teacher(VisionTeacher, nn.Module):
    """Frozen Gemma 4 teacher (E2B / E4B / 26B-A4B / 31B; anything with a vision tower).

    Args:
        model_id: HF repo id, e.g. ``google/gemma-4-E2B-it``.
        load: ``"vision"`` (vision tower only; Stage-1) or ``"vlm"`` (full model; Stage-2 / eval).
        dtype: weight dtype.
        config: build a randomly initialised model from this ``Gemma4Config`` (tests / dry runs).
        tiny: kwargs of :func:`onboard_vlm.utils.tiny.tiny_gemma4_config` (random tiny teacher, no download).
        attn_implementation: passed to ``from_pretrained``.
        hub_kwargs: ``revision``, ``cache_dir``, ``local_files_only`` ...
    """

    def __init__(
        self,
        model_id: str | None = None,
        load: str = "vision",
        dtype: str | torch.dtype = "bfloat16",
        config: Any | None = None,
        attn_implementation: str = "sdpa",
        tiny: dict[str, Any] | None = None,
        **hub_kwargs: Any,
    ):
        nn.Module.__init__(self)
        if tiny is not None:  # random tiny model for offline dry runs (see utils.tiny)
            from onboard_vlm.utils.tiny import tiny_gemma4_config

            config, model_id = tiny_gemma4_config(**tiny), None
        from transformers import AutoConfig, Gemma4Config, Gemma4ForConditionalGeneration
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        if load not in ("vision", "vlm"):
            raise ValueError(f"load must be 'vision' or 'vlm', got {load!r}")
        if model_id is not None and any(p.search(model_id) for p in ENCODER_FREE_PATTERNS):
            raise EncoderFreeTeacherError(
                f"{model_id} is an encoder-free Gemma 4 variant: it has no vision tower to distil. "
                "Use google/gemma-4-E2B-it, google/gemma-4-E4B-it or google/gemma-4-31B-it."
            )
        torch_dtype = _dtype(dtype)
        if config is None:
            if model_id is None:
                raise ValueError("either model_id or config is required")
            config = AutoConfig.from_pretrained(model_id, **hub_kwargs)
        elif isinstance(config, dict):
            config = Gemma4Config(**config)
        if getattr(config, "vision_config", None) is None:
            raise EncoderFreeTeacherError(
                f"{model_id or 'config'} has no vision_config (encoder-free model); cannot be a vision teacher."
            )
        self.model_id = model_id
        self.config = config
        self.vlm: nn.Module | None = None

        if model_id is None:  # random init from config
            if load == "vlm":
                self.vlm = Gemma4ForConditionalGeneration(config).to(torch_dtype)
                self.vision_tower = self.vlm.model.vision_tower
            else:
                self.vision_tower = Gemma4VisionModel(config.vision_config).to(torch_dtype)
        elif load == "vision" and (tower := _load_vision_tower_only(model_id, config, torch_dtype, **hub_kwargs)) is not None:
            self.vision_tower = tower
        else:
            vlm = Gemma4ForConditionalGeneration.from_pretrained(
                model_id, dtype=torch_dtype, attn_implementation=attn_implementation, **hub_kwargs
            )
            if load == "vlm":
                self.vlm = vlm
                self.vision_tower = vlm.model.vision_tower
            else:
                self.vision_tower = vlm.model.vision_tower
                del vlm

        vc = config.vision_config
        self.hidden_size = int(vc.hidden_size)
        self.num_layers = int(vc.num_hidden_layers)
        self.patch_size = int(vc.patch_size)
        self.pooling_kernel_size = int(vc.pooling_kernel_size)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "Gemma4Teacher":  # always frozen / eval
        return super().train(False)

    @property
    def text_hidden_size(self) -> int:
        return int(self.config.text_config.hidden_size)

    @torch.no_grad()
    def encode_vision(self, inputs: TeacherInputs, layers: Sequence[int] = ()) -> TeacherVisionOutput:
        """Teacher soft tokens.

        Args:
            inputs: Gemma-format patches (``pixel_values [B, P, 768]``, ``pixel_position_ids [B, P, 2]``).
            layers: encoder layer indices (0-based) whose outputs to return, pooled to ``[N, D]``.
        Returns:
            :class:`TeacherVisionOutput` with ``tokens [N, D_teacher]``.
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
            out = self.vision_tower(
                pixel_values=inputs.pixel_values.to(dtype), pixel_position_ids=inputs.pixel_position_ids
            )
        finally:
            for h in handles:
                h.remove()
        tokens = out.last_hidden_state
        inter = {}
        for i, h in captured.items():
            pooled, lengths = pool_patch_states(h, inputs.pixel_position_ids, self.pooling_kernel_size)
            if lengths != list(inputs.image_token_lengths):
                raise RuntimeError(f"pooled lengths {lengths} != {inputs.image_token_lengths}")
            inter[i] = pooled
        return TeacherVisionOutput(tokens=tokens, intermediate_tokens=inter, image_token_lengths=list(inputs.image_token_lengths))

    @torch.no_grad()
    def forward_vlm(self, **vlm_inputs: torch.Tensor) -> torch.Tensor:
        if self.vlm is None:
            raise RuntimeError("teacher was loaded with load='vision'; use load='vlm' for logits")
        return self.vlm(**vlm_inputs).logits
