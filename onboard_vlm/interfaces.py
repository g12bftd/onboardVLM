"""Core data contracts shared by every teacher, student, loss, collator and evaluator.

Shape conventions used throughout the codebase
----------------------------------------------
* ``B``  images in a batch, ``N = sum(image_token_lengths)`` valid soft tokens in the batch.
* ``P``  patch-sequence length of the Gemma-format teacher input (``max_soft_tokens * pooling**2``).
* ``H, W`` dense (batch-padded) image size; always multiples of ``TOKEN_STRIDE`` (48 px).
* ``Ht, Wt = H // 48, W // 48`` dense soft-token grid.

Packed token order (the "unified vision output contract")
---------------------------------------------------------
``tokens[N, D]`` concatenates images in batch order. Inside an image, tokens are in
row-major order over that image's own ``(h_i // 48, w_i // 48)`` grid, i.e.
``index = row * (w_i // 48) + col``. This is exactly the order produced by
``Gemma4VisionModel`` (its pooler indexes kernels as ``x // 3 + (W_patches // 3) * (y // 3)``
and strips padding with a boolean mask), so student and teacher tokens can be compared
element-wise without any re-indexing.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
from torch import nn

PATCH_SIZE = 16
POOLING_KERNEL = 3
TOKEN_STRIDE = PATCH_SIZE * POOLING_KERNEL  # 48 px per soft token
SUPPORTED_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TransformParams:
    """Deterministic, serialisable description of the geometric transform of one image.

    Applied exactly once per image (crop -> optional h-flip -> aspect-preserving resize)
    before the teacher / student views are produced. Stored alongside cached teacher
    features so that the student view can be reproduced bit-exactly.

    Attributes:
        src_hw: ``(height, width)`` of the decoded source image.
        crop_box: ``(top, left, height, width)`` in source pixels, or ``None`` for no crop.
        hflip: Whether the (cropped) image is mirrored horizontally.
        target_hw: Output ``(height, width)``; both multiples of 48.
        token_budget: Gemma 4 ``max_soft_tokens`` used to derive ``target_hw``.
        degradations: ``((op, params), ...)`` applied after the resize (``data.degrade``), identical
            for teacher and student; ``()`` = clean image (bit-exact Gemma preprocessing).
    """

    src_hw: tuple[int, int]
    crop_box: tuple[int, int, int, int] | None
    hflip: bool
    target_hw: tuple[int, int]
    token_budget: int
    degradations: tuple = ()

    @property
    def token_grid(self) -> tuple[int, int]:
        return self.target_hw[0] // TOKEN_STRIDE, self.target_hw[1] // TOKEN_STRIDE

    @property
    def num_tokens(self) -> int:
        gh, gw = self.token_grid
        return gh * gw

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TransformParams":
        from onboard_vlm.data.degrade import freeze

        crop = d.get("crop_box")
        return cls(
            src_hw=tuple(d["src_hw"]),
            crop_box=tuple(crop) if crop is not None else None,
            hflip=bool(d["hflip"]),
            target_hw=tuple(d["target_hw"]),
            token_budget=int(d["token_budget"]),
            degradations=freeze(d.get("degradations", ())),
        )


# --------------------------------------------------------------------------------------
# Vision outputs (packed / unpadded)
# --------------------------------------------------------------------------------------
@dataclass
class _PackedTokens:
    tokens: torch.Tensor
    image_token_lengths: list[int]
    intermediate_tokens: dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = sum(self.image_token_lengths)
        if self.tokens.dim() != 2 or self.tokens.shape[0] != n:
            raise ValueError(
                f"{type(self).__name__}.tokens must be [sum(image_token_lengths)={n}, D], "
                f"got {tuple(self.tokens.shape)}"
            )
        for k, v in self.intermediate_tokens.items():
            if v.dim() != 2 or v.shape[0] != n:
                raise ValueError(f"intermediate_tokens[{k}] must be [{n}, D_layer], got {tuple(v.shape)}")

    def split(self, which: int | None = None) -> tuple[torch.Tensor, ...]:
        """Per-image views ``(tokens_i[n_i, D], ...)`` of the final (``which=None``) or a layer's tokens."""
        t = self.tokens if which is None else self.intermediate_tokens[which]
        return torch.split(t, self.image_token_lengths, dim=0)

    @property
    def dim(self) -> int:
        return self.tokens.shape[-1]


@dataclass
class StudentVisionOutput(_PackedTokens):
    """Output contract of every student encoder.

    Attributes:
        tokens: ``FloatTensor[N, D_target]`` final soft tokens, unpadded, Gemma 4 order.
        intermediate_tokens: ``{layer_idx: FloatTensor[N, D_layer]}`` hidden tokens on the same
            48-px grid (finer-grained features are average-pooled with exact integer factors).
        image_token_lengths: tokens per image, ``sum == N``.
    """


@dataclass
class TeacherVisionOutput(_PackedTokens):
    """Output contract of every teacher; identical layout to :class:`StudentVisionOutput`."""


# --------------------------------------------------------------------------------------
# Batches
# --------------------------------------------------------------------------------------
def _to(x: Any, device: torch.device | str, non_blocking: bool = True) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: _to(v, device, non_blocking) for k, v in x.items()}
    if isinstance(x, (_PackedTokens, TeacherInputs)):
        return x.to(device)
    return x


@dataclass
class TeacherInputs:
    """Exactly what ``Gemma4VisionModel.forward`` consumes.

    Attributes:
        pixel_values: ``FloatTensor[B, P, 3 * 16 * 16]`` in ``[0, 1]`` (Gemma maps to [-1, 1] itself).
        pixel_position_ids: ``LongTensor[B, P, 2]`` patch ``(x, y)``; ``-1`` marks padding patches.
        image_token_lengths: valid soft tokens per image.
    """

    pixel_values: torch.Tensor
    pixel_position_ids: torch.Tensor
    image_token_lengths: list[int]

    def to(self, device: torch.device | str) -> "TeacherInputs":
        return TeacherInputs(
            self.pixel_values.to(device, non_blocking=True),
            self.pixel_position_ids.to(device, non_blocking=True),
            list(self.image_token_lengths),
        )

    def vision_kwargs(self) -> dict[str, torch.Tensor]:
        """kwargs for ``Gemma4VisionModel.forward``."""
        return {"pixel_values": self.pixel_values, "pixel_position_ids": self.pixel_position_ids}

    def vlm_kwargs(self) -> dict[str, torch.Tensor]:
        """kwargs for ``Gemma4ForConditionalGeneration.forward`` (note the different name)."""
        return {"pixel_values": self.pixel_values, "image_position_ids": self.pixel_position_ids}


@dataclass
class DistillBatch:
    """Batch emitted by :class:`onboard_vlm.data.collator.DualViewCollator`.

    Attributes:
        teacher_inputs: Gemma-format patch inputs (``None`` when teacher features are cached).
        student_images: ``FloatTensor[B, 3, H, W]`` in ``[0, 1]``, zero-padded bottom/right to the
            batch maximum (multiples of 48). Students normalise internally (see ``students.common``).
        valid_token_mask: ``BoolTensor[B, H // 48, W // 48]``; True on each image's top-left
            ``(h_i // 48, w_i // 48)`` rectangle.
        image_token_lengths: ``[h_i * w_i // 48**2 for each image]``.
        transforms: per-image :class:`TransformParams` actually applied.
        sample_ids: dataset keys.
        text_inputs: tokenised VLM inputs (``input_ids``, ``attention_mask``, ``mm_token_type_ids``,
            ``loss_mask`` ...) for logit distillation / evaluation; ``None`` otherwise.
        cached_teacher: teacher features read from the offline cache, if any.
        extras: un-collated per-sample fields (questions, answers, categories ...).
    """

    teacher_inputs: TeacherInputs | None
    student_images: torch.Tensor
    valid_token_mask: torch.Tensor
    image_token_lengths: list[int]
    transforms: list[TransformParams]
    sample_ids: list[str]
    text_inputs: dict[str, torch.Tensor] | None = None
    cached_teacher: TeacherVisionOutput | None = None
    extras: list[dict[str, Any]] = field(default_factory=list)

    def to(self, device: torch.device | str) -> "DistillBatch":
        return dataclasses.replace(
            self,
            teacher_inputs=self.teacher_inputs.to(device) if self.teacher_inputs is not None else None,
            student_images=self.student_images.to(device, non_blocking=True),
            valid_token_mask=self.valid_token_mask.to(device, non_blocking=True),
            text_inputs=_to(self.text_inputs, device) if self.text_inputs is not None else None,
            cached_teacher=self.cached_teacher.to(device) if self.cached_teacher is not None else None,
        )

    def __len__(self) -> int:
        return self.student_images.shape[0]


def _packed_to(self: _PackedTokens, device: torch.device | str) -> _PackedTokens:
    return dataclasses.replace(
        self,
        tokens=self.tokens.to(device, non_blocking=True),
        intermediate_tokens={k: v.to(device, non_blocking=True) for k, v in self.intermediate_tokens.items()},
    )


_PackedTokens.to = _packed_to  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# Model interfaces
# --------------------------------------------------------------------------------------
class VisionStudent(nn.Module, ABC):
    """Base class of student encoders (thin: it only fixes the call signature and metadata).

    Subclasses must set:
        out_dim: ``D_target`` of ``tokens``.
        layer_dims: ``{layer_idx: D_layer}`` for every index that may appear in ``intermediate_tokens``.
        batch_invariant: True if the tokens of an image do not depend on the other images in the
            batch (ViTs with attention masks). CNNs see zero padding of their neighbours and are
            evaluated per-image by :class:`onboard_vlm.vlm.grafted.StudentVisionTower`.
    """

    out_dim: int
    layer_dims: dict[int, int]
    batch_invariant: bool = True

    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        valid_token_mask: torch.Tensor,
        return_layers: Sequence[int] = (),
    ) -> StudentVisionOutput:
        """Encode a dense padded batch.

        Args:
            images: ``FloatTensor[B, 3, H, W]`` in ``[0, 1]``; ``H, W`` multiples of 48.
            valid_token_mask: ``BoolTensor[B, H // 48, W // 48]``.
            return_layers: student layer/stage indices to return in ``intermediate_tokens``.
        """


class VisionTeacher(ABC):
    """Teacher interface: frozen VLM whose vision tower provides distillation targets."""

    hidden_size: int
    num_layers: int

    @abstractmethod
    def encode_vision(self, inputs: TeacherInputs, layers: Sequence[int] = ()) -> TeacherVisionOutput:
        """Final soft tokens ``[N, D_teacher]`` + intermediate layer tokens pooled to the 48-px grid."""

    @abstractmethod
    def forward_vlm(self, **vlm_inputs: torch.Tensor) -> torch.Tensor:
        """Full VLM forward returning next-token logits ``[B, L, V]`` (logit-distillation targets)."""
