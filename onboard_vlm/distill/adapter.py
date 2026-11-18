"""Training-only projection heads between student and teacher token spaces.

The :class:`DistillationAdapter` is owned by the training task, never by the student, so it is
dropped automatically when the student is exported for inference / grafting.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping

from torch import nn

from onboard_vlm.interfaces import StudentVisionOutput


def _proj(d_in: int, d_out: int, kind: str, force: bool) -> nn.Module:
    if d_in == d_out and not force:
        return nn.Identity()
    if kind == "linear":  # == 1x1 conv on the token grid
        return nn.Linear(d_in, d_out)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(approximate="tanh"), nn.Linear(d_out, d_out))
    raise ValueError(f"unknown projection kind {kind!r}")


class DistillationAdapter(nn.Module):
    """Projects student final / intermediate tokens into the teacher's widths.

    Args:
        student_dim: ``D_target`` of the student's final tokens.
        teacher_dim: teacher vision hidden size (read from ``teacher.config.vision_config.hidden_size``).
        layer_map: ``{student_idx: teacher_idx}`` for intermediate-layer losses.
        student_layer_dims: ``{student_idx: D_layer}`` (``student.layer_dims``).
        teacher_layer_dim: width of teacher intermediate layers (== ``teacher_dim`` for Gemma 4).
        kind: ``"linear"`` or ``"mlp"``.
        force_final / force_layers: add a projection even when widths already match (FitNets-style).
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        layer_map: Mapping[int, int] | None = None,
        student_layer_dims: Mapping[int, int] | None = None,
        teacher_layer_dim: int | None = None,
        kind: str = "linear",
        force_final: bool = False,
        force_layers: bool = False,
    ):
        super().__init__()
        self.layer_map = {int(k): int(v) for k, v in (layer_map or {}).items()}
        if len(set(self.layer_map.values())) != len(self.layer_map):
            raise ValueError(f"layer_map must be injective (one student layer per teacher layer): {self.layer_map}")
        teacher_layer_dim = teacher_layer_dim or teacher_dim
        self.final = _proj(student_dim, teacher_dim, kind, force_final)
        dims = dict(student_layer_dims or {})
        missing = set(self.layer_map) - set(dims)
        if missing:
            raise KeyError(f"student has no layers {sorted(missing)}; available {sorted(dims)}")
        self.layers = nn.ModuleDict({str(s): _proj(dims[s], teacher_layer_dim, kind, force_layers) for s in self.layer_map})

    @classmethod
    def from_models(cls, student: nn.Module, teacher: nn.Module, layer_map: Mapping[int, int] | None = None, **kw) -> "DistillationAdapter":
        """Builds heads by inspecting ``teacher.hidden_size`` (== ``config.vision_config.hidden_size``)."""
        return cls(student.out_dim, teacher.hidden_size, layer_map, student.layer_dims, teacher.hidden_size, **kw)

    @classmethod
    def from_dims(cls, student: nn.Module, teacher_dim: int, layer_map: Mapping[int, int] | None = None, **kw) -> "DistillationAdapter":
        return cls(student.out_dim, teacher_dim, layer_map, student.layer_dims, teacher_dim, **kw)

    def forward(self, out: StudentVisionOutput) -> StudentVisionOutput:
        """Returns a new output whose ``intermediate_tokens`` are keyed by *teacher* layer index."""
        inter = {t: self.layers[str(s)](out.intermediate_tokens[s]) for s, t in self.layer_map.items()}
        return dataclasses.replace(out, tokens=self.final(out.tokens), intermediate_tokens=inter)
