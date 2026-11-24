"""Graft a distilled student into a Gemma 4 VLM so that ``forward`` / ``.generate()`` and every
evaluator run unchanged.

``Gemma4Model`` calls ``self.vision_tower(pixel_values=..., pixel_position_ids=...)`` for both
images (``get_image_features``) and video frames (``get_video_features``) and expects
``last_hidden_state [N, D_vision]`` (unpadded). :class:`StudentVisionTower` provides exactly that
interface: it reconstructs the dense image from the processor's patch sequence (lossless, see
``geometry.unpatchify_batch``) and runs the student.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

from onboard_vlm.geometry import pad_dense, unpatchify_batch
from onboard_vlm.interfaces import PATCH_SIZE


class StudentVisionTower(nn.Module):
    """Drop-in replacement for ``Gemma4VisionModel`` backed by a student encoder.

    Args:
        student: any :class:`VisionStudent`.
        per_image: run images one at a time (default: ``not student.batch_invariant``), so CNN
            students never see neighbouring images' zero padding at inference.
        chunk_size: max images per student call (video batches can carry hundreds of frames).
    """

    def __init__(self, student: nn.Module, per_image: bool | None = None, chunk_size: int = 64, domain: Any = None):
        super().__init__()
        self.student = student
        self.per_image = (not getattr(student, "batch_invariant", True)) if per_image is None else per_image
        self.chunk_size = chunk_size
        self.domain = domain  # InputDomain: inputs outside it are counted (reported by the evaluators)
        self.images_seen = 0
        self.out_of_domain = 0

    def _count_domain(self, pixel_position_ids: torch.Tensor) -> None:
        if self.domain is None:
            return
        valid = (pixel_position_ids >= 0).all(-1)
        for pos, v in zip(pixel_position_ids, valid):
            if not v.any():
                continue
            w, h = (pos[v].max(0).values + 1).tolist()
            self.images_seen += 1
            self.out_of_domain += int(not self.domain.contains(h * PATCH_SIZE, w * PATCH_SIZE))

    @property
    def dtype(self) -> torch.dtype:
        return next(self.student.parameters()).dtype

    def forward(self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor, **_: Any) -> BaseModelOutputWithPooling:
        """pixel_values ``[B, P, 768]`` in [0, 1]; pixel_position_ids ``[B, P, 2]`` -> ``last_hidden_state [N, D]``."""
        self._count_domain(pixel_position_ids)
        if getattr(self.student, "accepts_teacher_patches", False):  # Gemma-architecture students take the patches as-is
            out = self.student(pixel_values=pixel_values.to(self.dtype), pixel_position_ids=pixel_position_ids)
            return BaseModelOutputWithPooling(last_hidden_state=out.tokens)
        dense, mask, sizes = unpatchify_batch(pixel_values, pixel_position_ids, PATCH_SIZE)
        dense = dense.to(self.dtype)
        step = 1 if self.per_image else self.chunk_size
        outs = []
        for i in range(0, dense.shape[0], step):
            if step == 1:  # crop to the image's own size: no padding at all
                h, w = sizes[i]
                x, m, _ = pad_dense([dense[i, :, :h, :w]])
                outs.append(self.student(x, m).tokens)
            else:
                outs.append(self.student(dense[i : i + step], mask[i : i + step]).tokens)
        return BaseModelOutputWithPooling(last_hidden_state=torch.cat(outs, dim=0))


class LoRALinear(nn.Module):
    """``y = W x + (alpha / r) B A x`` with ``B`` zero-initialised (starts exactly at ``W``)."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.scale = alpha / rank
        self.lora_a.to(base.weight.dtype)
        self.lora_b.to(base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """A plain ``nn.Linear`` with ``W + scale * B A`` folded in (for export)."""
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin.to(self.base.weight.device, self.base.weight.dtype)
        delta = (self.lora_b.weight.float() @ self.lora_a.weight.float()) * self.scale
        lin.weight.copy_((self.base.weight.float() + delta).to(lin.weight.dtype))
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
        return lin


class GraftedVLM(nn.Module):
    """A ``Gemma4ForConditionalGeneration`` whose vision tower can be hot-swapped.

    Args:
        vlm: the deployment VLM (e.g. ``google/gemma-4-E2B-it``).
        student: student encoder (``out_dim`` must equal the VLM's vision hidden size).
        keep_teacher_tower: keep the original tower for :meth:`teacher_vision` (logit-distillation targets
            from the same VLM, or teacher-vs-student evaluation in one process).
    """

    def __init__(self, vlm: nn.Module, student: nn.Module | None = None, keep_teacher_tower: bool = False, per_image: bool | None = None,
                 domain: Any = None):
        super().__init__()
        self.vlm = vlm
        self.keep_teacher_tower = keep_teacher_tower
        # A plain list keeps the original tower out of the module tree (no duplicate parameters).
        self._teacher_tower: list[nn.Module | None] = [vlm.model.vision_tower]
        self.teacher_tower = vlm.model.vision_tower if keep_teacher_tower else None
        self.student_tower: StudentVisionTower | None = None
        if student is not None:
            self.graft(student, per_image=per_image, domain=domain)

    @classmethod
    def from_pretrained(
        cls,
        vlm_id: str,
        student_dir: str | None = None,
        dtype: str | torch.dtype = "bfloat16",
        keep_teacher_tower: bool = False,
        attn_implementation: str = "sdpa",
        load_embed_vision: bool = True,
        revision: str | None = None,
        domain: Any = None,
        **kw: Any,
    ) -> "GraftedVLM":
        """Load the original Gemma 4 VLM (``from_pretrained``) and graft ``student_dir``.

        If ``student_dir`` contains ``embed_vision.safetensors`` (from the embed_vision ablation of
        logit distillation) and ``load_embed_vision`` is set, it replaces the VLM's projection.
        """
        import os

        from safetensors.torch import load_file

        from onboard_vlm.students.io import load_student
        from onboard_vlm.teachers.gemma4 import load_gemma4_vlm

        torch_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        vlm = load_gemma4_vlm(vlm_id, revision, torch_dtype, attn_implementation, **kw)
        student = None
        if student_dir is not None:
            student, _ = load_student(student_dir)
            student = student.to(torch_dtype)
            ev_path = os.path.join(student_dir, "embed_vision.safetensors")
            if load_embed_vision and os.path.isfile(ev_path):
                vlm.model.embed_vision.load_state_dict({k: v.to(torch_dtype) for k, v in load_file(ev_path).items()})
        return cls(vlm, student, keep_teacher_tower=keep_teacher_tower, domain=domain)

    def embed_vision_state_dict(self) -> dict[str, torch.Tensor]:
        """``embed_vision`` weights in the stock Gemma layout (LoRA merged), ready for export."""
        ev = self.vlm.model.embed_vision
        state = {k: v for k, v in ev.state_dict().items() if ".lora_" not in k and ".base." not in k}
        proj = ev.embedding_projection
        if isinstance(proj, LoRALinear):
            for k, v in proj.merged().state_dict().items():
                state[f"embedding_projection.{k}"] = v
        return {k: v.detach().cpu().contiguous() for k, v in state.items()}

    @property
    def vision_hidden_size(self) -> int:
        return int(self.vlm.config.vision_config.hidden_size)

    def graft(self, student: nn.Module, per_image: bool | None = None, domain: Any = None) -> None:
        if getattr(student, "out_dim", self.vision_hidden_size) != self.vision_hidden_size:
            raise ValueError(
                f"student.out_dim={student.out_dim} != VLM vision hidden size {self.vision_hidden_size}; "
                "the student must be trained with out_dim equal to the deployment VLM's vision width"
            )
        ref = next(self.vlm.parameters())
        self.student_tower = StudentVisionTower(student.to(ref.device), per_image=per_image, domain=domain)
        self.vlm.model.vision_tower = self.student_tower
        if not self.keep_teacher_tower:
            self._teacher_tower = [None]  # free the Gemma tower's memory

    def _original_tower(self) -> nn.Module:
        tower = self._teacher_tower[0]
        if tower is None:
            raise RuntimeError("original vision tower was released; construct with keep_teacher_tower=True")
        return tower

    def ungraft(self) -> None:
        self.vlm.model.vision_tower = self._original_tower()

    @contextlib.contextmanager
    def teacher_vision(self) -> Iterator[None]:
        """Temporarily run the original Gemma vision tower."""
        current = self.vlm.model.vision_tower
        self.vlm.model.vision_tower = self._original_tower()
        try:
            yield
        finally:
            self.vlm.model.vision_tower = current

    def configure_stage2(self, train_embed_vision: str = "full", lora_rank: int = 16, lora_alpha: float = 32.0, train_student: bool = False) -> list[nn.Parameter]:
        """Freeze everything except ``embed_vision`` (full / LoRA / none) and optionally the student.

        Returns the trainable parameters.
        """
        self.vlm.requires_grad_(False)
        if self.teacher_tower is not None:
            self.teacher_tower.requires_grad_(False)
        ev = self.vlm.model.embed_vision
        if train_embed_vision == "full":
            ev.requires_grad_(True)
        elif train_embed_vision == "lora":
            if not isinstance(ev.embedding_projection, LoRALinear):
                ev.embedding_projection = LoRALinear(ev.embedding_projection, lora_rank, lora_alpha).to(next(ev.parameters()).device)
            ev.embedding_projection.lora_a.requires_grad_(True)
            ev.embedding_projection.lora_b.requires_grad_(True)
        elif train_embed_vision != "none":
            raise ValueError(f"train_embed_vision must be full | lora | none, got {train_embed_vision!r}")
        if self.student_tower is not None:
            self.student_tower.requires_grad_(train_student)
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, **inputs: Any) -> Any:
        return self.vlm(**inputs)

    @torch.no_grad()
    def generate(self, **inputs: Any) -> torch.Tensor:
        return self.vlm.generate(**inputs)
