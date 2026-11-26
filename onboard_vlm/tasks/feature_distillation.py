"""Distillation type 1 - feature distillation in the vision-token space.

The student's ``[N, 768]`` tokens (and optionally mapped intermediate layers and per-image token
relations) are matched to the original Gemma 4 vision tower's outputs. Only the vision tower of
the teacher is kept in memory; targets can also come from the offline teacher cache.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from onboard_vlm.distill.adapter import DistillationAdapter
from onboard_vlm.distill.fidelity import fidelity_metrics
from onboard_vlm.distill.losses import CompositeLoss, TokenFeatureLoss
from onboard_vlm.interfaces import DistillBatch, StudentVisionOutput, TeacherVisionOutput
from onboard_vlm.tasks.base_task import BaseDistillationTask, log, teacher_vision_dim
from onboard_vlm.utils.config import to_plain
from onboard_vlm.utils.logging import describe_module, stage


def auto_layer_map(student, points: int = 3, teacher_layers: int = 16) -> dict[int, int]:
    """``layer_map: auto``: ``points`` student layers spread over its depth (the last one included), each
    matched to the teacher layer it corresponds to: the layer it was initialised from
    (``student.source_layers``, gemma4_shrink) or the one at the same relative depth."""
    n = len(getattr(student, "layer_dims", {}) or {})
    if n == 0:
        return {}
    picks = sorted({max(0, round(k * n / points) - 1) for k in range(1, points + 1)})
    src = getattr(student, "source_layers", None) or [round((j + 1) * teacher_layers / n) - 1 for j in range(n)]
    return {j: int(src[j]) for j in picks}


class FeatureDistillationTask(BaseDistillationTask):
    """Student vision tokens <- original Gemma 4 vision-tower tokens.

    Reads from ``hparams``: ``student``, ``teacher``, ``criterion`` (list of weighted terms),
    ``layer_map`` (``{student_layer: teacher_layer}``) and ``cache_root`` (optional).

    Args:
        adapter: kwargs of :class:`DistillationAdapter` (training-only projection heads).
    """

    distillation_type = "feature"

    def __init__(self, hparams: Any, adapter: Mapping[str, Any] | None = None, optimizer: Mapping[str, Any] | None = None, scheduler: Mapping[str, Any] | None = None):
        super().__init__(hparams, optimizer, scheduler)
        self.student = self.build_student()
        lm = self.cfg.get("layer_map")
        self.layer_map = auto_layer_map(self.student) if lm == "auto" else {int(k): int(v) for k, v in (lm or {}).items()}
        with stage("Building distillation heads and criterion", log):
            dim = teacher_vision_dim(self.cfg["teacher"])
            self.adapter = DistillationAdapter.from_dims(self.student, dim, self.layer_map, **(to_plain(adapter) or {}))
            self.criterion = self.build_criterion()
            log.info("     teacher vision width %d | layer_map %s | adapter %s", dim, self.layer_map or "{} (final tokens only)", describe_module(self.adapter) if any(True for _ in self.adapter.parameters()) else "identity")
            log.info("     criterion: %s", self.criterion.describe())
        self.cos_metric = TokenFeatureLoss(metric="cosine")
        self.uses_cache = bool(self.cfg.get("cache_root"))

    def build_criterion(self) -> CompositeLoss:
        return CompositeLoss(self.cfg["criterion"])

    def setup(self, stage: str | None = None) -> None:
        """Load the teacher when training starts (skipped when every batch carries cached targets)."""
        if not self.uses_cache:
            _ = self.teacher
        else:
            log.info("     teacher targets come from the offline cache at %s; teacher weights not loaded", self.cfg["cache_root"])

    # -- forward pieces -------------------------------------------------------------------------
    def teacher_outputs(self, batch: DistillBatch) -> TeacherVisionOutput:
        if batch.cached_teacher is not None:
            return batch.cached_teacher
        return self.teacher.encode_vision(batch.teacher_inputs, sorted(self.layer_map.values()))

    def student_outputs(self, batch: DistillBatch) -> StudentVisionOutput:
        layers = sorted(self.layer_map)
        if getattr(self.student, "accepts_teacher_patches", False) and batch.teacher_inputs is not None:
            ti = batch.teacher_inputs  # Gemma-architecture students consume exactly the teacher's patch sequence
            return self.student(pixel_values=ti.pixel_values, pixel_position_ids=ti.pixel_position_ids, return_layers=layers)
        return self.student(batch.student_images, batch.valid_token_mask, return_layers=layers)

    def compute_losses(self, s_proj: StudentVisionOutput, t_out: TeacherVisionOutput) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.criterion(s_proj, t_out)

    def _step(self, batch: DistillBatch, split: str, group: str | None = None) -> torch.Tensor:
        t_out = self.teacher_outputs(batch)
        s_out = self.student_outputs(batch)
        if s_out.image_token_lengths != t_out.image_token_lengths:
            raise RuntimeError(f"token-count mismatch student {s_out.image_token_lengths} vs teacher {t_out.image_token_lengths}")
        s_proj = self.adapter(s_out)
        loss, logs = self.compute_losses(s_proj, t_out)
        logs["total"] = loss.detach()
        if split == "train":
            with torch.no_grad():
                logs["fid_vis_cos"] = 1.0 - self.cos_metric(s_out, t_out)
        else:  # the same fidelity metrics in every task (onboard_vlm.distill.fidelity)
            logs.update(fidelity_metrics(s_out.tokens, t_out.tokens, t_out.image_token_lengths, self.lm_projector()))
        self.log_losses(logs, split, len(batch), group)
        return loss

    def lm_projector(self):
        """The checkpoint's frozen embed_vision (None when targets come from an offline cache)."""
        return None if self.uses_cache else self.teacher.project_to_lm

    def export(self, directory: str) -> None:
        """Inference export: student weights + config (the adapter is discarded)."""
        from onboard_vlm.students.io import save_student

        save_student(directory, self.student, self.cfg["student"])
        log.info("     exported student (%s) to %s", describe_module(self.student), directory)
