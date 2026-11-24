"""Distillation type 2 - embedding distillation in the language model's input space.

Student and teacher tokens are both passed through the original, frozen ``embed_vision`` of the
chosen Gemma 4 checkpoint (scale-free RMSNorm + 768->1536 projection for E2B, 768->2560 for E4B),
so the loss is measured in exactly the space the language model reads. The language model itself
is never loaded. Optional vision-space terms (``vision_criterion``) can be added.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from onboard_vlm.distill.losses import CompositeLoss
from onboard_vlm.interfaces import StudentVisionOutput, TeacherVisionOutput
from onboard_vlm.tasks.base_task import log
from onboard_vlm.tasks.feature_distillation import FeatureDistillationTask
from onboard_vlm.utils.config import to_plain


class EmbeddingDistillationTask(FeatureDistillationTask):
    """``embed_vision(student tokens)`` <- ``embed_vision(teacher tokens)`` (original frozen projection).

    ``hparams.criterion`` lists the LM-space terms (final tokens only).

    Args:
        vision_criterion: optional extra terms on the 768-d vision tokens (may include intermediate layers).
    """

    distillation_type = "embedding"

    def __init__(
        self,
        hparams: Any,
        vision_criterion: Sequence[Mapping[str, Any]] | None = None,
        adapter: Mapping[str, Any] | None = None,
        optimizer: Mapping[str, Any] | None = None,
        scheduler: Mapping[str, Any] | None = None,
    ):
        super().__init__(hparams, adapter, optimizer, scheduler)
        vc = to_plain(vision_criterion)
        self.vision_criterion = CompositeLoss(vc) if vc else None
        if self.vision_criterion is not None:
            log.info("     + vision-space criterion: %s", self.vision_criterion.describe())

    def build_criterion(self) -> CompositeLoss:
        terms = self.cfg["criterion"]
        if any(t.get("name") == "intermediate_layer" for t in terms):
            raise ValueError("embedding_distillation compares final tokens after embed_vision; put intermediate_layer terms in task.vision_criterion")
        return CompositeLoss(terms)

    def lm_projector(self):
        return self.teacher.project_to_lm  # embed_vision is always loaded for this task

    def setup(self, stage: str | None = None) -> None:
        _ = self.teacher  # embed_vision is always needed (also with cached vision-token targets)

    def compute_losses(self, s_proj: StudentVisionOutput, t_out: TeacherVisionOutput) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        t = self.teacher
        s_lm = t.project_to_lm(s_proj.tokens)  # gradients flow to the student; projection weights are frozen
        with torch.no_grad():
            t_lm = t.project_to_lm(t_out.tokens)
        loss, logs = self.criterion(StudentVisionOutput(s_lm, s_proj.image_token_lengths), TeacherVisionOutput(t_lm, t_out.image_token_lengths), prefix="lm_")
        if self.vision_criterion is not None:
            v_loss, v_logs = self.vision_criterion(s_proj, t_out, prefix="vis_")
            loss = loss + v_loss
            logs.update(v_logs)
        return loss, logs
