"""Distillation tasks (one per distillation type, TimeFM-style ``Task(hparams, **task_cfg)``)."""

from onboard_vlm.tasks.base_task import BaseDistillationTask
from onboard_vlm.tasks.embedding_distillation import EmbeddingDistillationTask
from onboard_vlm.tasks.feature_distillation import FeatureDistillationTask
from onboard_vlm.tasks.logit_distillation import LogitDistillationTask

__all__ = ["BaseDistillationTask", "EmbeddingDistillationTask", "FeatureDistillationTask", "LogitDistillationTask"]
