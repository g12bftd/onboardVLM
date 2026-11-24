"""Distillation type 3 - logit distillation through the original Gemma 4 VLM.

The student replaces the vision tower inside the original ``Gemma4ForConditionalGeneration``; the
VLM is used only as a differentiable loss. Target logits come from the same VLM with its original
vision tower. By default only the student is trained, so the deployed VLM (``embed_vision`` +
language model) stays unchanged. KL is computed on the answer span of teacher-forced VQA samples.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from onboard_vlm.interfaces import DistillBatch
from onboard_vlm.registry import build_loss
from onboard_vlm.tasks.base_task import BaseDistillationTask, log
from onboard_vlm.utils.logging import describe_module, stage


class LogitDistillationTask(BaseDistillationTask):
    """KL(original VLM || VLM with grafted student) on next-token logits.

    Reads ``hparams.teacher`` (deployment VLM), ``hparams.student`` (used when ``student_dir`` is None)
    and ``hparams.criterion`` (``grafted_logit_kd``).

    Args:
        student_dir: a feature / embedding distillation export to start from (recommended).
        logit_teacher_id: VLM giving target logits; ``None`` = same VLM with its original vision tower.
            A different supported id (E4B logits for an E2B deployment) is approximate: the E2B and
            E4B tokenizers differ in 1 of 262,144 vocabulary entries and 14 BPE merges.
        train_student: update the student encoder (default).
        train_embed_vision: ``"none"`` (default) | ``"lora"`` | ``"full"`` (ablations that change the VLM).
        gradient_checkpointing: inside the language model.
        dtype: VLM dtype.
    """

    distillation_type = "logit"
    strict_loading = False  # checkpoints hold only the trainable tensors
    teacher_load_mode = "vlm"

    def __init__(
        self,
        hparams: Any,
        student_dir: str | None = None,
        logit_teacher_id: str | None = None,
        train_student: bool = True,
        train_embed_vision: str = "none",
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        gradient_checkpointing: bool = True,
        dtype: str = "bfloat16",
        optimizer: Mapping[str, Any] | None = None,
        scheduler: Mapping[str, Any] | None = None,
    ):
        super().__init__(hparams, optimizer, scheduler)
        from onboard_vlm.vlm.grafted import GraftedVLM

        tcfg = dict(self.cfg["teacher"])
        torch_dtype = getattr(torch, dtype)
        same_teacher = logit_teacher_id is None or logit_teacher_id == tcfg.get("model_id")
        if tcfg.get("tiny") is not None:
            from transformers import Gemma4ForConditionalGeneration

            from onboard_vlm.utils.tiny import tiny_gemma4_config

            with stage("Building tiny random Gemma 4 VLM (tests / dry runs)", log):
                vlm = Gemma4ForConditionalGeneration(tiny_gemma4_config(**tcfg["tiny"])).to(torch_dtype).eval()
        else:
            from onboard_vlm.teachers.gemma4 import load_gemma4_vlm

            vlm = load_gemma4_vlm(tcfg["model_id"], tcfg.get("revision"), torch_dtype, tcfg.get("attn_implementation", "sdpa"))
        if student_dir:
            from onboard_vlm.students.io import load_student

            with stage(f"Loading distilled student from {student_dir}", log):
                student, scfg = load_student(student_dir)
                self.cfg["student"] = scfg
        else:
            student = self.build_student()
        with stage("Grafting student into the VLM and freezing", log):
            self.grafted = GraftedVLM(vlm, student.to(torch_dtype), keep_teacher_tower=same_teacher)
            trainable = self.grafted.configure_stage2(train_embed_vision, lora_rank, lora_alpha, train_student)
            if gradient_checkpointing:
                self.grafted.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            log.info("     trainable: %.2fM params (student=%s, embed_vision=%s) of %s", sum(p.numel() for p in trainable) / 1e6,
                     train_student, train_embed_vision, describe_module(self.grafted))
        self._logit_teacher: list[nn.Module] = []
        if not same_teacher:
            from onboard_vlm.teachers.gemma4 import load_gemma4_vlm

            log.warning("logit teacher %s != deployment VLM %s: tokenizers differ slightly; logits compared on the deployment tokenization", logit_teacher_id, tcfg.get("model_id"))
            self._logit_teacher.append(load_gemma4_vlm(logit_teacher_id, None, torch_dtype).requires_grad_(False))
        self.criterion = build_loss(self.cfg["criterion"])
        log.info("     criterion: %s (T=%s, ce_weight=%s)", type(self.criterion).__name__, self.criterion.temperature, self.criterion.ce_weight)

    @property
    def student(self) -> nn.Module:
        return self.grafted.student_tower.student

    def _inputs(self, batch: DistillBatch) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
        ti = dict(batch.text_inputs or {})
        k = int(ti.pop("logits_to_keep"))
        mask, targets = ti.pop("kd_mask"), ti.pop("kd_targets")
        return {**ti, **batch.teacher_inputs.vlm_kwargs(), "logits_to_keep": k, "use_cache": False}, mask, targets

    @torch.no_grad()
    def _teacher_logits(self, inputs: dict[str, Any]) -> torch.Tensor:
        if self._logit_teacher:
            t = self._logit_teacher[0]
            if next(t.parameters()).device != self.device:
                t.to(self.device)
            return t(**inputs).logits
        with self.grafted.teacher_vision():
            return self.grafted(**inputs).logits

    def _step(self, batch: DistillBatch, split: str) -> torch.Tensor:
        inputs, mask, targets = self._inputs(batch)
        t_logits = self._teacher_logits(inputs)
        s_logits = self.grafted(**inputs).logits
        loss, logs = self.criterion(s_logits, t_logits, mask, targets)
        logs["total"] = loss.detach()
        self.log_losses(logs, split, len(batch))
        return loss

    def training_step(self, batch: DistillBatch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: DistillBatch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if k in trainable}

    def export(self, directory: str) -> None:
        """Student weights (+ ``embed_vision`` when it was trained, LoRA merged)."""
        import os

        from safetensors.torch import save_file

        from onboard_vlm.students.io import save_student

        save_student(directory, self.student, self.cfg["student"])
        if any(p.requires_grad for p in self.grafted.vlm.model.embed_vision.parameters()):
            os.makedirs(directory, exist_ok=True)
            save_file(self.grafted.embed_vision_state_dict(), os.path.join(directory, "embed_vision.safetensors"))
        log.info("     exported %s to %s", describe_module(self.student), directory)
