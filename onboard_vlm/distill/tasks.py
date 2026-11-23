"""PyTorch-Lightning tasks (TimeFM-style) for Stage-1 feature KD and Stage-2 grafted logit KD."""

from __future__ import annotations

import math
from typing import Any, Mapping

import pytorch_lightning as pl
import torch
from torch import nn

from onboard_vlm.distill.adapter import DistillationAdapter
from onboard_vlm.distill.losses import CompositeLoss, TokenFeatureLoss
from onboard_vlm.interfaces import DistillBatch
from onboard_vlm.registry import build_loss, build_student, build_teacher
from onboard_vlm.utils.config import to_plain as _plain




def cosine_warmup_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    return fn


def build_optimizer(module: nn.Module, params: list[nn.Parameter] | None, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    """AdamW without weight decay on biases / norms / 1-D tensors."""
    cfg = dict(cfg)
    named = [(n, p) for n, p in module.named_parameters() if p.requires_grad and (params is None or any(p is q for q in params))]
    decay = [p for n, p in named if p.ndim >= 2 and "norm" not in n.lower()]
    no_decay = [p for n, p in named if not (p.ndim >= 2 and "norm" not in n.lower())]
    wd = cfg.pop("weight_decay", 0.05)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.pop("lr", 1e-3), betas=tuple(cfg.pop("betas", (0.9, 0.999))), **cfg,
    )


def _scheduler(trainer: pl.Trainer, opt: torch.optim.Optimizer, cfg: Mapping[str, Any]) -> dict[str, Any]:
    total = int(trainer.estimated_stepping_batches)
    warm = int(cfg.get("warmup_steps", 0) or cfg.get("warmup_ratio", 0.05) * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_warmup_lambda(warm, total, cfg.get("min_lr_ratio", 0.01)))
    return {"scheduler": sched, "interval": "step"}


def resolve_teacher_dim(teacher_cfg: Mapping[str, Any] | None, cache_root: str | None = None) -> int:
    """Teacher vision width from the cache metadata or the HF config (no weight download)."""
    if cache_root:
        import json
        import os

        with open(os.path.join(cache_root, "cache_meta.json")) as f:
            return int(json.load(f)["hidden_size"])
    cfg = dict(teacher_cfg or {})
    if cfg.get("tiny") is not None:
        from onboard_vlm.utils.tiny import tiny_gemma4_config

        return int(tiny_gemma4_config(**cfg["tiny"]).vision_config.hidden_size)
    if cfg.get("config") is not None:
        from transformers import Gemma4Config

        c = cfg["config"]
        c = Gemma4Config(**c) if isinstance(c, dict) else c
        return int(c.vision_config.hidden_size)
    from onboard_vlm.teachers.hub import load_manifest, resolve_teacher

    resolve_teacher(cfg["model_id"], cfg.get("revision"))  # raises for anything but E2B/E4B
    shape = load_manifest()["vision_tower"]["tensors"]["patch_embedder.input_proj.weight"]["shape"]
    return int(shape[0])  # 768


class Stage1FeatureDistillTask(pl.LightningModule):
    """Student <- frozen Gemma 4 vision tower, on unpadded soft tokens (+ mapped intermediate layers).

    Args:
        student: registry config of the student.
        teacher: registry config of the teacher (``{"name": "gemma4", "model_id": ..., "load": "vision"}``);
            may be ``None`` when every batch carries cached teacher features.
        criterion: list of weighted loss terms (see :class:`CompositeLoss`).
        layer_map: ``{student_idx: teacher_idx}`` for intermediate losses.
        adapter: kwargs for :class:`DistillationAdapter` (``kind``, ``force_final`` ...).
        cache_root: teacher-feature cache (used to read ``hidden_size`` when ``teacher`` is None).
        optimizer / scheduler: dicts (``lr``, ``weight_decay``, ``betas`` / ``warmup_ratio``, ``min_lr_ratio``).
    """

    def __init__(
        self,
        student: Mapping[str, Any],
        teacher: Mapping[str, Any] | None,
        criterion: list[Mapping[str, Any]],
        layer_map: Mapping[int, int] | None = None,
        adapter: Mapping[str, Any] | None = None,
        cache_root: str | None = None,
        optimizer: Mapping[str, Any] | None = None,
        scheduler: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        student, teacher, criterion = _plain(student), _plain(teacher), _plain(criterion)
        layer_map = {int(k): int(v) for k, v in (_plain(layer_map) or {}).items()}
        self.save_hyperparameters(
            {"student": student, "teacher": teacher, "criterion": criterion, "layer_map": layer_map,
             "adapter": _plain(adapter), "cache_root": cache_root, "optimizer": _plain(optimizer), "scheduler": _plain(scheduler)}
        )
        self.student_cfg = student
        self.student = build_student(student)
        self.layer_map = layer_map
        teacher_dim = resolve_teacher_dim(teacher, cache_root)
        self.adapter = DistillationAdapter.from_dims(self.student, teacher_dim, layer_map, **(_plain(adapter) or {}))
        self.criterion = CompositeLoss(criterion)
        self.teacher_cfg = teacher
        self._teacher: list[Any] = []  # kept out of the module tree: never saved / optimised / DDP-wrapped
        self.cos_metric = TokenFeatureLoss(metric="cosine")

    @property
    def teacher(self):
        if not self._teacher:
            if self.teacher_cfg is None:
                raise RuntimeError("batch has no cached teacher features and no teacher was configured")
            self._teacher.append(build_teacher(self.teacher_cfg))
        t = self._teacher[0]
        if next(t.parameters()).device != self.device:
            t.to(self.device)
        return t

    def transfer_batch_to_device(self, batch: DistillBatch, device: torch.device, dataloader_idx: int) -> DistillBatch:
        return batch.to(device)

    def _step(self, batch: DistillBatch, stage: str) -> torch.Tensor:
        t_layers = sorted(self.layer_map.values())
        if batch.cached_teacher is not None:
            t_out = batch.cached_teacher
        else:
            t_out = self.teacher.encode_vision(batch.teacher_inputs, t_layers)
        s_layers = sorted(self.layer_map)
        if getattr(self.student, "accepts_teacher_patches", False) and batch.teacher_inputs is not None:
            # Gemma-architecture students consume exactly the teacher's patch sequence.
            ti = batch.teacher_inputs
            s_out = self.student(pixel_values=ti.pixel_values, pixel_position_ids=ti.pixel_position_ids, return_layers=s_layers)
        else:
            s_out = self.student(batch.student_images, batch.valid_token_mask, return_layers=s_layers)
        if s_out.image_token_lengths != t_out.image_token_lengths:
            raise RuntimeError(f"token-count mismatch student {s_out.image_token_lengths} vs teacher {t_out.image_token_lengths}")
        s_proj = self.adapter(s_out)
        projector = self.teacher.project_to_lm_fp32 if self.criterion.uses_lm_space else None
        loss, logs = self.criterion(s_proj, t_out, lm_projector=projector)
        with torch.no_grad():
            logs["token_cos"] = 1.0 - self.cos_metric(s_proj, t_out)
        self.log_dict({f"{stage}/{k}": v for k, v in logs.items()}, batch_size=len(batch), prog_bar=stage == "val", sync_dist=stage == "val")
        if stage == "val":
            self.log("val_loss", logs["total"], batch_size=len(batch), sync_dist=True)
        return loss

    def training_step(self, batch: DistillBatch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: DistillBatch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        params = list(self.student.parameters()) + list(self.adapter.parameters())
        opt = build_optimizer(self, params, self.hparams.optimizer or {})
        return {"optimizer": opt, "lr_scheduler": _scheduler(self.trainer, opt, self.hparams.scheduler or {})}

    def export_student(self, directory: str) -> None:
        """Inference export: student weights only (the adapter is discarded)."""
        from onboard_vlm.students.io import save_student

        save_student(directory, self.student, self.student_cfg)


class Stage2GraftedKDTask(pl.LightningModule):
    """Graft the Stage-1 student into the frozen deployment VLM and distil next-token logits.

    Default (``train_student=True, train_embed_vision="none"``): the whole Gemma 4 VLM, including
    ``embed_vision`` and the language model, stays byte-identical; the VLM is used only as a
    differentiable loss, and gradients update the student vision encoder alone. Unfreezing
    ``embed_vision`` (full / LoRA) is available as an ablation but changes the deployed VLM.

    Args:
        vlm_id: deployment VLM (vision width must equal ``student.out_dim``), e.g. ``google/gemma-4-E2B-it``.
        student_dir: Stage-1 export (``student.safetensors`` + ``student_config.json``).
        logit_teacher_id: VLM providing target logits; ``None`` = the deployment VLM with its original
            vision tower (recommended, exact). Another supported id (e.g. ``google/gemma-4-E4B-it`` for an
            E2B deployment) is fed the *deployment* tokenizer's ids. E2B and E4B tokenizers are not
            identical (1 of 262,144 vocabulary entries and 14 BPE merges differ), so this is an
            approximation and logs a warning.
        train_embed_vision: ``"full"`` | ``"lora"`` | ``"none"``; ``train_student`` additionally unfreezes the encoder.
        criterion: :class:`GraftedLogitKDLoss` config.
        tiny_config: build random tiny models instead of downloading (tests / dry runs).
    """

    strict_loading = False  # checkpoints only hold the trainable parameters

    def __init__(
        self,
        vlm_id: str | None,
        student_dir: str | None,
        logit_teacher_id: str | None = None,
        train_embed_vision: str = "none",
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        train_student: bool = True,
        criterion: Mapping[str, Any] | None = None,
        optimizer: Mapping[str, Any] | None = None,
        scheduler: Mapping[str, Any] | None = None,
        dtype: str = "bfloat16",
        gradient_checkpointing: bool = True,
        tiny_config: Mapping[str, Any] | None = None,
        student: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        from onboard_vlm.vlm.grafted import GraftedVLM

        torch_dtype = getattr(torch, dtype)
        same_teacher = logit_teacher_id is None or logit_teacher_id == vlm_id
        if tiny_config is not None:  # offline dry run
            from transformers import Gemma4ForConditionalGeneration

            from onboard_vlm.utils.tiny import tiny_gemma4_config

            vlm = Gemma4ForConditionalGeneration(tiny_gemma4_config(**_plain(tiny_config))).to(torch_dtype)
            stu = build_student(_plain(student)).to(torch_dtype)
            self.grafted = GraftedVLM(vlm, stu, keep_teacher_tower=same_teacher)
        else:
            self.grafted = GraftedVLM.from_pretrained(vlm_id, student_dir, dtype=torch_dtype, keep_teacher_tower=same_teacher)
        self.grafted.configure_stage2(train_embed_vision, lora_rank, lora_alpha, train_student)
        if gradient_checkpointing and hasattr(self.grafted.vlm, "gradient_checkpointing_enable"):
            self.grafted.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self._logit_teacher: list[nn.Module] = []
        if not same_teacher:
            import logging

            from transformers import Gemma4ForConditionalGeneration

            logging.getLogger(__name__).warning(
                "logit teacher %s differs from deployment VLM %s: their tokenizers differ slightly "
                "(1 vocab entry, 14 merges); logits are compared on the deployment tokenization", logit_teacher_id, vlm_id)

            from onboard_vlm.teachers.hub import pinned_revision

            t = Gemma4ForConditionalGeneration.from_pretrained(
                logit_teacher_id, revision=pinned_revision(logit_teacher_id), dtype=torch_dtype
            ).eval().requires_grad_(False)
            self._logit_teacher.append(t)
        self.criterion = build_loss(_plain(criterion) or {"name": "grafted_logit_kd"})

    def transfer_batch_to_device(self, batch: DistillBatch, device: torch.device, dataloader_idx: int) -> DistillBatch:
        return batch.to(device)

    def _inputs(self, batch: DistillBatch) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
        ti = dict(batch.text_inputs or {})
        k = int(ti.pop("logits_to_keep"))
        mask, targets = ti.pop("kd_mask"), ti.pop("kd_targets")
        inputs = {**ti, **batch.teacher_inputs.vlm_kwargs(), "logits_to_keep": k, "use_cache": False}
        return inputs, mask, targets

    @torch.no_grad()
    def _teacher_logits(self, inputs: dict[str, Any]) -> torch.Tensor:
        if self._logit_teacher:
            t = self._logit_teacher[0]
            if next(t.parameters()).device != self.device:
                t.to(self.device)
            return t(**inputs).logits
        with self.grafted.teacher_vision():
            return self.grafted(**inputs).logits

    def _step(self, batch: DistillBatch, stage: str) -> torch.Tensor:
        inputs, mask, targets = self._inputs(batch)
        t_logits = self._teacher_logits(inputs)
        s_logits = self.grafted(**inputs).logits
        loss, logs = self.criterion(s_logits, t_logits, mask, targets)
        self.log_dict({f"{stage}/{k}": v for k, v in logs.items()}, batch_size=len(batch), sync_dist=stage == "val")
        if stage == "val":
            self.log("val_loss", loss, batch_size=len(batch), sync_dist=True)
        return loss

    def training_step(self, batch: DistillBatch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: DistillBatch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        params = [p for p in self.grafted.parameters() if p.requires_grad]
        opt = build_optimizer(self, params, self.hparams.optimizer or {})
        return {"optimizer": opt, "lr_scheduler": _scheduler(self.trainer, opt, self.hparams.scheduler or {})}

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if k in trainable}

    def export(self, directory: str) -> None:
        """Saves the (possibly fine-tuned) student and the trainable ``embed_vision`` weights."""
        import os

        from safetensors.torch import save_file

        from onboard_vlm.students.io import load_student, save_student

        os.makedirs(directory, exist_ok=True)
        cfg = _plain(self.hparams.student)
        if cfg is None and self.hparams.student_dir:
            _, cfg = load_student(self.hparams.student_dir)
        save_student(directory, self.grafted.student_tower.student, cfg)
        save_file(self.grafted.embed_vision_state_dict(), os.path.join(directory, "embed_vision.safetensors"))
