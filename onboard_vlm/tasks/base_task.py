"""Shared machinery of every distillation task (TimeFM convention).

Tasks are built as ``hydra.utils.instantiate(cfg.task, cfg)``, i.e. ``Task(hparams, **task_cfg)``:
``hparams`` is the full run config (student / teacher / criterion / layer_map ...) and ``task_cfg``
holds the task's own options from ``config/task/<task>.yaml``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import pytorch_lightning as pl
import torch
from torch import nn

from onboard_vlm.interfaces import DistillBatch
from onboard_vlm.registry import build_student, build_teacher
from onboard_vlm.utils.config import to_plain
from onboard_vlm.utils.logging import describe_module, get_logger, stage

log = get_logger("onboard_vlm.tasks")


def cosine_warmup_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    return fn


def teacher_vision_dim(teacher_cfg: Mapping[str, Any]) -> int:
    """Vision width of the configured teacher, from its original ``config.json`` (no weights needed)."""
    cfg = dict(teacher_cfg)
    if cfg.get("tiny") is not None:
        from onboard_vlm.utils.tiny import tiny_gemma4_config

        return int(tiny_gemma4_config(**cfg["tiny"]).vision_config.hidden_size)
    from transformers import AutoConfig

    from onboard_vlm.teachers.gemma4 import resolve_revision

    c = AutoConfig.from_pretrained(cfg["model_id"], revision=resolve_revision(cfg["model_id"], cfg.get("revision")))
    return int(c.vision_config.hidden_size)


class BaseDistillationTask(pl.LightningModule):
    """Base class: student / teacher construction with stage logging, optimisation, device transfer.

    Args:
        hparams: full run config.
        optimizer: ``{lr, weight_decay, betas}`` (AdamW; no decay on biases / norms / 1-D tensors).
        scheduler: ``{warmup_ratio | warmup_steps, min_lr_ratio}`` (linear warmup + cosine, per step).
    """

    distillation_type = "base"

    def __init__(self, hparams: Any, optimizer: Mapping[str, Any] | None = None, scheduler: Mapping[str, Any] | None = None):
        super().__init__()
        cfg = to_plain(hparams)
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.optimizer_cfg = dict(to_plain(optimizer) or {})
        self.scheduler_cfg = dict(to_plain(scheduler) or {})
        self._teacher: list[nn.Module] = []  # kept outside the module tree: never saved / optimised / DDP-wrapped
        self._eval_sums: dict[str, dict[str, list[float]]] = {}
        log.info("     distillation type: %s (%s)", self.distillation_type, type(self).__name__)

    # -- construction ---------------------------------------------------------------------------
    def build_student(self) -> nn.Module:
        from onboard_vlm.geometry import InputDomain, check_student_covers

        scfg = self.cfg["student"]
        with stage(f"Building student '{scfg['name']}'", log):
            student = build_student(scfg)
            domain = InputDomain.from_cfg(self.cfg.get("input_domain"))
            check_student_covers(student, domain)
            log.info("     input domain: %s (student covers it)", domain.describe())
            return student

    def load_teacher(self, load: str = "vision") -> nn.Module:
        tcfg = {**self.cfg["teacher"], "load": load}
        with stage(f"Loading teacher {tcfg.get('model_id', 'tiny')} (load={load})", log):
            return build_teacher(tcfg)

    @property
    def teacher(self):
        if not self._teacher:
            self._teacher.append(self.load_teacher(self.teacher_load_mode))
        t = self._teacher[0]
        if next(t.parameters()).device != self.device:
            log.info("     moving teacher to %s", self.device)
            t.to(self.device)
        return t

    teacher_load_mode = "vision"

    def transfer_batch_to_device(self, batch: DistillBatch, device: torch.device, dataloader_idx: int) -> DistillBatch:
        return batch.to(device)

    # -- optimisation ---------------------------------------------------------------------------
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def configure_optimizers(self):
        params = self.trainable_parameters()
        ids = {id(p) for p in params}
        named = [(n, p) for n, p in self.named_parameters() if id(p) in ids]
        decay = [p for n, p in named if p.ndim >= 2 and "norm" not in n.lower()]
        no_decay = [p for n, p in named if not (p.ndim >= 2 and "norm" not in n.lower())]
        cfg = dict(self.optimizer_cfg)
        wd = cfg.pop("weight_decay", 0.05)
        lr = cfg.pop("lr", 1e-3)
        betas = tuple(cfg.pop("betas", (0.9, 0.999)))
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=betas, **cfg)
        total = int(self.trainer.estimated_stepping_batches)
        warm = int(self.scheduler_cfg.get("warmup_steps", 0) or self.scheduler_cfg.get("warmup_ratio", 0.05) * total)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_warmup_lambda(warm, total, self.scheduler_cfg.get("min_lr_ratio", 0.01)))
        log.info("     optimizer AdamW lr=%g wd=%g betas=%s | %d tensors with decay, %d without | %.2fM trainable params",
                 lr, wd, betas, len(decay), len(no_decay), sum(p.numel() for p in params) / 1e6)
        log.info("     schedule: %d warmup steps + cosine to %g x lr over %d steps", warm, self.scheduler_cfg.get("min_lr_ratio", 0.01), total)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    # -- logging --------------------------------------------------------------------------------
    # -- steps and logging --------------------------------------------------------------------
    def _step(self, batch: DistillBatch, split: str, group: str | None = None) -> torch.Tensor:  # pragma: no cover - overridden
        raise NotImplementedError

    def training_step(self, batch: DistillBatch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: DistillBatch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self._step(batch, "val", self._eval_group("val", dataloader_idx))

    def test_step(self, batch: DistillBatch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        return self._step(batch, "test", self._eval_group("test", dataloader_idx))

    def _eval_group(self, split: str, dataloader_idx: int) -> tuple | None:
        dm = getattr(self.trainer, "datamodule", None) if self._trainer is not None else None
        groups = getattr(dm, f"{split}_groups", None)
        return groups[dataloader_idx] if groups and dataloader_idx < len(groups) else None

    def log_losses(self, logs: Mapping[str, torch.Tensor], split: str, batch_size: int, group: tuple | str | None = None) -> None:
        """Train: per-step ``train/<k>``. Val / test: accumulated per (group, budget) and logged at
        epoch end as ``<split>/<group>/b<budget>/<k>``, ``<split>/<group>/<k>``, ``<split>/b<budget>/<k>``
        and ``<split>/<k>`` (image-weighted means; ``val_loss`` = ``val/total``)."""
        if split == "train":  # mean over ranks (every rank logs the same terms every step)
            self.log_dict({f"train/{k}": v for k, v in logs.items()}, batch_size=batch_size, sync_dist=True)
            return
        g, b = group if isinstance(group, tuple) else (group, None)
        acc = self._eval_sums.setdefault(split, {})
        for k, v in logs.items():
            a = acc.setdefault((g, b, k), [0.0, 0.0])
            a[0] += float(v) * batch_size
            a[1] += batch_size

    def _log_eval_totals(self, split: str) -> None:
        acc = self._eval_sums.pop(split, {})
        if self._trainer is not None and self.trainer.world_size > 1:  # every rank calls this (collective)
            gathered: list = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(gathered, acc)
            acc = {}
            for part in gathered:
                for key, (s, n) in part.items():
                    a = acc.setdefault(key, [0.0, 0.0])
                    a[0] += s
                    a[1] += n
        agg: dict[str, list[float]] = {}
        for (g, b, k), (s, n) in acc.items():
            names = [f"{split}/{k}"]
            if g is not None:
                names.append(f"{split}/{g}/{k}")
            if b is not None:
                names.append(f"{split}/b{b}/{k}")
            if g is not None and b is not None:
                names.append(f"{split}/{g}/b{b}/{k}")
            for name in names:
                a = agg.setdefault(name, [0.0, 0.0])
                a[0] += s
                a[1] += n
        if not agg:
            return
        means = {k: torch.tensor(s / max(n, 1.0), dtype=torch.float32) for k, (s, n) in agg.items()}
        self.log_dict(means, prog_bar=False)
        self.eval_report = {"split": split, "metrics": {k: float(v) for k, v in means.items()},
                            "num_images": {k: n for k, (_, n) in agg.items() if k.endswith("/total")}}
        if split == "val" and "val/total" in means:
            self.log("val_loss", means["val/total"], prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        self._eval_sums.pop("val", None)

    def on_validation_epoch_end(self) -> None:
        self._log_eval_totals("val")

    def on_test_epoch_start(self) -> None:
        self._eval_sums.pop("test", None)

    def on_test_epoch_end(self) -> None:
        self._log_eval_totals("test")

    def describe(self) -> None:
        for name, mod in self.named_children():
            log.info("     module %-10s %s", name, describe_module(mod) if any(True for _ in mod.parameters()) else type(mod).__name__)

    def export(self, directory: str) -> None:  # pragma: no cover - overridden
        raise NotImplementedError
