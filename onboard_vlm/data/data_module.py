"""Lightning DataModule for all distillation tasks (mirrors TimeFM's EEGDataModule), with
real-time logging of every dataset that is set up."""

from __future__ import annotations

from typing import Any, Sequence

import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader

from onboard_vlm.data.collator import DualViewCollator
from onboard_vlm.data.samplers import MixtureSampler, SizeBucketBatchSampler
from onboard_vlm.registry import build_dataset
from onboard_vlm.utils.config import to_plain
from onboard_vlm.utils.logging import get_logger
from onboard_vlm.utils.logging import stage as log_stage

log = get_logger(__name__)


def build_text_builder(cfg: dict[str, Any] | None):
    """``{processor_id, revision?, prompt, with_answer, system_prompt}`` -> ``Gemma4ChatInputs`` (or None)."""
    if not cfg:
        return None
    from transformers import AutoProcessor

    from onboard_vlm.data.prompts import Gemma4ChatInputs
    from onboard_vlm.teachers.gemma4 import resolve_revision

    cfg = dict(cfg)
    pid = cfg.pop("processor_id")
    if pid == "tiny":  # offline dry runs / tests
        from onboard_vlm.utils.tiny import tiny_gemma4_processor

        cfg.pop("revision", None)
        log.info("     using the tiny offline Gemma 4 processor (dry run)")
        return Gemma4ChatInputs(tiny_gemma4_processor(), **cfg)
    rev = resolve_revision(pid, cfg.pop("revision", None))
    log.info("     loading original Gemma 4 processor %s@%s (chat template, image-token expansion)", pid, rev[:7])
    processor = AutoProcessor.from_pretrained(pid, revision=rev)
    return Gemma4ChatInputs(processor, **cfg)


def _describe(cfg: dict[str, Any]) -> str:
    keys = ("name", "path", "root", "split", "mode")
    return ", ".join(f"{k}={cfg[k]}" for k in keys if k in cfg)


class DistillDataModule(pl.LightningDataModule):
    """
    Args:
        train / val: lists of dataset configs (``{"name": "image_corpus", "root": ...}``, ...). A train
            entry may carry ``weight`` (its mixture probability) and ``group`` (a label for logging).
        collator / val_collator: kwargs for :class:`DualViewCollator`.
        text: text-builder config for logit distillation (see :func:`build_text_builder`).
        mixture: named component weights that ``train`` entries reference by interpolation
            (``weight: ${data_module.mixture.cc12m}``); only logged here.
        drop_groups: groups whose entries get weight 0 (corpus ablations, e.g. ``[egocentric]``).
        mix_weights: relative sampling weight per training dataset; overrides per-entry ``weight``.
            ``None`` and no ``weight`` keys = uniform over the concatenation (proportional to size).
        samples_per_epoch: epoch length when weights are used (default: total size).
        bucket_by_size: group equal-target-size images (only meaningful with deterministic geometry).
    """

    def __init__(
        self,
        train: Sequence[dict[str, Any]],
        val: Sequence[dict[str, Any]] | None = None,
        collator: dict[str, Any] | None = None,
        val_collator: dict[str, Any] | None = None,
        text: dict[str, Any] | None = None,
        batch_size: int = 32,
        val_batch_size: int | None = None,
        num_workers: int = 8,
        mixture: dict[str, float] | None = None,
        drop_groups: Sequence[str] = (),
        mix_weights: Sequence[float] | None = None,
        samples_per_epoch: int | None = None,
        bucket_by_size: bool = False,
        pin_memory: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        train, val, collator, val_collator, text, mixture = map(to_plain, (train, val, collator, val_collator, text, mixture))
        self.mixture = mixture
        self.train_cfgs, self.val_cfgs = [dict(c) for c in train], list(val or [])
        entry_w = [c.pop("weight", None) for c in self.train_cfgs]
        self.groups = [c.pop("group", None) for c in self.train_cfgs]
        if mix_weights is None and any(w is not None for w in entry_w):
            if any(w is None for w in entry_w):
                raise ValueError("give every train entry a weight (or none of them)")
            mix_weights = entry_w
        self.drop_groups = list(to_plain(drop_groups) or [])
        unknown = set(self.drop_groups) - set(self.groups)
        if unknown:
            raise ValueError(f"drop_groups {sorted(unknown)} not among the train groups {sorted({g for g in self.groups if g})}")
        if self.drop_groups:
            if mix_weights is None:
                raise ValueError("drop_groups needs mixture weights (per-entry `weight` or mix_weights)")
            mix_weights = [0.0 if g in self.drop_groups else w for g, w in zip(self.groups, mix_weights)]
        self.collator_cfg = dict(collator or {})
        self.val_collator_cfg = dict(val_collator or {**self.collator_cfg, "train": False})
        self.text_cfg = text
        self.batch_size, self.val_batch_size = batch_size, val_batch_size or batch_size
        self.num_workers, self.pin_memory, self.seed = num_workers, pin_memory, seed
        self.mix_weights = to_plain(mix_weights)
        self.samples_per_epoch = samples_per_epoch
        self.bucket_by_size = bucket_by_size
        self.train_set = self.val_set = None
        self._is_setup = False

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002  (Lightning passes stage=)
        if self._is_setup:
            return
        with log_stage("Setting up datasets", log):
            if self.mixture:
                log.info("     mixture: %s", ", ".join(f"{k}={v}" for k, v in self.mixture.items()))
            if self.drop_groups:
                log.info("     ablation: dropping groups %s (weight 0)", self.drop_groups)
            text_builder = build_text_builder(self.text_cfg)
            self.train_parts = []
            for i, c in enumerate(self.train_cfgs):
                ds = build_dataset(c)
                w = f" | weight {self.mix_weights[i]}" if self.mix_weights else ""
                g = f"[{self.groups[i]}] " if self.groups[i] else ""
                log.info("     train[%d] %s%s -> %s samples%s", i, g, _describe(c), f"{len(ds):,}", w)
                self.train_parts.append(ds)
            self.train_set = ConcatDataset(self.train_parts)
            self.train_collate = DualViewCollator(**self.collator_cfg, text_builder=text_builder, seed=self.seed)
            log.info("     train total %s samples | collator %s", f"{len(self.train_set):,}", dict(self.collator_cfg))
            if self.mix_weights is not None:
                self._log_mixture()
            if self.val_cfgs:
                parts = []
                for i, c in enumerate(self.val_cfgs):
                    ds = build_dataset(c)
                    log.info("     val[%d]   %s -> %s samples", i, _describe(c), f"{len(ds):,}")
                    parts.append(ds)
                self.val_set = ConcatDataset(parts)
                self.val_collate = DualViewCollator(**self.val_collator_cfg, text_builder=text_builder, seed=self.seed + 1)
            log.info("     batch_size=%d (val %d) | num_workers=%d | mix_weights=%s | samples_per_epoch=%s",
                     self.batch_size, self.val_batch_size, self.num_workers, self.mix_weights, self.samples_per_epoch)
        self._is_setup = True

    def _log_mixture(self) -> None:
        if len(self.mix_weights) != len(self.train_parts):
            raise ValueError(f"mix_weights has {len(self.mix_weights)} entries for {len(self.train_parts)} training datasets")
        total = float(sum(self.mix_weights))
        n = self.samples_per_epoch or len(self.train_set)
        by_group: dict[str, float] = {}
        for i, (d, w) in enumerate(zip(self.train_parts, self.mix_weights)):
            p = w / total
            by_group[self.groups[i] or f"train[{i}]"] = by_group.get(self.groups[i] or f"train[{i}]", 0.0) + p
            log.info("     mixture train[%d]: p=%.3f -> %s draws/epoch = %.2f passes over its %s images", i, p, f"{round(p * n):,}",
                     p * n / max(len(d), 1), f"{len(d):,}")
        log.info("     mixture by group: %s", ", ".join(f"{k}={v:.3f}" for k, v in by_group.items()))

    def _loader_kwargs(self) -> dict[str, Any]:
        return dict(num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.num_workers > 0)

    def train_dataloader(self) -> DataLoader:
        if self.bucket_by_size:
            sizes = [s for d in self.train_parts for s in getattr(d, "sizes", [None] * len(d))]
            budget = self.collator_cfg.get("token_budgets", 280)
            budget = budget if isinstance(budget, int) else list(budget)[0]
            sampler = SizeBucketBatchSampler(sizes, self.batch_size, budget, shuffle=True, drop_last=True, seed=self.seed)
            return DataLoader(self.train_set, batch_sampler=sampler, collate_fn=self.train_collate, **self._loader_kwargs())
        if self.mix_weights is not None:
            sampler = MixtureSampler([len(d) for d in self.train_parts], self.mix_weights,
                                     self.samples_per_epoch or len(self.train_set), seed=self.seed)
            return DataLoader(self.train_set, batch_size=self.batch_size, sampler=sampler, drop_last=True, collate_fn=self.train_collate, **self._loader_kwargs())
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=self.train_collate, **self._loader_kwargs())

    def val_dataloader(self) -> DataLoader | list:
        if self.val_set is None:
            return []
        return DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False, collate_fn=self.val_collate, **self._loader_kwargs())
