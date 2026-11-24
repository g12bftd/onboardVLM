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


SPLIT_KEYS = ("val_fraction", "test_fraction", "split_seed")


class DistillDataModule(pl.LightningDataModule):
    """Builds every dataset from YAML entries; weights, groups and splits are configuration only.

    Each entry is one dataset (one benchmark LMDB, or one corpus source directory). Nothing about an
    experiment is stored in the data files: the mixture weight of a train entry, the group it is
    reported under and (for corpus sources) its train / val / test split all come from the config,
    so ablations are CLI overrides (see ``docs/corpus.md``).

    Args:
        train / val / test: lists of dataset configs (``{"name": "image_corpus", "root": ...}``).
            Every entry may carry ``group`` (a label). A train entry may carry ``weight`` (mixture
            probability). Val / test entries with the same group are concatenated, and every group
            gets its own dataloader, so metrics are logged per group (``val/<group>/...``) and overall.
        dataset_defaults: ``{dataset_name: {key: value}}`` merged under every entry of that dataset
            (entry keys win), e.g. one split definition for all ``image_corpus`` entries.
        collator / val_collator: kwargs for :class:`DualViewCollator` (val_collator also serves test).
        text: text-builder config for logit distillation (see :func:`build_text_builder`).
        mixture: named component weights that ``train`` entries reference by interpolation
            (``weight: ${data_module.mixture.cc12m}``); only logged here.
        drop_groups: train groups whose entries get weight 0 (e.g. ``[egocentric]``).
        mix_weights: relative sampling weight per training dataset; overrides per-entry ``weight``.
            ``None`` and no ``weight`` keys = uniform over the concatenation (proportional to size).
        samples_per_epoch: epoch length when weights are used (default: total size).
        bucket_by_size: group equal-target-size images (only meaningful with deterministic geometry).
    """

    def __init__(
        self,
        train: Sequence[dict[str, Any]],
        val: Sequence[dict[str, Any]] | None = None,
        test: Sequence[dict[str, Any]] | None = None,
        dataset_defaults: dict[str, dict[str, Any]] | None = None,
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
        train, val, test, defaults, collator, val_collator, text, mixture = map(
            to_plain, (train, val, test, dataset_defaults, collator, val_collator, text, mixture)
        )
        self.mixture = mixture
        self.dataset_defaults = dict(defaults or {})
        merge = lambda c: {**self.dataset_defaults.get(c["name"], {}), **c}  # noqa: E731
        self.train_cfgs = [merge(c) for c in train]
        self.val_cfgs, self.test_cfgs = [merge(c) for c in val or []], [merge(c) for c in test or []]
        _check_corpus_splits(self.train_cfgs, self.val_cfgs, self.test_cfgs)
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
        self.train_set = None
        self.val_sets: dict[str, ConcatDataset] = {}
        self.test_sets: dict[str, ConcatDataset] = {}
        self._is_setup = False

    @property
    def val_groups(self) -> list[str]:
        return list(self.val_sets)

    @property
    def test_groups(self) -> list[str]:
        return list(self.test_sets)

    def setup(self, stage: str | None = None) -> None:
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
                log.info("     train[%d] %s%s -> %s samples%s%s", i, g, _describe(c), f"{len(ds):,}", w, _split_info(ds))
                self.train_parts.append(ds)
            self.train_set = ConcatDataset(self.train_parts)
            self.train_collate = DualViewCollator(**self.collator_cfg, text_builder=text_builder, seed=self.seed)
            log.info("     train total %s samples | collator %s", f"{len(self.train_set):,}", dict(self.collator_cfg))
            if self.mix_weights is not None:
                self._log_mixture()
            self.val_sets = self._build_eval_sets("val", self.val_cfgs)
            self.test_sets = self._build_eval_sets("test", self.test_cfgs)
            self.eval_collate = DualViewCollator(**self.val_collator_cfg, text_builder=text_builder, seed=self.seed + 1)
            log.info("     batch_size=%d (eval %d) | num_workers=%d | samples_per_epoch=%s",
                     self.batch_size, self.val_batch_size, self.num_workers, self.samples_per_epoch)
        self._is_setup = True

    def _build_eval_sets(self, split: str, cfgs: list[dict[str, Any]]) -> dict[str, ConcatDataset]:
        parts: dict[str, list] = {}
        for i, c in enumerate(cfgs):
            c = dict(c)
            group = c.pop("group", None) or f"{split}{i}"
            ds = build_dataset(c)
            log.info("     %s[%d] [%s] %s -> %s samples", split, i, group, _describe(c), f"{len(ds):,}")
            parts.setdefault(group, []).append(ds)
        sets = {g: ConcatDataset(p) for g, p in parts.items()}
        if sets:
            log.info("     %s groups: %s", split, ", ".join(f"{g}={len(d):,}" for g, d in sets.items()))
        return sets

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

    def _eval_loaders(self, sets: dict[str, ConcatDataset]) -> list[DataLoader]:
        return [DataLoader(d, batch_size=self.val_batch_size, shuffle=False, collate_fn=self.eval_collate, **self._loader_kwargs()) for d in sets.values()]

    def val_dataloader(self) -> list[DataLoader]:
        return self._eval_loaders(self.val_sets)

    def test_dataloader(self) -> list[DataLoader]:
        return self._eval_loaders(self.test_sets)


def _split_info(ds: Any) -> str:
    counts = getattr(ds, "split_counts", None)
    return f" | split {ds.split} of {counts}" if counts else ""


def _check_corpus_splits(train: list[dict], val: list[dict], test: list[dict]) -> None:
    """Corpus splits are computed in the loader, so two entries reading the same source with different
    split definitions (or a val entry reading the train split) would leak. Reject both."""
    seen: dict[str, tuple] = {}
    for expected, cfgs in (("train", train), ("val", val), ("test", test)):
        for c in cfgs:
            if c.get("name") != "image_corpus":
                continue
            split = c.get("split", "train")
            if split != expected:
                raise ValueError(f"an image_corpus entry in `{expected}` reads split {split!r}: {c.get('root')}")
            definition = tuple(c.get(k) for k in SPLIT_KEYS)
            roots = c["root"] if isinstance(c["root"], (list, tuple)) else [c["root"]]
            for r in roots:
                if seen.setdefault(str(r), definition) != definition:
                    raise ValueError(f"conflicting split definitions {SPLIT_KEYS} for {r}: {seen[str(r)]} vs {definition}; "
                                     "set them once in data_module.dataset_defaults.image_corpus")
