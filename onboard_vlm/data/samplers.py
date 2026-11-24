"""Size-bucketing batch sampler.

With deterministic (eval / cached) geometry, images whose Gemma target size coincides can be
batched together so that the dense student batch has *no* spatial padding. This makes CNN
students (which are not batch invariant) see exactly what they would see one image at a time.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterator, Sequence

import torch.distributed as dist
from torch.utils.data import Sampler

from onboard_vlm.geometry import target_size


class SizeBucketBatchSampler(Sampler[list[int]]):
    """Batches indices whose ``target_size(h, w, budget)`` is identical (DDP-aware).

    Args:
        sizes: per-index source ``(h, w)`` (``None`` entries go to a shared fallback bucket).
        batch_size: images per batch.
        token_budget: budget used to compute the target size.
        shuffle, drop_last, seed: usual semantics; call :meth:`set_epoch` each epoch.
    """

    def __init__(
        self,
        sizes: Sequence[tuple[int, int] | None],
        batch_size: int,
        token_budget: int = 280,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        num_replicas: int | None = None,
        rank: int | None = None,
    ):
        self.batch_size, self.shuffle, self.drop_last, self.seed = batch_size, shuffle, drop_last, seed
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.num_replicas, self.rank = num_replicas, rank
        self.epoch = 0
        buckets: dict[tuple[int, int] | None, list[int]] = defaultdict(list)
        for i, s in enumerate(sizes):
            buckets[target_size(*s, token_budget) if s else None].append(i)
        self.buckets = list(buckets.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for b in self.buckets:
            idx = list(b)
            if self.shuffle:
                rng.shuffle(idx)
            for i in range(0, len(idx), self.batch_size):
                chunk = idx[i : i + self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        # equal number of batches per rank
        per_rank = math.ceil(len(batches) / self.num_replicas) if not self.drop_last else len(batches) // self.num_replicas
        total = per_rank * self.num_replicas
        batches = (batches + batches[: max(0, total - len(batches))])[:total]
        return batches[self.rank :: self.num_replicas]

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())


class MixtureSampler(Sampler[int]):
    """Samples a fixed-length epoch from a weighted mixture of datasets (indices into their concat).

    The number of draws per component is multinomial(``num_samples``, ``weights``); inside a component
    indices are drawn uniformly without replacement (with replacement only when a component is asked
    for more samples than it holds). Unlike ``WeightedRandomSampler`` this needs O(num_samples) memory
    and has no 2^24-category limit, so it works for 10^7+ image corpora. Deterministic given
    ``seed`` + epoch, so every DDP rank draws the same list and Lightning's
    ``DistributedSamplerWrapper`` partitions it (it forwards ``set_epoch``).

    Args:
        lengths: size of every component, in ``ConcatDataset`` order.
        weights: relative sampling probability of every component (normalised internally).
        num_samples: draws per epoch.
        seed: base seed.
    """

    def __init__(self, lengths: Sequence[int], weights: Sequence[float], num_samples: int, seed: int = 0):
        if len(lengths) != len(weights):
            raise ValueError(f"{len(weights)} weights for {len(lengths)} datasets")
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(f"weights must be non-negative with a positive sum, got {list(weights)}")
        for n, w in zip(lengths, weights):
            if w > 0 and n == 0:
                raise ValueError("a dataset with positive weight is empty")
        self.lengths = [int(n) for n in lengths]
        total = float(sum(weights))
        self.probs = [float(w) / total for w in weights]
        self.offsets = [0]
        for n in self.lengths[:-1]:
            self.offsets.append(self.offsets[-1] + n)
        self.num_samples, self.seed, self.epoch = int(num_samples), seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def component_counts(self, generator=None) -> list[int]:
        import torch

        g = generator or torch.Generator().manual_seed(self.seed + 7919 * self.epoch)
        probs = torch.tensor(self.probs, dtype=torch.float64)
        draws = torch.multinomial(probs, self.num_samples, replacement=True, generator=g)
        return torch.bincount(draws, minlength=len(self.probs)).tolist()

    def __iter__(self) -> Iterator[int]:
        import torch

        g = torch.Generator().manual_seed(self.seed + 7919 * self.epoch)
        parts = []
        for n, off, c in zip(self.lengths, self.offsets, self.component_counts(g)):
            if c == 0:
                continue
            idx = torch.randperm(n, generator=g)[:c] if c <= n else torch.randint(n, (c,), generator=g)
            parts.append(idx + off)
        out = torch.cat(parts)[torch.randperm(self.num_samples, generator=g)]
        return iter(out.tolist())

    def __len__(self) -> int:
        return self.num_samples
