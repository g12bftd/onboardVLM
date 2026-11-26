"""Size-bucketing batch sampler.

With deterministic (eval / cached) geometry, images whose Gemma target size coincides can be
batched together so that the dense student batch has *no* spatial padding. This makes CNN
students (which are not batch invariant) see exactly what they would see one image at a time.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterator, Sequence

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

    def __init__(self, lengths: Sequence[int], weights: Sequence[float], num_samples: int, seed: int = 0, shard: bool = False,
                 num_replicas: int | None = None, rank: int | None = None):
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
        # shard=True: this sampler splits the epoch across DDP ranks itself (trainer use_distributed_sampler=false)
        self.num_replicas, self.rank = _dist_info(num_replicas, rank) if shard else (1, 0)

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
        per_rank = self.num_samples // self.num_replicas
        return iter(out[self.rank * per_rank : (self.rank + 1) * per_rank].tolist())

    def __len__(self) -> int:
        return self.num_samples // self.num_replicas


# ----------------------------------------------------------------------------------------------
# Shape-exact batching: every batch holds images of ONE processor output shape (no padding at all)
# ----------------------------------------------------------------------------------------------
def _dist_info(num_replicas: int | None, rank: int | None) -> tuple[int, int]:
    if num_replicas is None:
        num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if rank is None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    return num_replicas, rank


def batch_size_for(shape: tuple[int, int], tokens_per_batch: int) -> int:
    """Token-balanced batch size: ~``tokens_per_batch`` soft tokens per batch whatever the budget."""
    from onboard_vlm.interfaces import TOKEN_STRIDE

    return max(1, tokens_per_batch // ((shape[0] // TOKEN_STRIDE) * (shape[1] // TOKEN_STRIDE)))


class BucketedBatchSampler(Sampler[list[tuple]]):
    """Training batches for feature / embedding distillation.

    Every batch has one token budget, one output shape (a processor output shape, see
    :func:`onboard_vlm.geometry.processor_shapes`) and ~``tokens_per_batch`` soft tokens:

    * budget ~ ``budget_weights`` (with token-balanced batches, the weights are the share of
      *compute* per budget; uniform weights give every budget the same gradient signal);
    * aspect ratio W/H: with probability ``camera_prob`` one of ``camera_aspects`` (the deployment
      cameras), otherwise log-uniform in ``[1/max_aspect, max_aspect]``;
    * images: drawn from the dataset mixture (``weights``), uniformly without replacement within a
      component; each is random-resized-cropped to the batch's aspect by the collator.

    Students therefore never see padding during training (identical to batch-1 deployment) and every
    budget / aspect of the input domain is trained. The plan is a function of ``seed`` + epoch only:
    all DDP ranks build the same global plan and take every ``num_replicas``-th batch.

    Yields lists of ``("crop", index, budget, H, W)``.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        weights: Sequence[float] | None,
        budgets: Sequence[int],
        tokens_per_batch: int,
        samples_per_epoch: int,
        budget_weights: Sequence[float] | None = None,
        max_aspect: float = 4.0,
        camera_aspects: Sequence[float] = (4 / 3, 3 / 4, 16 / 9, 9 / 16),
        camera_prob: float = 0.5,
        seed: int = 0,
        num_replicas: int | None = None,
        rank: int | None = None,
    ):
        import numpy as np

        self.lengths = [int(n) for n in lengths]
        w = np.asarray(weights if weights is not None else self.lengths, dtype=np.float64)
        if len(w) != len(self.lengths) or (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"bad mixture weights {list(w)} for {len(self.lengths)} datasets")
        if any(n == 0 and p > 0 for n, p in zip(self.lengths, w)):
            raise ValueError("a dataset with positive weight is empty")
        self.probs = w / w.sum()
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(np.int64)
        self.budgets = [int(b) for b in budgets]
        bw = np.asarray(budget_weights if budget_weights is not None else [1.0] * len(self.budgets), dtype=np.float64)
        if len(bw) != len(self.budgets) or (bw < 0).any() or bw.sum() <= 0:
            raise ValueError(f"bad budget_weights {list(bw)} for budgets {self.budgets}")
        self.budget_probs = bw / bw.sum()
        if not all(1 / max_aspect <= a <= max_aspect for a in camera_aspects):
            raise ValueError(f"camera_aspects {list(camera_aspects)} outside the domain [1/{max_aspect}, {max_aspect}]")
        self.tokens_per_batch, self.samples_per_epoch = int(tokens_per_batch), int(samples_per_epoch)
        self.max_aspect, self.camera_aspects, self.camera_prob = float(max_aspect), [float(a) for a in camera_aspects], float(camera_prob)
        self.seed, self.epoch = seed, 0
        self.num_replicas, self.rank = _dist_info(num_replicas, rank)
        self._plans: dict[int, list[tuple[int, int, int, int]]] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rng(self, salt: int):
        import numpy as np

        return np.random.default_rng([self.seed, self.epoch, salt])

    def _aspect(self, rng) -> float:
        import math

        if rng.random() < self.camera_prob:
            return self.camera_aspects[rng.integers(len(self.camera_aspects))]
        return math.exp(rng.uniform(-math.log(self.max_aspect), math.log(self.max_aspect)))

    def expected_batch_size(self) -> dict[int, float]:
        """Mean images per batch for every budget under the aspect distribution (fixed-seed estimate)."""
        import numpy as np

        from onboard_vlm.geometry import shape_for_aspect

        rng = np.random.default_rng([self.seed, 2**31 - 1])
        aspects = [self._aspect(rng) for _ in range(2000)]
        return {b: float(np.mean([batch_size_for(shape_for_aspect(a, b), self.tokens_per_batch) for a in aspects])) for b in self.budgets}

    def num_batches(self) -> int:
        """Global batches per epoch: enough for ~``samples_per_epoch`` images at the budget mix, rounded
        down to a multiple of the number of ranks."""
        import math

        ebs = self.expected_batch_size()
        mean_bs = sum(p * ebs[b] for b, p in zip(self.budgets, self.budget_probs))
        n = max(self.num_replicas, math.ceil(self.samples_per_epoch / mean_bs))
        return n // self.num_replicas * self.num_replicas

    def plan(self) -> list[tuple[int, int, int, int]]:
        """Global ``(budget, H, W, batch_size)`` list of this epoch (identical on all ranks).

        Budgets are allocated *exactly* in proportion to ``budget_weights`` (largest-remainder
        rounding, then shuffled): the share of batches, and hence of compute, per budget is fixed
        rather than sampled, and does not depend on when the epoch ends.
        """
        import numpy as np

        from onboard_vlm.geometry import shape_for_aspect

        if self.epoch in self._plans:
            return self._plans[self.epoch]
        rng = self._rng(0)
        n = self.num_batches()
        quota = self.budget_probs * n
        counts = np.floor(quota).astype(int)
        for k in np.argsort(-(quota - counts))[: n - counts.sum()]:
            counts[k] += 1
        budgets = np.repeat(self.budgets, counts)[rng.permutation(n)]
        out = []
        for b in budgets:
            h, w = shape_for_aspect(self._aspect(rng), int(b))
            out.append((int(b), h, w, batch_size_for((h, w), self.tokens_per_batch)))
        self._plans = {self.epoch: out}
        return out

    def __iter__(self) -> Iterator[list[tuple]]:
        import numpy as np

        plan = self.plan()
        rng = self._rng(1)
        perms: dict[int, np.ndarray] = {}
        cursor: dict[int, int] = {}

        def take(c: int, n: int) -> np.ndarray:
            got = []
            while n > 0:
                if c not in perms or cursor[c] >= len(perms[c]):
                    perms[c], cursor[c] = rng.permutation(self.lengths[c]), 0
                k = min(n, len(perms[c]) - cursor[c])
                got.append(perms[c][cursor[c] : cursor[c] + k])
                cursor[c] += k
                n -= k
            return np.concatenate(got) + self.offsets[c]

        for j, (b, h, w, bs) in enumerate(plan):
            comps = rng.choice(len(self.lengths), size=bs, p=self.probs)
            idx = np.empty(bs, dtype=np.int64)
            for c in np.unique(comps):
                sel = comps == c
                idx[sel] = take(int(c), int(sel.sum()))
            if j % self.num_replicas == self.rank:  # every rank draws the whole plan, keeps its share
                yield [("crop", int(i), b, h, w) for i in idx]

    def __len__(self) -> int:
        return len(self.plan()) // self.num_replicas


class ExactShapeBatchSampler(Sampler[list[tuple]]):
    """Batches of full images whose processor output shape is identical (evaluation, and training on
    full images, e.g. logit distillation where the prompt refers to the whole image).

    Each image gets one budget per epoch (``budgets`` / ``budget_weights``; evaluation uses a single
    budget), is grouped with images of the same output shape, and batched with ~``tokens_per_batch``
    soft tokens. Images outside ``domain`` are excluded and counted in ``excluded``.

    Yields lists of ``("full", index, budget, H, W)``.
    """

    def __init__(
        self,
        sizes: Sequence[tuple[int, int]],
        budgets: Sequence[int],
        tokens_per_batch: int,
        budget_weights: Sequence[float] | None = None,
        shuffle: bool = False,
        domain: Any = None,
        seed: int = 0,
        num_replicas: int | None = None,
        rank: int | None = None,
        drop_uneven: bool | None = None,
        lengths: Sequence[int] | None = None,
        weights: Sequence[float] | None = None,
        num_samples: int | None = None,
    ):
        """``lengths`` / ``weights`` / ``num_samples`` (training on full images from a weighted mixture): each epoch
        draws ``num_samples`` images, every one from component ``c`` with probability ``weights[c]`` (uniformly without
        replacement within a component, as :class:`BucketedBatchSampler`), then batches them by exact shape.
        Without them, every image is used once per epoch."""
        import numpy as np

        from onboard_vlm.geometry import target_size

        if (weights is None) != (num_samples is None) or (weights is not None and lengths is None):
            raise ValueError("a mixture needs lengths, weights and num_samples together")
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if len(w) != len(lengths) or (w < 0).any() or w.sum() <= 0 or sum(lengths) != len(sizes):
                raise ValueError(f"bad mixture: weights {list(w)}, lengths {list(lengths)}, {len(sizes)} sizes")
            self.mix = (np.asarray(lengths, dtype=np.int64), w / w.sum(), int(num_samples))
        else:
            self.mix = None
        self.budgets = [int(b) for b in budgets]
        bw = np.asarray(budget_weights if budget_weights is not None else [1.0] * len(self.budgets), dtype=np.float64)
        self.budget_probs = bw / bw.sum()
        self.tokens_per_batch, self.shuffle, self.seed, self.epoch = int(tokens_per_batch), shuffle, seed, 0
        self.num_replicas, self.rank = _dist_info(num_replicas, rank)
        self.drop_uneven = shuffle if drop_uneven is None else drop_uneven  # training needs equal #batches per rank
        self.shapes: dict[int, list] = {}
        self.excluded: dict[int, int] = {}
        for b in self.budgets:
            shp = [target_size(int(h), int(w), b) for h, w in sizes]
            if domain is not None:
                ok = [domain.contains(*s) for s in shp]
                self.excluded[b] = len(ok) - sum(ok)
                shp = [s if k else None for s, k in zip(shp, ok)]
            self.shapes[b] = shp
        self.n = len(sizes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _draw(self, rng) -> "np.ndarray":
        """The images of this epoch: all of them, or ``num_samples`` drawn from the mixture."""
        import numpy as np

        if self.mix is None:
            return np.arange(self.n)
        lengths, probs, n = self.mix
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        comps = rng.choice(len(lengths), size=n, p=probs)
        out = []
        for c in np.unique(comps):
            k = int((comps == c).sum())
            reps = -(-k // int(lengths[c]))  # whole passes when a component is drawn more often than its size
            perm = np.concatenate([rng.permutation(int(lengths[c])) for _ in range(reps)])[:k]
            out.append(perm + offsets[c])
        return np.concatenate(out)

    def _batches(self) -> list[list[tuple]]:
        import numpy as np

        rng = np.random.default_rng([self.seed, self.epoch])
        items = self._draw(rng)
        assign = (rng.choice(len(self.budgets), size=len(items), p=self.budget_probs) if len(self.budgets) > 1
                  else np.zeros(len(items), dtype=np.int64))
        groups: dict[tuple[int, int, int], list[int]] = {}
        for j, i in enumerate(items):
            i = int(i)
            b = self.budgets[assign[j]]
            s = self.shapes[b][i]
            if s is not None:
                groups.setdefault((b, *s), []).append(i)
        batches = []
        for (b, h, w), idx in sorted(groups.items()):
            if self.shuffle:
                idx = list(rng.permutation(idx))
            bs = batch_size_for((h, w), self.tokens_per_batch)
            batches += [[("full", int(i), b, h, w) for i in idx[k : k + bs]] for k in range(0, len(idx), bs)]
        if self.shuffle:
            batches = [batches[k] for k in rng.permutation(len(batches))]
        if self.drop_uneven:
            batches = batches[: len(batches) // self.num_replicas * self.num_replicas]
        return batches[self.rank :: self.num_replicas]

    def __iter__(self) -> Iterator[list[tuple]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())
