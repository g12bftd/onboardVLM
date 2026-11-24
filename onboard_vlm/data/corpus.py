"""Storage format of the general distillation corpus (``make_datasets/corpus``) and its dataset.

Every source (CC12M, ImageNet-21k, Open Images, SA-1B, text-rich sets, Ego4D frames, ...) is
converted to the same layout, so all of them are read by one dataset class::

    <corpus_root>/<source>/
        shard_<worker>_<k>.lmdb         records under positional keys b"00000000", b"00000001", ...
        shard_<worker>_<k>.meta.json    {"format": "corpus-v1", "dataset", "num_samples", "stats"}
        shard_<worker>_<k>.index.npz    sizes int32 [n, 2] (h, w) | dhash uint64 [n] | sha1 S20 [n]
        stats_<worker>.json             filter counts of the builder worker
        keep.npz / holdout.npz          written by finalize_corpus.py: train / held-out bool mask per shard

Values are pickled :class:`~onboard_vlm.data.lmdb_io.VQARecord` dicts with the encoded image bytes
and provenance in ``sample_id`` (``<source>/<original key>``) and ``meta``. Positional keys plus the
numpy index keep start-up and memory flat at 10^7+ images (no Python string per sample).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Sequence

import lmdb
import numpy as np
from torch.utils.data import Dataset

from onboard_vlm.data.lmdb_io import MAP_SIZE, VQARecord, decode_image, open_readonly
from onboard_vlm.registry import register_dataset

FORMAT = "corpus-v1"
SPLIT_FILES = {"train": "keep.npz", "holdout": "holdout.npz"}
COMMIT_INTERVAL = 1000


def corpus_key(i: int) -> bytes:
    return f"{i:08d}".encode("ascii")


def list_shards(root: str) -> list[str]:
    """Shard names (``shard_000_00000``, ...) under ``root``, sorted."""
    return sorted(os.path.basename(p)[: -len(".index.npz")] for p in glob.glob(os.path.join(root, "shard_*.index.npz")))


class CorpusShardWriter:
    """Writes one corpus shard. Use as a context manager or call :meth:`close`."""

    def __init__(self, root: str, shard: str, dataset: str, map_size: int = MAP_SIZE):
        os.makedirs(root, exist_ok=True)
        self.root, self.shard, self.dataset = root, shard, dataset
        self.env = lmdb.open(os.path.join(root, f"{shard}.lmdb"), map_size=map_size, subdir=True, lock=True)
        self.txn = self.env.begin(write=True)
        self.sizes: list[tuple[int, int]] = []
        self.dhash: list[int] = []
        self.sha1: list[bytes] = []
        self._pending = 0

    def __len__(self) -> int:
        return len(self.sizes)

    def add(self, rec: VQARecord, size_hw: tuple[int, int], dhash: int, sha1_hex: str) -> None:
        self.txn.put(corpus_key(len(self.sizes)), rec.dumps())
        self.sizes.append((int(size_hw[0]), int(size_hw[1])))
        self.dhash.append(int(dhash))
        self.sha1.append(bytes.fromhex(sha1_hex))
        self._pending += 1
        if self._pending >= COMMIT_INTERVAL:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self._pending = 0

    def close(self) -> None:
        self.txn.commit()
        self.env.sync()
        self.env.close()
        np.savez(
            os.path.join(self.root, f"{self.shard}.index.npz"),
            sizes=np.asarray(self.sizes, dtype=np.int32).reshape(-1, 2),
            dhash=np.asarray(self.dhash, dtype=np.uint64),
            sha1=np.asarray(self.sha1, dtype="S20"),
        )
        with open(os.path.join(self.root, f"{self.shard}.meta.json"), "w") as f:
            json.dump({"format": FORMAT, "dataset": self.dataset, "num_samples": len(self.sizes)}, f)

    def __enter__(self) -> "CorpusShardWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        if exc[0] is None:
            self.close()
        else:
            self.txn.abort()
            self.env.close()


def load_index(root: str, shard: str) -> dict[str, np.ndarray]:
    with np.load(os.path.join(root, f"{shard}.index.npz")) as z:
        return {k: z[k] for k in z.files}


def _expand_roots(root: str | Sequence[str]) -> list[str]:
    roots = [root] if isinstance(root, str) else list(root)
    out: list[str] = []
    for r in roots:
        matches = sorted(p for p in glob.glob(r) if os.path.isdir(p)) if glob.has_magic(r) else [r]
        out.extend(matches)
    if not out:
        raise FileNotFoundError(f"no corpus directory matches {root}")
    return out


@register_dataset("image_corpus")
class ImageCorpusDataset(Dataset):
    """One or more corpus directories (see module docstring) read as a single dataset.

    Args:
        root: corpus directory, a glob (``.../text_rich/*``) or a list of either. Several roots form
            one mixture component (sampled proportionally to their size).
        split: ``"train"`` (``keep.npz`` from ``finalize_corpus.py``: de-duplicated, held-out images
            removed; every image if the corpus is not finalized and ``require_finalized`` is False) or
            ``"holdout"`` (``holdout.npz``: per-source in-distribution validation images).
        require_finalized: raise if a root was not finalized (set in the paper configs).
        max_shards: per root, use only the first N shards (quick experiments / lite corpus).
        limit: truncate (debugging).
    """

    def __init__(
        self,
        root: str | Sequence[str],
        split: str = "train",
        require_finalized: bool = False,
        max_shards: int | None = None,
        limit: int | None = None,
    ):
        if split not in SPLIT_FILES:
            raise ValueError(f"split must be one of {sorted(SPLIT_FILES)}, got {split!r}")
        self.roots, self.split = _expand_roots(root), split
        self.shards: list[tuple[str, str]] = []  # (root, shard name)
        shard_ids, local, sizes = [], [], []
        self.num_raw = 0
        for r in self.roots:
            names = list_shards(r)[:max_shards]
            if not names:
                raise FileNotFoundError(f"no shard_*.index.npz under {r} (build it with make_datasets/corpus/)")
            mask_path = os.path.join(r, SPLIT_FILES[split])
            if (require_finalized or split == "holdout") and not os.path.isfile(mask_path):
                raise FileNotFoundError(f"{mask_path} missing: run make_datasets/corpus/finalize_corpus.py first")
            keep = np.load(mask_path) if os.path.isfile(mask_path) else None
            for name in names:
                idx = load_index(r, name)
                n = len(idx["sizes"])
                self.num_raw += n
                mask = keep[name] if keep is not None and name in keep.files else np.ones(n, dtype=bool)
                pos = np.flatnonzero(mask).astype(np.int32)
                shard_ids.append(np.full(len(pos), len(self.shards), dtype=np.int32))
                local.append(pos)
                sizes.append(idx["sizes"][pos])
                self.shards.append((r, name))
        self.shard_of = np.concatenate(shard_ids) if shard_ids else np.zeros(0, np.int32)
        self.local = np.concatenate(local) if local else np.zeros(0, np.int32)
        self.sizes = np.concatenate(sizes) if sizes else np.zeros((0, 2), np.int32)
        if limit is not None:
            self.shard_of, self.local, self.sizes = self.shard_of[:limit], self.local[:limit], self.sizes[:limit]

    def __len__(self) -> int:
        return len(self.local)

    def _env(self, k: int) -> lmdb.Environment:
        r, name = self.shards[k]
        return open_readonly(os.path.join(r, f"{name}.lmdb"))

    def record(self, i: int) -> VQARecord:
        k, j = int(self.shard_of[i]), int(self.local[i])
        with self._env(k).begin(write=False) as txn:
            b = txn.get(corpus_key(j))
        if b is None:
            raise KeyError(f"{self.shards[k]}[{j}]")
        return VQARecord.loads(b)

    def __getitem__(self, i: int) -> dict[str, Any]:
        rec = self.record(i)
        return {"sample_id": rec.sample_id, "dataset": rec.dataset, "image": decode_image(rec.image)}  # type: ignore[arg-type]
