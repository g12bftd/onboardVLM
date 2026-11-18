"""Offline teacher-feature cache (safetensors shards, memory-mapped reads).

Layout of ``<root>/``::

    cache_meta.json               teacher id, dtype, D, layers, budgets, num_views
    index.jsonl                   one line per (sample, view): key, shard, row, transform params
    shard_00000.safetensors       tokens [N_s, D], offsets [n_s + 1], layer_<i> [N_s, D]

Each cached entry stores the exact :class:`TransformParams` that produced it, so the student
view is re-created deterministically by the collator. Several augmented views per image can
be cached (``num_views``); the cached dataset picks one view per epoch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from onboard_vlm.interfaces import TeacherVisionOutput, TransformParams


@dataclass
class CacheEntry:
    key: str
    shard: int
    row: int
    transform: TransformParams


class TeacherFeatureCacheWriter:
    """Accumulates per-image teacher features and flushes fixed-size shards."""

    def __init__(self, root: str, meta: dict[str, Any], shard_size: int = 2048, dtype: torch.dtype = torch.bfloat16, rank: int = 0):
        self.root = root
        self.shard_size = shard_size
        self.dtype = dtype
        self.rank = rank
        os.makedirs(root, exist_ok=True)
        if rank == 0:
            with open(os.path.join(root, "cache_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        self._buf: list[tuple[str, TransformParams, torch.Tensor, dict[int, torch.Tensor]]] = []
        self._shard = 0
        self._index = open(os.path.join(root, f"index.rank{rank}.jsonl"), "a")

    def add_batch(self, keys: Iterable[str], transforms: Iterable[TransformParams], out: TeacherVisionOutput) -> None:
        toks = out.split()
        layers = {i: out.split(i) for i in out.intermediate_tokens}
        for j, (k, t) in enumerate(zip(keys, transforms)):
            self._buf.append((k, t, toks[j].detach().to("cpu", self.dtype), {i: v[j].detach().to("cpu", self.dtype) for i, v in layers.items()}))
            if len(self._buf) >= self.shard_size:
                self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        name = f"shard_r{self.rank}_{self._shard:05d}.safetensors"
        lengths = [b[2].shape[0] for b in self._buf]
        offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
        tensors = {"tokens": torch.cat([b[2] for b in self._buf]), "offsets": offsets}
        for i in self._buf[0][3]:
            tensors[f"layer_{i}"] = torch.cat([b[3][i] for b in self._buf])
        save_file(tensors, os.path.join(self.root, name))
        for row, (k, t, _, _) in enumerate(self._buf):
            self._index.write(json.dumps({"key": k, "shard": name, "row": row, "transform": t.to_dict()}) + "\n")
        self._index.flush()
        self._buf.clear()
        self._shard += 1

    def close(self) -> None:
        self.flush()
        self._index.close()


class TeacherFeatureCache:
    """Read-only, lazily memory-mapped access to cached teacher features."""

    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "cache_meta.json")) as f:
            self.meta = json.load(f)
        self.entries: dict[str, list[tuple[str, int, TransformParams]]] = {}
        for fn in sorted(os.listdir(root)):
            if fn.startswith("index") and fn.endswith(".jsonl"):
                with open(os.path.join(root, fn)) as f:
                    for line in f:
                        e = json.loads(line)
                        self.entries.setdefault(e["key"], []).append((e["shard"], e["row"], TransformParams.from_dict(e["transform"])))
        self._handles: dict[str, Any] = {}
        self._offsets: dict[str, torch.Tensor] = {}

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def num_views(self, key: str) -> int:
        return len(self.entries[key])

    def _open(self, shard: str):
        if shard not in self._handles:  # opened lazily per DataLoader worker
            h = safe_open(os.path.join(self.root, shard), framework="pt")
            self._handles[shard] = h
            self._offsets[shard] = h.get_tensor("offsets")
        return self._handles[shard]

    def get(self, key: str, view: int = 0, layers: Iterable[int] = ()) -> tuple[torch.Tensor, dict[int, torch.Tensor], TransformParams]:
        """Returns ``tokens [n, D]``, ``{layer: [n, D]}`` and the transform of view ``view``."""
        shard, row, transform = self.entries[key][view]
        h = self._open(shard)
        off = self._offsets[shard]
        s, e = int(off[row]), int(off[row + 1])
        tokens = h.get_slice("tokens")[s:e]
        inter = {i: h.get_slice(f"layer_{i}")[s:e] for i in layers}
        return tokens, inter, transform

    def __getstate__(self) -> dict[str, Any]:  # file handles are not picklable across workers
        d = dict(self.__dict__)
        d["_handles"], d["_offsets"] = {}, {}
        return d
