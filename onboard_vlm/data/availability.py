"""Which configured datasets are on disk, for runs on a partially built corpus (``allow_missing_sources``).

Paper runs use every source and fail on a missing one. Preliminary runs (e.g. while a licence is pending)
set ``allow_missing_sources=true``: entries whose data is absent are left out, loudly, and the mixture is
renormalised over the rest. The run logs list exactly what was left out.
"""

from __future__ import annotations

import glob
import inspect
import os
from typing import Any, Mapping

from onboard_vlm.data.corpus import list_shards


def _corpus_roots(root: str | list[str]) -> tuple[list[str], list[str]]:
    """(directories with shards, root patterns with none)."""
    present, absent = [], []
    for r in [root] if isinstance(root, str) else list(root):
        matches = sorted(p for p in glob.glob(r) if os.path.isdir(p)) if glob.has_magic(r) else [r]
        ok = [m for m in matches if os.path.isdir(m) and list_shards(m)]
        present.extend(ok)
        if not ok:
            absent.append(r)
    return present, absent


def available(cfg: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """``(cfg restricted to what is on disk or None if nothing is, note on what was left out or None)``.

    Knows ``image_corpus`` (roots with shards), ``lmdb_vqa`` (``<path>/<split>.lmdb``) and ``target_train``
    (benchmarks with a train split); other datasets are assumed present.
    """
    c = dict(cfg)
    name = c.get("name")
    if name == "image_corpus":
        present, absent = _corpus_roots(c["root"])
        if not present:
            return None, f"no corpus shards under {c['root']}"
        if absent:
            c["root"] = present
            return c, f"left out roots without shards: {absent}"
        return c, None
    if name == "lmdb_vqa":
        path = os.path.join(c["path"], f"{c.get('split', 'test')}.lmdb")
        return (c, None) if os.path.exists(path) else (None, f"{path} missing")
    if name == "target_train":
        from onboard_vlm.data.datasets import TargetTrainDataset

        wanted = list(c.get("datasets") or inspect.signature(TargetTrainDataset).parameters["datasets"].default)
        present = [d for d in wanted if os.path.exists(os.path.join(c["data_root"], d, "train.lmdb"))]
        if not present:
            return None, f"no benchmark train split among {wanted} under {c['data_root']}"
        if len(present) < len(wanted):
            c["datasets"] = present
            return c, f"left out benchmarks without a train split: {sorted(set(wanted) - set(present))}"
        return c, None
    return c, None
