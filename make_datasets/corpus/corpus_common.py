"""Shared pieces of the distillation-corpus builders: sharded LMDB writer, filters, blocklist."""

from __future__ import annotations

import io
import json
import os
from typing import Iterable

from PIL import Image

from onboard_vlm.data.hashing import NearDupIndex, dhash, sha1_bytes
from onboard_vlm.data.lmdb_io import LMDBWriter, VQARecord, encode_image


def add_common_args(p) -> None:
    p.add_argument("--output_path", required=True, help="corpus directory (shard_*.lmdb are written here)")
    p.add_argument("--name", required=True, help="corpus name stored in every record (e.g. cc12m, sa1b, ego4d)")
    p.add_argument("--shard_size", type=int, default=50_000)
    p.add_argument("--worker_id", type=int, default=0, help="this worker's index (parallel builds)")
    p.add_argument("--num_workers", type=int, default=1, help="total parallel workers; each processes a 1/N slice")
    p.add_argument("--min_side", type=int, default=224, help="drop images whose short side is smaller")
    p.add_argument("--max_aspect", type=float, default=4.0, help="drop images with long/short side ratio above this")
    p.add_argument("--max_side", type=int, default=2048, help="downscale so the long side <= this (Gemma 1120-token frames need ~2112)")
    p.add_argument("--blocklist", default=None, help="JSON from make_eval_blocklist.py: drop benchmark (near-)duplicates")
    p.add_argument("--near_dup_bits", type=int, default=4, help="dHash Hamming radius treated as duplicate")
    p.add_argument("--dedup", action="store_true", help="also drop near-duplicates within this corpus")
    p.add_argument("--limit", type=int, default=None)


class ShardedCorpusWriter:
    """Filters, de-duplicates and writes images into ``shard_<worker>_<k>.lmdb`` files."""

    def __init__(self, args):
        self.args = args
        self.k = 0
        self.count = 0
        self.writer: LMDBWriter | None = None
        self.stats = {"written": 0, "too_small": 0, "bad_aspect": 0, "benchmark_dup": 0, "corpus_dup": 0, "decode_error": 0}
        self.block_sha: set[str] = set()
        self.block_idx = NearDupIndex(args.near_dup_bits)
        if args.blocklist:
            with open(args.blocklist) as f:
                bl = json.load(f)
            self.block_sha = set(bl.get("sha1", []))
            for h in bl.get("dhash", []):
                self.block_idx.add(int(h, 16))
        self.seen_sha: set[str] = set()
        self.seen_idx = NearDupIndex(args.near_dup_bits) if args.dedup else None
        os.makedirs(args.output_path, exist_ok=True)

    def _rotate(self) -> None:
        if self.writer is not None:
            self.writer.close()
        split = f"shard_{self.args.worker_id:03d}_{self.k:05d}"
        self.writer = LMDBWriter(self.args.output_path, split, self.args.name)
        self.k += 1
        self.count = 0

    def add(self, key: str, data: bytes | Image.Image, meta: dict | None = None) -> bool:
        a = self.args
        try:
            raw = data if isinstance(data, (bytes, bytearray)) else None
            img = Image.open(io.BytesIO(raw)) if raw is not None else data
            w, h = img.size
        except Exception:
            self.stats["decode_error"] += 1
            return False
        if min(w, h) < a.min_side:
            self.stats["too_small"] += 1
            return False
        if max(w, h) / max(1, min(w, h)) > a.max_aspect:
            self.stats["bad_aspect"] += 1
            return False
        sha = sha1_bytes(raw) if raw is not None else None
        h64 = dhash(img)
        if (sha and sha in self.block_sha) or self.block_idx.contains(h64):
            self.stats["benchmark_dup"] += 1
            return False
        if self.seen_idx is not None:
            if (sha and sha in self.seen_sha) or self.seen_idx.contains(h64):
                self.stats["corpus_dup"] += 1
                return False
            if sha:
                self.seen_sha.add(sha)
            self.seen_idx.add(h64)
        enc, hw = encode_image(raw if raw is not None else img, a.max_side)
        if self.writer is None or self.count >= a.shard_size:
            self._rotate()
        self.writer.add(VQARecord(sample_id=f"{a.name}/{key}", dataset=a.name, split="train", image=enc, meta={"dhash": f"{h64:016x}", **(meta or {})}), hw)
        self.count += 1
        self.stats["written"] += 1
        return True

    def close(self) -> dict:
        if self.writer is not None:
            self.writer.close()
        with open(os.path.join(self.args.output_path, f"stats_{self.args.worker_id:03d}.json"), "w") as f:
            json.dump(self.stats, f, indent=1)
        print(self.stats)
        return self.stats


def take_slice(items: Iterable, worker_id: int, num_workers: int):
    for i, x in enumerate(items):
        if i % num_workers == worker_id:
            yield x
