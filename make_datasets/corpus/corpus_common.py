"""Shared pieces of the corpus builders: argument set, filters, decontamination and the shard writer.

Every builder (images from folders / webdataset tars / HF parquet, frames from videos) goes through
:class:`ShardedCorpusWriter.add`, so all sources get identical filtering and storage
(:mod:`onboard_vlm.data.corpus`). Per-image steps, in order:

1. decode (``decode_error``) and EXIF-transpose;
2. drop short side < ``--min_side`` (``too_small``) or long/short > ``--max_aspect`` (``bad_aspect``);
3. drop near-uniform images: std of a 32x32 grey thumbnail < ``--min_std`` (``near_uniform``);
4. benchmark decontamination against ``--blocklist`` (``make_eval_blocklist.py``): exact SHA-1 of the
   encoded bytes, dHash within ``--near_dup_bits`` (skipped for degenerate hashes with < 8 or > 56 set
   bits, which collide for unrelated low-texture images), and source ids (``--id_key``) such as the Open
   Images ids behind TextVQA (``benchmark_dup``);
5. downscale so the long side <= ``--max_side`` (re-encoded as JPEG ``--jpeg_quality``; untouched otherwise);
6. write ``<source>/<key>`` with its size, dHash and SHA-1 to the shard index.

Exact and perceptual duplicates *within and across* sources are removed afterwards, globally, by
``finalize_corpus.py`` (it only needs the shard indices).
"""

from __future__ import annotations

import io
import json
import os
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

from onboard_vlm.data.corpus import CorpusShardWriter
from onboard_vlm.data.hashing import NearDupIndex, dhash, sha1_bytes
from onboard_vlm.data.lmdb_io import VQARecord, encode_image

STAT_KEYS = ("seen", "written", "decode_error", "too_small", "bad_aspect", "near_uniform", "benchmark_dup")


def add_common_args(p) -> None:
    p.add_argument("--output_path", required=True, help="source directory, e.g. $CORPUS_ROOT/cc12m")
    p.add_argument("--name", required=True, help="source name stored in every record (cc12m, sa1b, ego4d, ...)")
    p.add_argument("--shard_size", type=int, default=50_000)
    p.add_argument("--worker_id", type=int, default=0, help="this worker's index (parallel builds)")
    p.add_argument("--num_workers", type=int, default=1, help="total parallel workers; each processes a 1/N slice")
    p.add_argument("--min_side", type=int, default=224, help="drop images whose short side is smaller")
    p.add_argument("--max_aspect", type=float, default=4.0, help="drop images with long/short side ratio above this")
    p.add_argument("--min_std", type=float, default=3.0, help="drop near-uniform images (grey 32x32 thumbnail std below this)")
    p.add_argument("--max_side", type=int, default=1536,
                   help="downscale so the long side <= this. 1536 covers the 560-token budget of Gemma 4 at any "
                        "aspect and the 1120-token budget up to ~4:3; larger images are downsampled by the processor anyway")
    p.add_argument("--jpeg_quality", type=int, default=93)
    p.add_argument("--blocklist", default=None, help="JSON from make_eval_blocklist.py (benchmark decontamination)")
    p.add_argument("--near_dup_bits", type=int, default=4, help="dHash Hamming radius treated as a benchmark duplicate")
    p.add_argument("--id_key", default=None, help="blocklist id namespace matched against each item's source id (e.g. openimages)")
    p.add_argument("--limit", type=int, default=None, help="stop after writing this many images (per worker)")


def _degenerate(h: int) -> bool:
    c = bin(h).count("1")
    return c < 8 or c > 56


class ShardedCorpusWriter:
    """Filters and writes images into ``shard_<worker>_<k>.lmdb`` (see module docstring)."""

    def __init__(self, args):
        self.args = args
        self.k = 0
        self.writer: CorpusShardWriter | None = None
        self.stats = {k: 0 for k in STAT_KEYS}
        self.block_sha: set[str] = set()
        self.block_ids: set[str] = set()
        self.block_idx = NearDupIndex(args.near_dup_bits)
        if args.blocklist:
            with open(args.blocklist) as f:
                bl = json.load(f)
            self.block_sha = set(bl.get("sha1", []))
            for h in bl.get("dhash", []):
                h = int(h, 16)
                if not _degenerate(h):
                    self.block_idx.add(h)
            if args.id_key:
                self.block_ids = set(bl.get("source_ids", {}).get(args.id_key, []))
                print(f"[{args.name}] blocking {len(self.block_ids)} '{args.id_key}' ids", flush=True)
        os.makedirs(args.output_path, exist_ok=True)

    def _rotate(self) -> None:
        if self.writer is not None:
            self.writer.close()
        shard = f"shard_{self.args.worker_id:03d}_{self.k:05d}"
        self.writer = CorpusShardWriter(self.args.output_path, shard, self.args.name)
        self.k += 1

    def add(self, key: str, data: bytes | Image.Image, meta: dict | None = None, source_id: str | None = None) -> bool:
        """Filter + write one image. ``source_id`` is matched against the blocklist's ``--id_key`` ids."""
        a = self.args
        self.stats["seen"] += 1
        if source_id is not None and source_id in self.block_ids:
            self.stats["benchmark_dup"] += 1
            return False
        try:
            raw = bytes(data) if isinstance(data, (bytes, bytearray)) else None
            img = Image.open(io.BytesIO(raw)) if raw is not None else data
            img = ImageOps.exif_transpose(img)
            img.load()
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
        if float(np.asarray(img.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32).std()) < a.min_std:
            self.stats["near_uniform"] += 1
            return False
        if raw is None:  # PIL input (video frames): encode once, fingerprint the bytes that are stored
            raw, _ = encode_image(img.convert("RGB"), None, a.jpeg_quality)
        sha = sha1_bytes(raw)
        h64 = dhash(img)
        if sha in self.block_sha or (not _degenerate(h64) and self.block_idx.contains(h64)):
            self.stats["benchmark_dup"] += 1
            return False
        enc, hw = encode_image(raw, a.max_side, a.jpeg_quality)
        if self.writer is None or len(self.writer) >= a.shard_size:
            self._rotate()
        rec = VQARecord(sample_id=f"{a.name}/{key}", dataset=a.name, split="train", image=enc, meta=dict(meta or {}))
        self.writer.add(rec, hw, h64, sha)
        self.stats["written"] += 1
        return True

    def done(self) -> bool:
        return self.args.limit is not None and self.stats["written"] >= self.args.limit

    def close(self) -> dict:
        if self.writer is not None:
            self.writer.close()
        with open(os.path.join(self.args.output_path, f"stats_{self.args.worker_id:03d}.json"), "w") as f:
            json.dump(self.stats, f, indent=1)
        print(f"[{self.args.name}] worker {self.args.worker_id}: {self.stats}", flush=True)
        return self.stats


def take_slice(items: Iterable, worker_id: int, num_workers: int):
    for i, x in enumerate(items):
        if i % num_workers == worker_id:
            yield x
