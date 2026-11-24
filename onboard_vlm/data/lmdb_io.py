"""LMDB storage format shared by the ``make_datasets/lmdb`` builders and the loaders.

Directory layout (mirrors TimeFM/cerebro_v2)::

    <root>/<dataset>/
        train.lmdb   val.lmdb   test.lmdb ...
        train.meta.json ...      # keys, sizes, categories, counts (fast indexing without a scan)

Every LMDB value is a pickled :class:`VQARecord` dict. Images / frames are stored as their
original encoded bytes (JPEG/PNG), optionally downscaled at build time so the longest side is
at most ``max_side`` (2048 px by default, more than Gemma's largest 1120-token budget needs).
"""

from __future__ import annotations

import io
import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

import lmdb
import torch

MAP_SIZE = 1 << 40  # 1 TiB virtual; LMDB only uses what it writes
COMMIT_INTERVAL = 500


@dataclass
class VQARecord:
    """One benchmark sample.

    Attributes:
        sample_id: unique key (also the LMDB key).
        dataset / split: provenance.
        media_type: ``"image"`` or ``"video"``.
        image: encoded image bytes (``media_type == "image"``).
        frames: encoded frame bytes (``media_type == "video"``), temporally ordered.
        frame_timestamps: seconds from video start for each frame.
        question: question text (``""`` for caption-free image corpora).
        answers: reference answers (10 for VQA-style, 4 for EgoBlind, 1 for EgoTextVQA ...).
        choices / answer_index: multiple-choice options and gold index (``-1`` if hidden).
        answerable: VizWiz answerability label (1/0) or ``None``.
        answer_type: e.g. VizWiz ``yes/no | number | other | unanswerable``.
        category: question type / task used for per-slice metrics.
        meta: dataset-specific extras (e.g. CV-Bench ``source``, WearVQA quality issues).
    """

    sample_id: str
    dataset: str
    split: str
    media_type: str = "image"
    image: bytes | None = None
    frames: list[bytes] | None = None
    frame_timestamps: list[float] | None = None
    question: str = ""
    answers: list[str] = field(default_factory=list)
    choices: list[str] | None = None
    answer_index: int | None = None
    answerable: int | None = None
    answer_type: str | None = None
    category: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def dumps(self) -> bytes:
        return pickle.dumps(asdict(self), protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def loads(cls, b: bytes) -> "VQARecord":
        return cls(**pickle.loads(b))


def encode_image(src: str | bytes | Any, max_side: int | None = 2048, quality: int = 95) -> tuple[bytes, tuple[int, int]]:
    """Return (encoded bytes, (h, w) after EXIF transpose). Re-encodes only when downscaling."""
    from PIL import Image, ImageOps

    if isinstance(src, (bytes, bytearray)):
        raw = bytes(src)
        img = Image.open(io.BytesIO(raw))
    elif isinstance(src, str):
        with open(src, "rb") as f:
            raw = f.read()
        img = Image.open(io.BytesIO(raw))
    else:  # PIL image
        raw = None
        img = src
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    if raw is not None and (max_side is None or max(w, h) <= max_side) and img.format in ("JPEG", "PNG"):
        return raw, (h, w)
    if max_side is not None and max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        w, h = img.size
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), (h, w)


def decode_image(b: bytes) -> torch.Tensor:
    """Encoded bytes -> ``UInt8Tensor[3, H, W]`` RGB (EXIF orientation applied, like HF ``load_image``)."""
    from torchvision.io import ImageReadMode
    from torchvision.io import decode_image as tv_decode

    data = torch.frombuffer(bytearray(b), dtype=torch.uint8)
    try:
        return tv_decode(data, mode=ImageReadMode.RGB, apply_exif_orientation=True)
    except RuntimeError:  # formats torchvision cannot decode (e.g. some PNG palettes / WEBP builds)
        import numpy as np
        from PIL import Image, ImageOps

        img = ImageOps.exif_transpose(Image.open(io.BytesIO(b))).convert("RGB")
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).contiguous()


class LMDBWriter:
    """Writes records + meta json for one split. Use as a context manager."""

    def __init__(self, root: str, split: str, dataset: str, map_size: int = MAP_SIZE):
        os.makedirs(root, exist_ok=True)
        self.path = os.path.join(root, f"{split}.lmdb")
        self.meta_path = os.path.join(root, f"{split}.meta.json")
        self.env = lmdb.open(self.path, map_size=map_size, subdir=True, lock=False)  # one writer process per LMDB by construction; no flock needed (Lustre-safe)
        self.txn = self.env.begin(write=True)
        self.dataset, self.split = dataset, split
        self.keys: list[str] = []
        self.sizes: list[list[int]] = []
        self.num_frames: list[int] = []
        self.categories: list[str | None] = []
        self._pending = 0

    def add(self, rec: VQARecord, size_hw: tuple[int, int]) -> None:
        key = rec.sample_id.encode("utf-8")
        if not self.txn.put(key, rec.dumps(), overwrite=False):
            raise KeyError(f"duplicate sample_id {rec.sample_id}")
        self.keys.append(rec.sample_id)
        self.sizes.append(list(size_hw))
        self.num_frames.append(len(rec.frames) if rec.frames else 0)
        self.categories.append(rec.category)
        self._pending += 1
        if self._pending >= COMMIT_INTERVAL:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self._pending = 0

    def close(self) -> None:
        self.txn.commit()
        self.env.sync()
        self.env.close()
        cats: dict[str, int] = {}
        for c in self.categories:
            cats[str(c)] = cats.get(str(c), 0) + 1
        meta = {
            "dataset": self.dataset,
            "split": self.split,
            "num_samples": len(self.keys),
            "keys": self.keys,
            "sizes": self.sizes,
            "num_frames": self.num_frames,
            "categories": self.categories,
            "category_counts": cats,
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)
        print(f"[{self.dataset}/{self.split}] wrote {len(self.keys)} records -> {self.path}")

    def __enter__(self) -> "LMDBWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        if exc[0] is None:
            self.close()
        else:
            self.txn.abort()
            self.env.close()


_ENVS: dict[tuple[int, str], lmdb.Environment] = {}


def open_readonly(path: str) -> lmdb.Environment:
    """Process-wide cache of read-only environments. LMDB forbids opening one environment twice in
    a process (e.g. a corpus shard read by both the train and the held-out dataset); the pid in the
    key gives every forked DataLoader worker its own handles."""
    key = (os.getpid(), os.path.abspath(path))
    env = _ENVS.get(key)
    if env is None:
        env = lmdb.open(path, readonly=True, lock=False, readahead=False, max_readers=512, subdir=True)
        _ENVS[key] = env
    return env


class LMDBReader:
    """Lazily-opened (fork/worker-safe) read-only LMDB handle."""

    def __init__(self, path: str):
        self.path = path
        self._env: lmdb.Environment | None = None
        self._pid = os.getpid()

    @property
    def env(self) -> lmdb.Environment:
        if self._env is None or self._pid != os.getpid():
            self._env, self._pid = open_readonly(self.path), os.getpid()
        return self._env

    def get(self, key: str) -> VQARecord:
        with self.env.begin(write=False) as txn:
            b = txn.get(key.encode("utf-8"))
        if b is None:
            raise KeyError(key)
        return VQARecord.loads(b)

    def iter_keys(self) -> Iterator[str]:
        with self.env.begin(write=False) as txn:
            for k in txn.cursor().iternext(keys=True, values=False):
                yield k.decode("utf-8")

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path, "_env": None, "_pid": -1}


def load_meta(root: str, split: str) -> dict[str, Any]:
    p = os.path.join(root, f"{split}.meta.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    reader = LMDBReader(os.path.join(root, f"{split}.lmdb"))  # slow fallback: scan keys
    keys = list(reader.iter_keys())
    return {"keys": keys, "sizes": None, "num_frames": None, "categories": None}
