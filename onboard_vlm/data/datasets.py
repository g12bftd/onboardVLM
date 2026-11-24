"""Registered datasets. All return plain dicts; geometry happens later, once, in the collator.

Sample dict keys:
    sample_id (str), dataset (str),
    image  UInt8Tensor[3, H, W]          for image samples, or
    frames UInt8Tensor[T, 3, H, W] + frame_timestamps list[float]   for video samples,
    question, answers, choices, answer_index, answerable, answer_type, category, meta.
Cached-teacher samples additionally carry ``transform``, ``teacher_tokens [n, D]`` and
``teacher_layers {i: [n, D]}``.
"""

from __future__ import annotations

import os
import random
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from onboard_vlm.data.lmdb_io import LMDBReader, VQARecord, decode_image, load_meta
from onboard_vlm.registry import build_dataset, register_dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
QUESTION_SEP = "::q"  # builders key questions as "<image_id>::q<question_id>"


def image_key_of(sample_id: str) -> str:
    """Image-level key of a question-level sample id (used for de-duplication and the teacher cache)."""
    return sample_id.split(QUESTION_SEP)[0]


def _record_to_sample(rec: VQARecord, frame_ids: Sequence[int] | None = None) -> dict[str, Any]:
    s: dict[str, Any] = {
        "sample_id": rec.sample_id,
        "dataset": rec.dataset,
        "question": rec.question,
        "answers": list(rec.answers),
        "choices": rec.choices,
        "answer_index": rec.answer_index,
        "answerable": rec.answerable,
        "answer_type": rec.answer_type,
        "category": rec.category,
        "meta": dict(rec.meta),
    }
    if rec.media_type == "image":
        s["image"] = decode_image(rec.image)  # type: ignore[arg-type]
    else:
        ids = list(range(len(rec.frames))) if frame_ids is None else list(frame_ids)  # type: ignore[arg-type]
        s["frames"] = torch.stack([decode_image(rec.frames[i]) for i in ids])  # type: ignore[index]
        s["frame_timestamps"] = [rec.frame_timestamps[i] for i in ids] if rec.frame_timestamps else None
    return s


@register_dataset("lmdb_vqa")
class LMDBVQADataset(Dataset):
    """Reads ``<path>/<split>.lmdb`` produced by ``make_datasets/lmdb/make_*.py``.

    Args:
        path: dataset directory (``.../vizwiz``).
        split: ``train`` / ``val`` / ``test`` / ...
        mode: ``"vqa"`` (one item per question, all fields; logit distillation / eval) or ``"image"``
            (one item per visual unit for label-free feature / embedding distillation; questions sharing an
            image are de-duplicated by ``meta.image_id`` when present).
        video_frames: in ``"image"`` mode, ``"all"`` (every stored frame is an item),
            ``"random"`` (one random frame per video per access) or ``"middle"``.
        max_frames: in ``"vqa"`` mode, uniformly subsample at most this many stored frames.
        categories: keep only records whose ``category`` is in this list.
        limit: truncate (debugging).
    """

    def __init__(
        self,
        path: str,
        split: str,
        mode: str = "vqa",
        video_frames: str = "all",
        max_frames: int | None = None,
        categories: Sequence[str] | None = None,
        limit: int | None = None,
    ):
        if mode not in ("vqa", "image"):
            raise ValueError(f"mode must be 'vqa' or 'image', got {mode!r}")
        self.path, self.split, self.mode = path, split, mode
        self.video_frames, self.max_frames = video_frames, max_frames
        self.reader = LMDBReader(os.path.join(path, f"{split}.lmdb"))
        meta = load_meta(path, split)
        keys: list[str] = meta["keys"]
        cats = meta.get("categories") or [None] * len(keys)
        nfr = meta.get("num_frames") or [0] * len(keys)
        sizes = meta.get("sizes") or [None] * len(keys)
        rows = [(k, c, n, s) for k, c, n, s in zip(keys, cats, nfr, sizes) if categories is None or c in set(categories)]
        if limit is not None:
            rows = rows[:limit]
        self.index: list[tuple[str, int | None]] = []
        self.sizes: list[tuple[int, int] | None] = []
        seen_images: set[str] = set()
        for k, _, n, s in rows:
            size = tuple(s) if s else None
            if mode == "image" and n:  # video record: frames differ per question, never de-duplicate
                frames = range(n) if video_frames == "all" else [None]
                self.index.extend((k, f) for f in frames)
                self.sizes.extend([size] * len(frames))
            elif mode == "image":  # VQA datasets repeat images across questions; keep each once
                image_key = image_key_of(k)
                if image_key in seen_images:
                    continue
                seen_images.add(image_key)
                self.index.append((k, None))
                self.sizes.append(size)
            else:
                self.index.append((k, None))
                self.sizes.append(size)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        key, frame = self.index[i]
        rec = self.reader.get(key)
        if self.mode == "vqa":
            ids = None
            if rec.media_type == "video" and self.max_frames and len(rec.frames) > self.max_frames:  # type: ignore[arg-type]
                n = len(rec.frames)  # type: ignore[arg-type]
                ids = [round(j * (n - 1) / (self.max_frames - 1)) for j in range(self.max_frames)] if self.max_frames > 1 else [n - 1]
            return _record_to_sample(rec, ids)
        # image mode
        if rec.media_type == "image":
            return {"sample_id": image_key_of(rec.sample_id), "dataset": rec.dataset, "image": decode_image(rec.image)}  # type: ignore[arg-type]
        n = len(rec.frames)  # type: ignore[arg-type]
        if frame is None:
            frame = random.randrange(n) if self.video_frames == "random" else n // 2
        return {"sample_id": f"{rec.sample_id}#f{frame}", "dataset": rec.dataset, "image": decode_image(rec.frames[frame])}  # type: ignore[index]


@register_dataset("image_folder")
class ImageFolderDataset(Dataset):
    """Unlabelled images under ``root`` (e.g. COCO / Ego4D frames) for feature / embedding distillation."""

    def __init__(self, root: str, extensions: Sequence[str] = IMAGE_EXTENSIONS, limit: int | None = None):
        self.root = root
        files = []
        for d, _, fs in os.walk(root):
            files.extend(os.path.join(d, f) for f in fs if f.lower().endswith(tuple(extensions)))
        self.files = sorted(files)[:limit]
        self.sizes: list[tuple[int, int] | None] = [None] * len(self.files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict[str, Any]:
        with open(self.files[i], "rb") as f:
            img = decode_image(f.read())
        return {"sample_id": os.path.relpath(self.files[i], self.root), "dataset": "image_folder", "image": img}


@register_dataset("cached_teacher")
class CachedTeacherDataset(Dataset):
    """Wraps an image-mode dataset with features from :class:`TeacherFeatureCache`.

    Args:
        base: dataset config (``{"name": ..., ...}``) in ``image`` mode.
        cache_root: directory written by ``run_cache_teacher.py``.
        layers: teacher layers to read (must have been cached; ``None`` = every cached layer).
        random_view: pick a random cached augmentation view per access (else view 0).
    """

    def __init__(self, base: dict[str, Any], cache_root: str, layers: Sequence[int] | None = None, random_view: bool = True):
        from onboard_vlm.teachers.feature_cache import TeacherFeatureCache

        self.base = build_dataset(base)
        self.cache = TeacherFeatureCache(cache_root)
        self.layers = list(self.cache.meta.get("layers", [])) if layers is None else list(layers)
        missing = set(self.layers) - set(self.cache.meta.get("layers", []))
        if missing:
            raise KeyError(f"layers {sorted(missing)} are not in the cache at {cache_root}")
        self.random_view = random_view
        self.sizes = getattr(self.base, "sizes", None)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = self.base[i]
        key = s["sample_id"]
        view = random.randrange(self.cache.num_views(key)) if self.random_view else 0
        tokens, inter, transform = self.cache.get(key, view, self.layers)
        s.update(transform=transform, teacher_tokens=tokens, teacher_layers=inter)
        return s


@register_dataset("synthetic")
class SyntheticVQADataset(Dataset):
    """Random images (random sizes/aspects) with toy QA fields, for offline dry runs and tests.

    Args:
        num_samples: dataset length.
        min_side, max_side: source image side range in pixels.
        num_frames: >0 makes video samples with this many frames.
        seed: makes the dataset deterministic per index.
    """

    def __init__(self, num_samples: int = 64, min_side: int = 96, max_side: int = 640, num_frames: int = 0, seed: int = 0):
        self.n, self.min_side, self.max_side, self.num_frames, self.seed = num_samples, min_side, max_side, num_frames, seed
        g = torch.Generator().manual_seed(seed)
        self.sizes = [tuple(torch.randint(min_side, max_side + 1, (2,), generator=g).tolist()) for _ in range(num_samples)]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, Any]:
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        h, w = self.sizes[i]
        s: dict[str, Any] = {
            "sample_id": f"synthetic_{i}",
            "dataset": "synthetic",
            "question": "what is this",
            "answers": ["yes"] * 10,
            "choices": None,
            "answer_index": None,
            "answerable": 1,
            "answer_type": "yes/no",
            "category": ["a", "b"][i % 2],
            "meta": {},
        }
        if self.num_frames:
            s["frames"] = torch.randint(0, 256, (self.num_frames, 3, h, w), dtype=torch.uint8, generator=g)
            s["frame_timestamps"] = [float(t) for t in range(self.num_frames)]
        else:
            s["image"] = torch.randint(0, 256, (3, h, w), dtype=torch.uint8, generator=g)
        return s


@register_dataset("target_train")
class TargetTrainDataset(Dataset):
    """Images of the benchmark TRAIN splits (one item per image / stored frame), as one mixture component.

    Args:
        data_root: benchmark LMDB root (``make_datasets/build_all_lmdb.sh``).
        datasets: benchmark directories with a ``train`` split.
    """

    def __init__(self, data_root: str, datasets: Sequence[str] = ("vizwiz", "textvqa", "egoblind")):
        self.parts = [LMDBVQADataset(os.path.join(data_root, d), split="train", mode="image") for d in datasets]
        self.offsets = [0]
        for p in self.parts:
            self.offsets.append(self.offsets[-1] + len(p))
        self.sizes = [s for p in self.parts for s in p.sizes]

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, i: int) -> dict[str, Any]:
        import bisect

        k = bisect.bisect_right(self.offsets, i) - 1
        return self.parts[k][i - self.offsets[k]]


# The general distillation corpus format and its dataset live in corpus.py (registered as image_corpus).
from onboard_vlm.data import corpus as _corpus  # noqa: E402,F401
