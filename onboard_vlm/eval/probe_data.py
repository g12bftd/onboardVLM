"""Labelled datasets for the E2 probes (standard on-disk layouts of the official releases)."""

from __future__ import annotations

import glob
import io
import os
import tarfile
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from onboard_vlm.data.lmdb_io import decode_image


class ImageFolderClassification(Dataset):
    """``root/<class>/<image>`` (e.g. ImageNet-1k after the standard val re-organisation).

    ``per_class`` keeps the first N images of every class (sorted file order): a fixed, reproducible
    few-shot training subset for the linear / k-NN probes."""

    def __init__(self, root: str, per_class: int | None = None, classes: list[str] | None = None):
        self.classes = classes or sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.items = []
        for c, name in enumerate(self.classes):
            files = sorted(glob.glob(os.path.join(root, name, "*")))[:per_class]
            self.items += [(f, c) for f in files]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        path, label = self.items[i]
        with open(path, "rb") as f:
            return {"image": decode_image(f.read()), "label": label}


class WdsClassification(Dataset):
    """WebDataset tars with ``<key>.jpg`` / ``<key>.cls`` members (``timm/imagenet-1k-wds``).

    ``per_class`` streams the tars in order and keeps the first N images of every class."""

    def __init__(self, pattern: str, per_class: int | None = None, num_classes: int = 1000):
        self.blobs: list[tuple[bytes, int]] = []
        counts: Counter = Counter()
        for shard in sorted(glob.glob(pattern)):
            pending: dict[str, dict] = {}
            with tarfile.open(shard, mode="r|*") as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    key, ext = os.path.splitext(m.name)
                    pending.setdefault(key, {})[ext] = tf.extractfile(m).read()
                    rec = pending[key]
                    if ".cls" in rec and (".jpg" in rec or ".jpeg" in rec):
                        label = int(rec[".cls"].decode().strip())
                        if per_class is None or counts[label] < per_class:
                            self.blobs.append((rec.get(".jpg") or rec[".jpeg"], label))
                            counts[label] += 1
                        del pending[key]
            if per_class is not None and len(counts) == num_classes and min(counts.values()) >= per_class:
                break

    def __len__(self) -> int:
        return len(self.blobs)

    def __getitem__(self, i: int) -> dict:
        data, label = self.blobs[i]
        return {"image": decode_image(data), "label": label}


class ADE20KSegmentation(Dataset):
    """``ADEChallengeData2016/{images,annotations}/{training,validation}``; 150 classes, 0 (other) -> ignore."""

    NUM_CLASSES = 150

    def __init__(self, root: str, split: str = "validation", limit: int | None = None):
        self.images = sorted(glob.glob(os.path.join(root, "images", split, "*.jpg")))[:limit]
        self.root, self.split = root, split

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> dict:
        from PIL import Image

        path = self.images[i]
        ann = os.path.join(self.root, "annotations", self.split, os.path.basename(path)[:-4] + ".png")
        lab = torch.from_numpy(np.asarray(Image.open(ann), dtype=np.int64))
        lab = torch.where(lab == 0, torch.full_like(lab, 255), lab - 1)
        with open(path, "rb") as f:
            return {"image": decode_image(f.read()), "seg": lab}


class NYUv2Depth(Dataset):
    """``<root>/<split>/<idx>.png`` + ``<idx>_depth.npy`` (metres) from ``make_datasets/eval_sets/make_nyuv2.py``
    (official labelled set, 795 train / 654 test). Targets are log-depth; invalid (<= 0) pixels are NaN."""

    def __init__(self, root: str, split: str = "test", limit: int | None = None):
        self.images = sorted(glob.glob(os.path.join(root, split, "*[0-9].png")))[:limit]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> dict:
        path = self.images[i]
        depth = torch.from_numpy(np.load(path[:-4] + "_depth.npy").astype(np.float32))
        log_depth = torch.where(depth > 0, depth.clamp_min(1e-3).log(), torch.full_like(depth, float("nan")))
        with open(path, "rb") as f:
            return {"image": decode_image(f.read()), "depth": log_depth}


def png_bytes(arr: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
