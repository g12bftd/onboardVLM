"""Image fingerprints for de-duplication and benchmark decontamination of distillation corpora."""

from __future__ import annotations

import hashlib
import io

from PIL import Image, ImageOps


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def dhash(img: Image.Image | bytes, size: int = 8) -> int:
    """64-bit difference hash (robust to resizing / re-encoding / mild colour changes)."""
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    g = ImageOps.exif_transpose(img).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1) : (r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | int(row[c] > row[c + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class NearDupIndex:
    """Exact-match index on dHash bands: two hashes within ``max_dist`` bits share at least one of
    ``max_dist + 1`` bands (pigeonhole), so candidates are found without a linear scan."""

    def __init__(self, max_dist: int = 4, bits: int = 64):
        self.max_dist, self.bands = max_dist, max_dist + 1
        self.width = bits // self.bands
        self.tables: list[dict[int, list[int]]] = [{} for _ in range(self.bands)]

    def _keys(self, h: int) -> list[int]:
        mask = (1 << self.width) - 1
        return [(h >> (i * self.width)) & mask for i in range(self.bands)]

    def add(self, h: int) -> None:
        for t, k in zip(self.tables, self._keys(h)):
            t.setdefault(k, []).append(h)

    def contains(self, h: int) -> bool:
        for t, k in zip(self.tables, self._keys(h)):
            for c in t.get(k, ()):
                if hamming(c, h) <= self.max_dist:
                    return True
        return False

    def __len__(self) -> int:
        return sum(len(v) for v in self.tables[0].values())
