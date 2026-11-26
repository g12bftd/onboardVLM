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


def near_dup_mask(hashes, blocked, max_dist: int = 4):
    """Bool mask over ``hashes`` (uint64 array): True where some ``blocked`` hash is within ``max_dist`` bits.

    Vectorised :class:`NearDupIndex` for whole corpora (tens of millions of hashes): the same
    ``max_dist + 1`` bands (pigeonhole), with candidates compared per band value as small XOR /
    popcount matrices. Callers exclude degenerate hashes, as the builders do.
    """
    import numpy as np

    hashes, blocked = np.asarray(hashes, dtype=np.uint64), np.unique(np.asarray(blocked, dtype=np.uint64))
    hit = np.zeros(len(hashes), dtype=bool)
    if len(hashes) == 0 or len(blocked) == 0:
        return hit
    bands = max_dist + 1
    width = 64 // bands
    mask = np.uint64((1 << width) - 1)
    for i in range(bands):
        shift = np.uint64(i * width)
        hk, bk = (hashes >> shift) & mask, (blocked >> shift) & mask
        h_order, b_order = np.argsort(hk, kind="stable"), np.argsort(bk, kind="stable")
        hk_s, bk_s = hk[h_order], bk[b_order]
        common = np.intersect1d(hk_s, bk_s, assume_unique=False)
        h_lo, h_hi = np.searchsorted(hk_s, common, "left"), np.searchsorted(hk_s, common, "right")
        b_lo, b_hi = np.searchsorted(bk_s, common, "left"), np.searchsorted(bk_s, common, "right")
        for hl, hh, bl, bh in zip(h_lo, h_hi, b_lo, b_hi):
            rows = h_order[hl:hh]
            rows = rows[~hit[rows]]
            if len(rows) == 0:
                continue
            d = np.bitwise_count(hashes[rows, None] ^ blocked[b_order[bl:bh]][None, :])
            hit[rows[(d <= max_dist).any(axis=1)]] = True
    return hit
