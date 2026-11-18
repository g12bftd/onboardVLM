"""Shared helpers for the LMDB builders (video frame sampling, field auto-detection, CLI)."""

from __future__ import annotations

import argparse
import glob
import io
import os
from typing import Any, Iterable, Sequence

from PIL import Image

from onboard_vlm.data.lmdb_io import encode_image


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--input_path", required=True, help="raw download directory (see make_datasets/download)")
    p.add_argument("--output_path", required=True, help="output directory for <split>.lmdb + <split>.meta.json")
    p.add_argument("--max_side", type=int, default=2048, help="downscale images/frames so the longest side <= this")
    p.add_argument("--limit", type=int, default=None, help="only write the first N samples per split (debug)")
    p.add_argument("--dry_run", action="store_true", help="parse annotations and report counts without writing")
    return p


def video_parser(description: str, num_frames: int) -> argparse.ArgumentParser:
    p = base_parser(description)
    p.add_argument("--num_frames", type=int, default=num_frames, help="frames stored per question")
    p.add_argument("--video_dir", default=None, help="directory holding the videos (default: search under --input_path)")
    return p


def find_file(root: str, names: Sequence[str]) -> str | None:
    """First existing path among ``root/name`` or any ``root/**/name``."""
    for n in names:
        p = os.path.join(root, n)
        if os.path.isfile(p):
            return p
    for n in names:
        hits = sorted(glob.glob(os.path.join(root, "**", n), recursive=True))
        if hits:
            return hits[0]
    return None


def index_videos(root: str, exts: Sequence[str] = (".mp4", ".mkv", ".webm", ".avi", ".mov")) -> dict[str, str]:
    """``{file stem: path}`` for every video below ``root``."""
    out: dict[str, str] = {}
    for d, _, files in os.walk(root):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in exts:
                out.setdefault(stem, os.path.join(d, f))
    return out


def video_duration(path: str) -> float:
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        if s.duration is not None and s.time_base is not None:
            return float(s.duration * s.time_base)
        if c.duration is not None:
            return c.duration / 1_000_000
    raise RuntimeError(f"cannot determine duration of {path}")


def prefix_timestamps(end_time: float, num_frames: int, start_time: float = 0.0) -> list[float]:
    """``num_frames`` uniformly spaced times in ``[start, end]`` *including* ``end`` (question time)."""
    end_time = max(end_time, start_time)
    if num_frames == 1:
        return [end_time]
    step = (end_time - start_time) / (num_frames - 1)
    return [start_time + i * step for i in range(num_frames)]


def centered_timestamps(duration: float, num_frames: int) -> list[float]:
    """Segment-centred uniform sampling over a whole clip (standard for MCQ video QA)."""
    return [(i + 0.5) * duration / num_frames for i in range(num_frames)]


def read_frames(path: str, timestamps: Sequence[float]) -> list[Image.Image]:
    """Decode the frame at (or right after) each timestamp, in one forward pass over the stream."""
    import av

    targets = sorted((t, i) for i, t in enumerate(timestamps))
    out: list[Image.Image | None] = [None] * len(timestamps)
    with av.open(path) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        k = 0
        last = None
        c.seek(max(0, int(targets[0][0] / stream.time_base)) if stream.time_base else 0, stream=stream, backward=True, any_frame=False)
        for frame in c.decode(stream):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            last = frame
            while k < len(targets) and t + 1e-6 >= targets[k][0]:
                out[targets[k][1]] = frame.to_image()
                k += 1
            if k == len(targets):
                break
        while k < len(targets) and last is not None:  # timestamps past the last frame
            out[targets[k][1]] = last.to_image()
            k += 1
    if any(o is None for o in out):
        raise RuntimeError(f"could not decode frames of {path}")
    return out  # type: ignore[return-value]


def encode_frames(frames: Iterable[Image.Image], max_side: int | None, quality: int = 90) -> tuple[list[bytes], tuple[int, int]]:
    encoded, size = [], (0, 0)
    for f in frames:
        if max_side is not None and max(f.size) > max_side:
            s = max_side / max(f.size)
            f = f.resize((max(1, round(f.size[0] * s)), max(1, round(f.size[1] * s))), Image.BICUBIC)
        buf = io.BytesIO()
        f.convert("RGB").save(buf, format="JPEG", quality=quality)
        encoded.append(buf.getvalue())
        size = (f.size[1], f.size[0])
    return encoded, size


def pick(row: dict[str, Any], candidates: Sequence[str], override: str | None = None, default: Any = None) -> Any:
    """Return ``row[override]`` or the first present candidate column."""
    if override:
        return row.get(override, default)
    for c in candidates:
        if c in row and row[c] is not None:
            return row[c]
    return default


def image_from_any(v: Any, image_root: str | None = None, max_side: int | None = 2048) -> tuple[bytes, tuple[int, int]] | None:
    """HF ``Image`` feature (PIL or ``{"bytes", "path"}``), raw bytes, or a path -> encoded bytes + (h, w)."""
    if v is None:
        return None
    if isinstance(v, Image.Image):
        return encode_image(v, max_side)
    if isinstance(v, dict):
        if v.get("bytes"):
            return encode_image(v["bytes"], max_side)
        v = v.get("path")
    if isinstance(v, (bytes, bytearray)):
        return encode_image(bytes(v), max_side)
    if isinstance(v, str):
        path = v if os.path.isabs(v) or image_root is None else os.path.join(image_root, v)
        return encode_image(path, max_side)
    raise TypeError(f"unsupported image value {type(v)}")
