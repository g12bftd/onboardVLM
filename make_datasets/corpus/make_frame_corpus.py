"""Build a sharded LMDB image corpus from egocentric *videos* (Ego4D, EPIC-KITCHENS, Ego-Exo4D ego
views, HoloAssist, Aria Everyday Activities, ...).

Frames are taken every 1/--fps seconds, near-duplicate consecutive frames (dHash within
--still_bits) are skipped, and videos / time spans used by benchmarks are excluded:
``--blocklist`` (from make_eval_blocklist.py) lists Ego4D video uids (and optional second ranges)
behind EgoSchema and EgoTextVQA-Indoor, plus dHashes of every benchmark frame.

Example:
  python make_datasets/corpus/make_frame_corpus.py --input_path /data/ego4d/v2/full_scale --fps 0.5 \\
      --name ego4d --output_path $CORPUS_ROOT/egocentric/ego4d --blocklist $CORPUS_ROOT/eval_blocklist.json \\
      --worker_id 0 --num_workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_common import ShardedCorpusWriter, add_common_args, take_slice  # noqa: E402

from onboard_vlm.data.hashing import dhash, hamming  # noqa: E402

VID_EXT = (".mp4", ".mkv", ".webm", ".mov", ".avi")


def iter_frames(path: str, fps: float, max_frames: int | None, skip_spans: list[tuple[float, float]]):
    """Seek-based sampling (efficient for sparse fps on hour-long videos). Yields (t, PIL image)."""
    import av

    with av.open(path) as c:
        s = c.streams.video[0]
        dur = float(s.duration * s.time_base) if s.duration else (c.duration or 0) / 1e6
        t, n = 0.0, 0
        while t < dur and (max_frames is None or n < max_frames):
            if not any(a <= t <= b for a, b in skip_spans):
                c.seek(int(t / s.time_base), stream=s, backward=True, any_frame=False)
                for frame in c.decode(s):
                    if frame.pts is not None and float(frame.pts * s.time_base) + 1e-6 >= t:
                        yield t, frame.to_image()
                        n += 1
                        break
            t += 1.0 / fps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_path", required=True, help="directory of videos (recursive)")
    p.add_argument("--fps", type=float, default=0.5)
    p.add_argument("--max_frames_per_video", type=int, default=None)
    p.add_argument("--still_bits", type=int, default=6, help="skip a frame within this dHash distance of the last kept one")
    p.add_argument("--exclude_mode", choices=["video", "span"], default="video",
                   help="video: drop whole benchmark source videos (safest); span: drop only the benchmark clips' seconds")
    add_common_args(p)
    args = p.parse_args()
    bl = json.load(open(args.blocklist)) if args.blocklist else {}
    bad_videos = set(bl.get("video_ids", []))
    spans: dict[str, list[tuple[float, float]]] = {}
    for vid, a, b in bl.get("video_spans", []):
        spans.setdefault(vid, []).append((float(a), float(b)))
    videos = sorted(
        os.path.join(d, f) for d, _, fs in os.walk(args.input_path) for f in fs if f.lower().endswith(VID_EXT)
    )
    w = ShardedCorpusWriter(args)
    skipped = 0
    for path in take_slice(videos, args.worker_id, args.num_workers):
        vid = os.path.splitext(os.path.basename(path))[0]
        if args.exclude_mode == "video" and (vid in bad_videos or vid in spans):
            skipped += 1
            continue
        last = None
        try:
            for t, img in iter_frames(path, args.fps, args.max_frames_per_video, spans.get(vid, []) if args.exclude_mode == "span" else []):
                h = dhash(img)
                if last is not None and hamming(h, last) <= args.still_bits:
                    continue
                last = h
                w.add(f"{vid}/{t:.2f}", img, {"video_id": vid, "t": round(t, 2)})
                if args.limit is not None and w.stats["written"] >= args.limit:
                    break
        except Exception as e:  # corrupt video: report and continue
            print(f"[{args.name}] skip {path}: {e}", flush=True)
        if args.limit is not None and w.stats["written"] >= args.limit:
            break
    print(f"[{args.name}] excluded {skipped} benchmark source videos")
    w.close()


if __name__ == "__main__":
    main()
