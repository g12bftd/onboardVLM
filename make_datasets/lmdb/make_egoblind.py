"""EgoBlind (doc-doc/EgoBlind) -> LMDB.

Annotations are CSVs with ``question_id, video_name, question, answer0..answer3, type, start-time/s``
(``test_half_release.csv`` is the publicly released half of the test set; pass the train CSV
from the Google-Drive training folder with ``--train_csv``). Questions are answered from the
video *up to the moment they were asked*, so ``--num_frames`` frames are sampled uniformly in
``[max(0, t - window), t]`` with ``t = start-time/s`` (the last frame is at question time).

Videos are located by file stem: ``<video_name>``, zero-padded ``<video_name:05d>``, or via
``--video_pattern`` (e.g. ``"{:05d}"``).
"""

from __future__ import annotations

import csv
import os

from common import encode_frames, index_videos, prefix_timestamps, read_frames, video_parser

from onboard_vlm.data.lmdb_io import LMDBWriter, VQARecord


def _video_path(videos: dict[str, str], name: str, pattern: str | None) -> str | None:
    cands = [name]
    if name.isdigit():
        cands += [f"{int(name):05d}", f"{int(name):04d}", f"{int(name):06d}", f"v_{int(name):05d}"]
    if pattern:
        cands.insert(0, pattern.format(int(name) if name.isdigit() else name))
    return next((videos[c] for c in cands if c in videos), None)


def main() -> None:
    p = video_parser(__doc__, num_frames=16)
    p.add_argument("--test_csv", default="test_half_release.csv")
    p.add_argument("--train_csv", default=None)
    p.add_argument("--window", type=float, default=None, help="seconds of history before the question (default: all)")
    p.add_argument("--video_pattern", default=None)
    args = p.parse_args()
    videos = index_videos(args.video_dir or args.input_path)
    print(f"[egoblind] indexed {len(videos)} videos")
    splits = {"test": args.test_csv}
    if args.train_csv:
        splits["train"] = args.train_csv
    for split, csv_name in splits.items():
        path = csv_name if os.path.isabs(csv_name) else os.path.join(args.input_path, csv_name)
        rows = list(csv.DictReader(open(path, encoding="utf-8")))[: args.limit]
        missing = [r for r in rows if _video_path(videos, r["video_name"], args.video_pattern) is None]
        print(f"[egoblind/{split}] {len(rows)} questions, {len(missing)} without a video file")
        if args.dry_run:
            continue
        with LMDBWriter(args.output_path, split, "egoblind") as w:
            for r in rows:
                vp = _video_path(videos, r["video_name"], args.video_pattern)
                if vp is None:
                    continue
                t = float(r.get("start-time/s") or 0.0)
                start = max(0.0, t - args.window) if args.window else 0.0
                ts = prefix_timestamps(t, args.num_frames, start)
                frames, hw = encode_frames(read_frames(vp, ts), args.max_side)
                answers = [r.get(f"answer{i}") or "" for i in range(4) if f"answer{i}" in r] or [r.get("answer", "")]
                w.add(
                    VQARecord(
                        sample_id=r["question_id"], dataset="egoblind", split=split, media_type="video",
                        frames=frames, frame_timestamps=ts, question=r["question"], answers=answers,
                        category=(r.get("type") or "").strip().lower().rstrip("."),
                        meta={"question_id": r["question_id"], "video_name": r["video_name"], "start_time": t},
                    ),
                    hw,
                )


if __name__ == "__main__":
    main()
