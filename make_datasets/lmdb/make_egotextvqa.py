"""EgoTextVQA (CVPR 2025, zhousheng97/EgoTextVQA) -> LMDB (test-only benchmark).

Annotations: ``data/egotextvqa_{outdoor,indoor}_annotation.csv`` with
``video_id, question_id, question_type, timestamp (s), question, correct_answer``
(Outdoor: 4,848 QA / 694 videos; Indoor: 2,216 QA / 813 Ego4D videos).

Real-time QA: the answer must be inferred from the video *before* the question timestamp, so
frames are sampled uniformly in ``[0, timestamp]``. The paper's resolution study applies to
Outdoor only (high: 1920x1080 / 1280x720, low: 960x540 / 640x360); build one LMDB per setting::

    --subset outdoor --resize 1920x1080 --output_path .../egotextvqa_outdoor_high
    --subset outdoor --resize 960x540   --output_path .../egotextvqa_outdoor_low

(Gemma 4 then resizes every frame to its token budget: at 70 tokens/frame (~161K px) both
settings collapse to the same size; the high/low gap appears at 280 (~645K px) and grows at
560 / 1120 tokens per frame.)
"""

from __future__ import annotations

import csv

from common import encode_frames, find_file, index_videos, prefix_timestamps, read_frames, video_parser
from PIL import Image

from onboard_vlm.data.lmdb_io import LMDBWriter, VQARecord


def main() -> None:
    p = video_parser(__doc__, num_frames=16)
    p.add_argument("--subset", choices=["outdoor", "indoor"], required=True)
    p.add_argument("--annotation", default=None, help="path to egotextvqa_<subset>_annotation.csv")
    p.add_argument("--resize", default=None, help="WxH applied to every frame (resolution study), e.g. 960x540")
    args = p.parse_args()
    ann = args.annotation or find_file(args.input_path, [f"egotextvqa_{args.subset}_annotation.csv"])
    if ann is None:
        raise FileNotFoundError("annotation csv not found; pass --annotation")
    rows = list(csv.DictReader(open(ann, encoding="utf-8")))[: args.limit]
    videos = index_videos(args.video_dir or args.input_path)
    missing = sum(str(r["video_id"]) not in videos for r in rows)
    print(f"[egotextvqa/{args.subset}] {len(rows)} questions, {len({r['video_id'] for r in rows})} videos, {missing} questions without a video file")
    if args.dry_run:
        return
    size = tuple(int(x) for x in args.resize.lower().split("x")) if args.resize else None
    with LMDBWriter(args.output_path, "test", f"egotextvqa_{args.subset}") as w:
        for r in rows:
            vp = videos.get(str(r["video_id"]))
            if vp is None:
                continue
            t = float(r["timestamp (s)"])
            ts = prefix_timestamps(t, args.num_frames)
            frames = read_frames(vp, ts)
            if size:
                frames = [f.resize(size, Image.BICUBIC) for f in frames]
            enc, hw = encode_frames(frames, args.max_side if not size else None)
            w.add(
                VQARecord(
                    sample_id=f"{args.subset}_{r['video_id']}_{r['question_id']}", dataset="egotextvqa", split="test",
                    media_type="video", frames=enc, frame_timestamps=ts, question=r["question"],
                    answers=[r["correct_answer"]], category=r["question_type"].lower(),
                    meta={"subset": args.subset, "video_id": r["video_id"], "question_id": r["question_id"], "timestamp": t, "resize": args.resize},
                ),
                hw,
            )


if __name__ == "__main__":
    main()
