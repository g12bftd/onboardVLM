"""EgoSchema -> LMDB.

Annotations from github.com/egoschema/EgoSchema: ``questions.json`` (5,031 items with
``q_uid, question, option 0..option 4``) and ``subset_answers.json`` (``{q_uid: int}`` for the
500 public answers). Videos are ~3-minute clips named ``<q_uid>.mp4``. ``--num_frames`` frames
are sampled segment-centred over the whole clip.

Writes ``test.lmdb`` (all 5,031; ``answer_index = -1`` where hidden) and ``subset.lmdb`` (the 500
with answers, for local accuracy).
"""

from __future__ import annotations

import json

from common import centered_timestamps, encode_frames, find_file, index_videos, read_frames, video_duration, video_parser

from onboard_vlm.data.lmdb_io import LMDBWriter, VQARecord


def main() -> None:
    p = video_parser(__doc__, num_frames=32)
    p.add_argument("--splits", nargs="+", default=["subset", "test"])
    args = p.parse_args()
    qs = json.load(open(find_file(args.input_path, ["questions.json"])))
    answers = json.load(open(find_file(args.input_path, ["subset_answers.json"])))
    videos = index_videos(args.video_dir or args.input_path)
    print(f"[egoschema] {len(qs)} questions, {len(answers)} public answers, {sum(q['q_uid'] in videos for q in qs)} with video")
    if args.dry_run:
        return
    for split in args.splits:
        items = [q for q in qs if split == "test" or q["q_uid"] in answers][: args.limit]
        with LMDBWriter(args.output_path, split, "egoschema") as w:
            for q in items:
                vp = videos.get(q["q_uid"])
                if vp is None:
                    continue
                ts = centered_timestamps(video_duration(vp), args.num_frames)
                frames, hw = encode_frames(read_frames(vp, ts), args.max_side)
                w.add(
                    VQARecord(
                        sample_id=q["q_uid"], dataset="egoschema", split=split, media_type="video", frames=frames,
                        frame_timestamps=ts, question=q["question"], choices=[q[f"option {i}"] for i in range(5)],
                        answer_index=int(answers.get(q["q_uid"], -1)), meta={"q_uid": q["q_uid"]},
                    ),
                    hw,
                )


if __name__ == "__main__":
    main()
