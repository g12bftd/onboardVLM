"""Check that a video download is complete: every video the annotations refer to is on disk.

Download scripts call this last, so an incomplete download fails its job (and is retried when the
pipeline is re-submitted) instead of silently producing a benchmark with missing questions.

    python verify_videos.py --name egoschema --video_dir RAW/egoschema/videos \
        --ids RAW/egoschema/questions.json --key q_uid [--required RAW/egoschema/subset_answers.json]

``--ids`` / ``--required``: CSV (column ``--key``), JSON (a list of objects with ``--key``, a list of
ids, or an object whose keys are the ids) or JSONL. Videos are matched by file stem anywhere below
``--video_dir``; numeric ids also match zero-padded stems (as the benchmark builders do).
``--max_missing``: tolerated missing videos among ``--ids`` (a count, or a fraction if < 1); none of
the ``--required`` ids may be missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

VIDEO_EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def load_ids(path: str, key: str) -> list[str]:
    if path.endswith(".csv"):
        with open(path, encoding="utf-8", newline="") as fh:
            return [str(r[key]).strip() for r in csv.DictReader(fh) if str(r.get(key) or "").strip()]
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:  # JSONL
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        return [str(k) for k in data]
    return [str(d[key] if isinstance(d, dict) else d).strip() for d in data]


def index_stems(root: str) -> set[str]:
    return {os.path.splitext(f)[0] for _, _, fs in os.walk(root) for f in fs if f.lower().endswith(VIDEO_EXT)}


def present(vid: str, stems: set[str]) -> bool:
    cands = {vid}
    if vid.isdigit():
        n = int(vid)
        cands |= {f"{n:04d}", f"{n:05d}", f"{n:06d}", f"v_{n:05d}", str(n)}
    return bool(cands & stems)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--video_dir", required=True)
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--key", default="video_id")
    p.add_argument("--required", nargs="*", default=[])
    p.add_argument("--max_missing", type=float, default=0)
    args = p.parse_args()

    stems = index_stems(args.video_dir)
    ids = sorted({i for path in args.ids for i in load_ids(path, args.key)})
    required = sorted({i for path in args.required for i in load_ids(path, args.key)})
    missing = [i for i in ids if not present(i, stems)]
    missing_req = [i for i in required if not present(i, stems)]
    allowed = args.max_missing * len(ids) if 0 < args.max_missing < 1 else args.max_missing
    print(f"[verify {args.name}] {len(stems)} video files under {args.video_dir}; "
          f"{len(ids) - len(missing)}/{len(ids)} referenced videos present"
          + (f", {len(required) - len(missing_req)}/{len(required)} required present" if required else ""))
    if missing:
        print(f"[verify {args.name}] missing ({len(missing)}): {' '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")
    if missing_req or len(missing) > allowed:
        sys.exit(f"[verify {args.name}] INCOMPLETE download (allowed missing: {allowed:g}); re-run the download")


if __name__ == "__main__":
    main()
