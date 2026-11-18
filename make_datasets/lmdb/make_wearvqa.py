"""WearVQA (NeurIPS 2025 D&B; HF ``tonyliao-meta/WearVQA``) -> LMDB (single ``test`` split).

The release format was not inspectable when this builder was written, so columns are
auto-detected from common names and can be overridden on the command line. Run once with
``--dry_run`` to print the detected columns. Accepted inputs:

* a hub id or local ``hf download`` directory readable by ``datasets.load_dataset``;
* a ``.json`` / ``.jsonl`` / ``.csv`` / ``.parquet`` annotation file (+ ``--image_root``).

Slicing metadata stored in ``meta``: ``quality_issues`` (blur, cut-off, low light, unmagnified,
occlusion, rotation; list), ``domain`` (7 image domains) and ``task_type`` (10 task types).
"""

from __future__ import annotations

import json
import os

from common import base_parser, image_from_any, pick

from onboard_vlm.data.lmdb_io import LMDBWriter, VQARecord

COLS = {
    "id": ["id", "question_id", "qid", "uid", "sample_id"],
    "image": ["image", "img", "image_path", "image_file", "file_name", "filename", "image_name"],
    "question": ["question", "query", "prompt"],
    "answer": ["answer", "answers", "gt_answer", "ground_truth", "reference", "gt"],
    "quality": ["quality_issues", "quality_issue", "image_quality", "image_quality_issues", "artifacts", "artifact_type", "issues"],
    "domain": ["domain", "image_domain", "scene", "image_category"],
    "task": ["task_type", "task", "question_type", "cognitive_task", "category"],
}


def _load_rows(path: str, split: str):
    ext = os.path.splitext(path)[1].lower()
    if os.path.isfile(path) and ext in (".json", ".jsonl"):
        with open(path) as f:
            rows = [json.loads(line) for line in f] if ext == ".jsonl" else json.load(f)
        return rows if isinstance(rows, list) else rows.get("data", rows)
    from datasets import load_dataset

    if os.path.isfile(path):  # csv / parquet
        return load_dataset(ext.lstrip("."), data_files=path, split="train")
    return load_dataset(path, split=split)


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
        return [s.strip() for s in v.replace(";", ",").split(",") if s.strip()]
    return [str(x) for x in v]


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--hf_split", default="test")
    p.add_argument("--image_root", default=None)
    for k in COLS:
        p.add_argument(f"--{k}_column", default=None)
    args = p.parse_args()
    rows = _load_rows(args.input_path, args.hf_split)
    n = len(rows) if args.limit is None else min(args.limit, len(rows))
    first = dict(rows[0])
    detected = {k: (getattr(args, f"{k}_column") or next((c for c in v if c in first), None)) for k, v in COLS.items()}
    print(f"[wearvqa] {len(rows)} rows; columns={list(first)}; detected={detected}")
    if detected["image"] is None or detected["question"] is None or detected["answer"] is None:
        raise SystemExit("could not detect image/question/answer columns; pass --image_column/--question_column/--answer_column")
    if args.dry_run:
        return
    image_root = args.image_root or (args.input_path if os.path.isdir(args.input_path) else os.path.dirname(args.input_path))
    with LMDBWriter(args.output_path, "test", "wearvqa") as w:
        for i in range(n):
            row = dict(rows[i])
            img_bytes, hw = image_from_any(row[detected["image"]], image_root, args.max_side)
            ans = row[detected["answer"]]
            answers = [str(a) for a in ans] if isinstance(ans, (list, tuple)) else [str(ans)]
            meta = {
                "quality_issues": _as_list(pick(row, [], detected["quality"])) if detected["quality"] else [],
                "domain": pick(row, [], detected["domain"]) if detected["domain"] else None,
                "task_type": pick(row, [], detected["task"]) if detected["task"] else None,
            }
            qid = str(pick(row, [], detected["id"], default=i)) if detected["id"] else str(i)
            w.add(
                VQARecord(sample_id=f"wearvqa_{qid}", dataset="wearvqa", split="test", image=img_bytes,
                          question=str(row[detected["question"]]), answers=answers, category=meta["task_type"], meta=meta),
                hw,
            )


if __name__ == "__main__":
    main()
