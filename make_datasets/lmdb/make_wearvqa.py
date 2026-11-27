"""WearVQA (NeurIPS 2025 D&B; HF ``tonyliao-meta/WearVQA``) -> LMDB (single ``test`` split: it is an evaluation set).

The release (checked 2026-11-27) is one WebDataset tar, ``wearvqa.tar``, with ``<k>.jpg`` + ``<k>.json`` per sample
(2,520 samples). Each JSON has ``idx``, ``question``, ``response`` (the reference answer), ``domain``,
``question_type`` and six yes/no image-quality flags (``is_not_zoomed_in``, ``is_leveling``, ``is_cut_off``,
``is_blur``, ``is_low_light``, ``is_occluded``; "yes" = the issue is present), plus ``pix_image_file_name`` and
``hand_finger_elements``. The tar is read directly (no extraction). Loading the directory with
``datasets.load_dataset`` would call it a "train" split (WebDataset default), which is why this builder does not.

Other inputs are still accepted (columns auto-detected, override with ``--<field>_column``): a ``datasets``-readable
directory or hub id (the requested split, or its only split), or a ``.json`` / ``.jsonl`` / ``.csv`` / ``.parquet``
annotation file (+ ``--image_root``). ``--dry_run`` prints what was found.

Stored per sample: ``answers = [response]``, ``category = question_type``, and ``meta``: ``quality_issues`` (list of
``unmagnified | rotation | cut-off | blur | low light | occlusion``), ``domain``, ``task_type``, ``source_image``,
``hand_finger_elements``.
"""

from __future__ import annotations

import glob
import json
import os
import tarfile

from common import base_parser, image_from_any, pick

from onboard_vlm.data.lmdb_io import LMDBWriter, VQARecord

QUALITY_FLAGS = {"is_not_zoomed_in": "unmagnified", "is_leveling": "rotation", "is_cut_off": "cut-off",
                 "is_blur": "blur", "is_low_light": "low light", "is_occluded": "occlusion"}
COLS = {
    "id": ["idx", "id", "question_id", "qid", "uid", "sample_id", "__key__"],
    "image": ["image", "jpg", "jpeg", "png", "img", "image_path", "image_file", "file_name", "filename", "image_name"],
    "question": ["question", "query", "prompt"],
    "answer": ["response", "answer", "answers", "gt_answer", "ground_truth", "reference", "gt"],
    "quality": ["quality_issues", "quality_issue", "image_quality", "image_quality_issues", "artifacts", "artifact_type", "issues"],
    "domain": ["domain", "image_domain", "scene", "image_category"],
    "task": ["question_type", "task_type", "task", "cognitive_task", "category"],
}


def _rows_from_tars(tars: list[str]) -> list[dict]:
    """WebDataset tars -> one row per key: the JSON fields plus ``image`` (bytes of the first image member)."""
    rows: dict[str, dict] = {}
    for path in tars:
        with tarfile.open(path) as tf:
            for m in tf:
                if not m.isfile():
                    continue
                key, ext = os.path.splitext(os.path.basename(m.name))
                ext = ext.lower().lstrip(".")
                data = tf.extractfile(m).read()
                row = rows.setdefault(f"{path}:{key}", {"__key__": key})
                if ext == "json":
                    row.update(json.loads(data))
                elif ext in ("jpg", "jpeg", "png", "webp") and "image" not in row:
                    row["image"] = data
    out = [r for r in rows.values() if "image" in r and "question" in r]
    return sorted(out, key=lambda r: (int(r["idx"]) if str(r.get("idx", "")).isdigit() else 1 << 62, str(r["__key__"])))


def _load_rows(path: str, split: str):
    tars = sorted(glob.glob(os.path.join(path, "*.tar"))) if os.path.isdir(path) else ([path] if path.endswith(".tar") else [])
    if tars:
        return _rows_from_tars(tars)
    ext = os.path.splitext(path)[1].lower()
    if os.path.isfile(path) and ext in (".json", ".jsonl"):
        with open(path) as f:
            rows = [json.loads(line) for line in f] if ext == ".jsonl" else json.load(f)
        return rows if isinstance(rows, list) else rows.get("data", rows)
    from datasets import load_dataset

    if os.path.isfile(path):  # csv / parquet
        return load_dataset(ext.lstrip("."), data_files=path, split="train")
    d = load_dataset(path)
    if split in d:
        return d[split]
    if len(d) == 1:  # a single split, whatever its name (e.g. "train" for WebDataset files): the benchmark itself
        return next(iter(d.values()))
    raise SystemExit(f"split {split!r} not in {list(d)}: pass --hf_split")


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


def _quality(row: dict, column: str | None) -> list[str]:
    if column:
        return _as_list(row.get(column))
    return [name for flag, name in QUALITY_FLAGS.items() if str(row.get(flag, "")).strip().lower() in ("yes", "true", "1")]


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--hf_split", default="test")
    p.add_argument("--image_root", default=None)
    for k in COLS:
        p.add_argument(f"--{k}_column", default=None)
    args = p.parse_args()
    rows = _load_rows(args.input_path, args.hf_split)
    if not len(rows):
        raise SystemExit(f"no samples found in {args.input_path}")
    n = len(rows) if args.limit is None else min(args.limit, len(rows))
    first = dict(rows[0])
    detected = {k: (getattr(args, f"{k}_column") or next((c for c in v if c in first), None)) for k, v in COLS.items()}
    print(f"[wearvqa] {len(rows)} samples; fields={sorted(first)}; detected={detected}; "
          f"quality from {'column ' + detected['quality'] if detected['quality'] else 'the is_* flags'}")
    if detected["image"] is None or detected["question"] is None or detected["answer"] is None:
        raise SystemExit("could not detect image/question/answer columns; pass --image_column/--question_column/--answer_column")
    if args.dry_run:
        return
    image_root = args.image_root or (args.input_path if os.path.isdir(args.input_path) else os.path.dirname(args.input_path))
    seen: set[str] = set()
    with LMDBWriter(args.output_path, "test", "wearvqa") as w:
        for i in range(n):
            row = dict(rows[i])
            img_bytes, hw = image_from_any(row[detected["image"]], image_root, args.max_side)
            ans = row[detected["answer"]]
            answers = [str(a) for a in ans] if isinstance(ans, (list, tuple)) else [str(ans)]
            task = pick(row, [], detected["task"]) if detected["task"] else None
            meta = {
                "quality_issues": _quality(row, detected["quality"]),
                "domain": pick(row, [], detected["domain"]) if detected["domain"] else None,
                "task_type": task,
                "source_image": row.get("pix_image_file_name"),
                "hand_finger_elements": row.get("hand_finger_elements"),
            }
            qid = str(pick(row, [], detected["id"], default=i)) if detected["id"] else str(i)
            if qid in seen:
                raise SystemExit(f"duplicate sample id {qid}: pass --id_column")
            seen.add(qid)
            w.add(
                VQARecord(sample_id=f"wearvqa_{qid}", dataset="wearvqa", split="test", image=img_bytes,
                          question=str(row[detected["question"]]), answers=answers, category=task, meta=meta),
                hw,
            )
    print(f"[wearvqa] wrote {n} samples to {args.output_path}/test.lmdb")


if __name__ == "__main__":
    main()
