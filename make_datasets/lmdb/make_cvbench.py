"""CV-Bench (Cambrian-1, ``nyu-visionx/CV-Bench``) -> LMDB (single ``test`` split, 2,638 questions).

Reads the HF dataset (hub id or a local ``hf download`` directory). Columns used: ``idx, type (2D/3D),
task (Count/Relation/Depth/Distance), image, question, choices, answer ("(A)"), prompt, source
(ADE20K/COCO/Omni3D)``. ``source`` is kept in ``meta`` for the official overall score.
"""

from __future__ import annotations

import re

from common import base_parser, image_from_any

from onboard_vlm.data.lmdb_io import LMDBWriter, VQARecord


def main() -> None:
    p = base_parser(__doc__)
    p.set_defaults(input_path="nyu-visionx/CV-Bench")
    p.add_argument("--hf_split", default="test")
    args = p.parse_args()
    from datasets import load_dataset

    ds = load_dataset(args.input_path, split=args.hf_split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[cvbench] {len(ds)} questions; columns={ds.column_names}")
    if args.dry_run:
        return
    with LMDBWriter(args.output_path, "test", "cvbench") as w:
        for i, row in enumerate(ds):
            img_bytes, hw = image_from_any(row["image"], max_side=args.max_side)
            m = re.search(r"\(?([A-F])\)?", str(row["answer"]))
            w.add(
                VQARecord(
                    sample_id=f"cvbench_{i:05d}", dataset="cvbench", split="test", image=img_bytes,
                    question=row["question"], answers=[str(row["answer"])], choices=list(row["choices"]),
                    answer_index="ABCDEF".index(m.group(1)) if m else -1, category=row.get("task"),
                    meta={"source": row.get("source"), "type": row.get("type"), "prompt": row.get("prompt"), "idx": row.get("idx")},
                ),
                hw,
            )


if __name__ == "__main__":
    main()
