"""VizWiz-VQA -> LMDB.

Expected raw layout (``make_datasets/download/download_vizwiz.sh``)::

    <input>/train/VizWiz_train_*.jpg   <input>/val/...   <input>/test/...
    <input>/Annotations/{train,val,test}.json      (or the json files directly under <input>)

Annotation entries: ``{image, question, answers: [{answer, answer_confidence}] x10, answer_type, answerable}``
(test has no answers). One question per image, so ``sample_id`` is the image stem.
"""

from __future__ import annotations

import json
import os

from common import base_parser, find_file

from onboard_vlm.data.lmdb_io import LMDBWriter, VQARecord, encode_image


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()
    for split in args.splits:
        ann_path = find_file(args.input_path, [f"Annotations/{split}.json", f"{split}.json"])
        if ann_path is None:
            raise FileNotFoundError(f"no {split}.json under {args.input_path}")
        anns = json.load(open(ann_path))[: args.limit]
        img_dir = os.path.join(args.input_path, split)
        print(f"[vizwiz/{split}] {len(anns)} questions from {ann_path}")
        if args.dry_run:
            continue
        with LMDBWriter(args.output_path, split, "vizwiz_vqa") as w:
            for a in anns:
                img_bytes, hw = encode_image(os.path.join(img_dir, a["image"]), args.max_side)
                answers = [x["answer"] for x in a.get("answers", [])]
                w.add(
                    VQARecord(
                        sample_id=os.path.splitext(a["image"])[0], dataset="vizwiz_vqa", split=split, image=img_bytes,
                        question=a["question"], answers=answers, answerable=a.get("answerable"),
                        answer_type=a.get("answer_type"), category=a.get("answer_type"),
                        meta={"image": a["image"], "answer_confidence": [x.get("answer_confidence") for x in a.get("answers", [])]},
                    ),
                    hw,
                )


if __name__ == "__main__":
    main()
