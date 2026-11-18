"""TextVQA v0.5.1 -> LMDB.

Expected raw layout (``make_datasets/download/download_textvqa.sh``)::

    <input>/train_images/*.jpg     (train AND val images; from train_val_images.zip)
    <input>/test_images/*.jpg
    <input>/TextVQA_0.5.1_{train,val,test}.json

Each json has ``data: [{question_id, question, image_id, answers (10; absent for test), ...}]``.
Images repeat across questions: ``sample_id = "<image_id>::q<question_id>"``.
"""

from __future__ import annotations

import json
import os

from common import base_parser

from onboard_vlm.data.lmdb_io import LMDBWriter, VQARecord, encode_image


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()
    for split in args.splits:
        ann = json.load(open(os.path.join(args.input_path, f"TextVQA_0.5.1_{split}.json")))["data"][: args.limit]
        img_dir = os.path.join(args.input_path, "test_images" if split == "test" else "train_images")
        print(f"[textvqa/{split}] {len(ann)} questions, {len({a['image_id'] for a in ann})} images")
        if args.dry_run:
            continue
        cache: dict[str, tuple[bytes, tuple[int, int]]] = {}
        with LMDBWriter(args.output_path, split, "textvqa") as w:
            for a in ann:
                if a["image_id"] not in cache:
                    cache = {a["image_id"]: encode_image(os.path.join(img_dir, a["image_id"] + ".jpg"), args.max_side)}
                img_bytes, hw = cache[a["image_id"]]
                w.add(
                    VQARecord(
                        sample_id=f"{a['image_id']}::q{a['question_id']}", dataset="textvqa", split=split, image=img_bytes,
                        question=a["question"], answers=list(a.get("answers", [])),
                        category=(a.get("image_classes") or [None])[0],
                        meta={"question_id": a["question_id"], "image_id": a["image_id"], "image_classes": a.get("image_classes")},
                    ),
                    hw,
                )


if __name__ == "__main__":
    main()
