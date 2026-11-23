"""Build a sharded LMDB image corpus for Stage-1 distillation (label-free; only pixels are stored).

Sources:
  --source folder       --input_path DIR                       (recursive *.jpg/*.png/*.webp)
  --source webdataset   --input_path "DIR/*.tar"               (img2dataset / pixparse *-wds shards)
  --source hf           --input_path HF_ID --hf_split train --image_column image [--hf_config NAME]

Every image is filtered (min side, aspect ratio), checked against the benchmark blocklist
(exact SHA-1 + dHash within --near_dup_bits) and optionally de-duplicated within the corpus.
Run N workers in parallel with --worker_id i --num_workers N (each processes a 1/N slice).

Examples:
  python make_datasets/corpus/make_image_corpus.py --source webdataset --input_path "/data/cc12m-wds/*.tar" \\
      --name cc12m --output_path $CORPUS_ROOT/general/cc12m --blocklist $CORPUS_ROOT/eval_blocklist.json --dedup
  python make_datasets/corpus/make_image_corpus.py --source hf --input_path HuggingFaceM4/the_cauldron \\
      --hf_config chartqa --image_column images --name cauldron_chartqa --output_path $CORPUS_ROOT/text_rich/chartqa
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_common import ShardedCorpusWriter, add_common_args, take_slice  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def iter_folder(root: str):
    for d, _, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXT):
                p = os.path.join(d, f)
                with open(p, "rb") as fh:
                    yield os.path.relpath(p, root), fh.read(), {}


def iter_webdataset(pattern: str, worker_id: int, num_workers: int):
    shards = sorted(glob.glob(pattern))
    for shard in take_slice(shards, worker_id, num_workers):
        with tarfile.open(shard) as tf:
            for m in tf:
                if m.isfile() and m.name.lower().endswith(IMG_EXT):
                    yield f"{os.path.basename(shard)}/{m.name}", tf.extractfile(m).read(), {}


def iter_hf(dataset_id: str, split: str, config: str | None, column: str, worker_id: int, num_workers: int):
    from datasets import load_dataset

    ds = load_dataset(dataset_id, config, split=split, streaming=True)
    if num_workers > 1:
        ds = ds.shard(num_shards=num_workers, index=worker_id)
    for i, row in enumerate(ds):
        imgs = row[column]
        for j, im in enumerate(imgs if isinstance(imgs, list) else [imgs]):
            data = im["bytes"] if isinstance(im, dict) and im.get("bytes") else im
            if hasattr(data, "save"):  # PIL image
                buf = io.BytesIO()
                data.convert("RGB").save(buf, format="JPEG", quality=95)
                data = buf.getvalue()
            yield f"{config or 'default'}/{split}/{worker_id}_{i}_{j}", data, {}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["folder", "webdataset", "hf"], required=True)
    p.add_argument("--input_path", required=True)
    p.add_argument("--hf_split", default="train")
    p.add_argument("--hf_config", default=None)
    p.add_argument("--image_column", default="image")
    add_common_args(p)
    args = p.parse_args()
    if args.source == "folder":
        items = take_slice(iter_folder(args.input_path), args.worker_id, args.num_workers)
    elif args.source == "webdataset":
        items = iter_webdataset(args.input_path, args.worker_id, args.num_workers)
    else:
        items = iter_hf(args.input_path, args.hf_split, args.hf_config, args.image_column, args.worker_id, args.num_workers)
    w = ShardedCorpusWriter(args)
    for n, (key, data, meta) in enumerate(items):
        if args.limit is not None and w.stats["written"] >= args.limit:
            break
        w.add(key, data, meta)
        if n % 10_000 == 0:
            print(f"[{args.name}] seen {n}: {w.stats}", flush=True)
    w.close()


if __name__ == "__main__":
    main()
