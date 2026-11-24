"""Build one corpus source (sharded LMDB, :mod:`onboard_vlm.data.corpus`) from downloaded images.

Input formats (all sources in make_datasets/corpus/sources.yaml map to one of these):
  --source webdataset --input_path "DIR/*.tar"        CC12M / ImageNet wds shards, SA-1B tars,
                                                     Open Images *.tar.gz (any tar, gz or not)
  --source parquet    --input_path "DIR/**/*.parquet" HF parquet exports (PixMo-Docs, SynthDoG, The Cauldron);
                      --image_column image|images     struct{bytes, path} or a list of them
  --source folder     --input_path DIR                any directory tree of *.jpg/*.png/*.webp

Filtering, benchmark decontamination and storage are shared by every source (corpus_common.py).
Parallel builds: run --worker_id 0..N-1 with --num_workers N (each takes every N-th tar / parquet file).

Examples:
  python make_datasets/corpus/make_image_corpus.py --source webdataset --input_path "$RAW/cc12m/*.tar" \\
      --name cc12m --output_path $CORPUS_ROOT/cc12m --blocklist $CORPUS_ROOT/eval_blocklist.json --worker_id 0 --num_workers 32
  python make_datasets/corpus/make_image_corpus.py --source webdataset --input_path "$RAW/openimages/train_*.tar.gz" \\
      --name openimages --id_key openimages --output_path $CORPUS_ROOT/openimages --blocklist $CORPUS_ROOT/eval_blocklist.json
  python make_datasets/corpus/make_image_corpus.py --source parquet --input_path "$RAW/the_cauldron/chartqa/*.parquet" \\
      --image_column images --name cauldron_chartqa --output_path $CORPUS_ROOT/text_rich/cauldron/chartqa
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_common import ShardedCorpusWriter, add_common_args, take_slice

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _stem(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def iter_folder(root: str, worker_id: int, num_workers: int):
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.lower().endswith(IMG_EXT))
    for p in take_slice(files, worker_id, num_workers):
        with open(p, "rb") as fh:
            yield os.path.relpath(p, root), fh.read(), {}


def iter_tars(pattern: str, worker_id: int, num_workers: int):
    shards = sorted(glob.glob(pattern, recursive=True))
    if not shards:
        raise FileNotFoundError(pattern)
    for shard in take_slice(shards, worker_id, num_workers):
        with tarfile.open(shard, mode="r|*") as tf:  # streaming: works for .tar and .tar.gz
            for m in tf:
                if m.isfile() and m.name.lower().endswith(IMG_EXT):
                    yield f"{os.path.basename(shard)}/{m.name}", tf.extractfile(m).read(), {}


def iter_parquet(pattern: str, columns: str, worker_id: int, num_workers: int):
    """``columns``: one image column or a comma-separated list (BLINK image_1..4, MMMU image_1..7). A cell
    is ``struct{bytes, path}``, raw ``bytes`` or a list of either; empty cells are skipped."""
    import pyarrow.parquet as pq

    cols = [c.strip() for c in columns.split(",") if c.strip()]
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(pattern)
    for path in take_slice(files, worker_id, num_workers):
        pf = pq.ParquetFile(path)
        present = [c for c in cols if c in pf.schema_arrow.names]
        if not present:
            raise KeyError(f"{path}: none of the image columns {cols} (has {pf.schema_arrow.names})")
        row = 0
        for batch in pf.iter_batches(batch_size=256, columns=present):
            table = {c: batch.column(c).to_pylist() for c in present}
            for r in range(batch.num_rows):
                for c in present:
                    cell = table[c][r]
                    items = cell if isinstance(cell, list) else [cell]
                    for j, im in enumerate(items):
                        data = im.get("bytes") if isinstance(im, dict) else im
                        if data:
                            yield f"{_stem(path)}/{row}_{c}_{j}", data, {}
                row += 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["webdataset", "parquet", "folder"], required=True)
    p.add_argument("--input_path", required=True)
    p.add_argument("--image_column", default="image", help="parquet column(s) holding the image(s), comma-separated")
    add_common_args(p)
    args = p.parse_args()
    if args.source == "folder":
        items = iter_folder(args.input_path, args.worker_id, args.num_workers)
    elif args.source == "webdataset":
        items = iter_tars(args.input_path, args.worker_id, args.num_workers)
    else:
        items = iter_parquet(args.input_path, args.image_column, args.worker_id, args.num_workers)
    w = ShardedCorpusWriter(args)
    for key, data, meta in items:
        w.add(key, data, meta, source_id=_stem(key) if args.id_key else None)
        if w.stats["seen"] % 20_000 == 0:
            print(f"[{args.name}] worker {args.worker_id}: {w.stats}", flush=True)
        if w.done():
            break
    w.close()


if __name__ == "__main__":
    main()
