"""Fingerprint every evaluation image / frame so the distillation corpus can be decontaminated.

Writes ``{"sha1": [...], "dhash": [...], "source_ids": {"openimages": [...]}, "video_ids": [...],
"video_spans": [[uid, start, end], ...], "counts": {...}}``:

* SHA-1 and dHash of every image and stored frame in the benchmark LMDBs (every split except
  ``train``: benchmark train splits are legitimate distillation data);
* ``source_ids.openimages``: TextVQA ``image_id`` s of the val/test splits. TextVQA images *are*
  Open Images images (same ids), so they are blocked by id as well as by content;
* SHA-1 and dHash of every image of the evaluation image sets stored in the corpus format
  (``--corpus_dirs``, from ``make_datasets/eval_sets/build_eval_sets.py``): E1 fidelity sets (COCO,
  ADE20K, ImageNet-1k val, DocVQA / InfographicVQA val, ChartQA test) and the lmms-eval E3 benchmarks
  (POPE, MMVP, MMBench, MMStar, SEED-Bench, RealWorldQA, OCRBench, AI2D, BLINK, MMMU, CV-Bench, ...),
  read directly from their fingerprint indices;
* Ego4D ``video_uid`` (+ clip seconds) behind every EgoSchema clip (``uid_to_ego4d.json`` from the
  EgoSchema repository). EgoTextVQA-Indoor re-uses EgoSchema clips, so its videos are covered too.

Example:
  python make_datasets/corpus/make_eval_blocklist.py \\
      --lmdb_dirs $LMDB_ROOT/{vizwiz,textvqa,wearvqa,egoblind,egotextvqa_outdoor_high,egotextvqa_outdoor_low,egotextvqa_indoor,egoschema,cvbench} \\
      --corpus_dirs $EVAL_ROOT/* \\
      --egoschema_uid_map $RAW_ROOT/egoschema/repo/uid_to_ego4d.json --output $CORPUS_ROOT/eval_blocklist.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from onboard_vlm.data.hashing import dhash, sha1_bytes
from onboard_vlm.data.lmdb_io import LMDBReader, load_meta

ID_SOURCES = {"textvqa": "openimages"}  # benchmark -> corpus source whose ids it shares


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lmdb_dirs", nargs="*", default=[], help="benchmark LMDB directories (every non-train split)")
    p.add_argument("--corpus_dirs", nargs="*", default=[], help="evaluation image sets in the corpus format (all images)")
    p.add_argument("--egoschema_uid_map", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    sha, dh = set(), set()
    ids: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for d in args.lmdb_dirs:
        name = os.path.basename(os.path.normpath(d))
        splits = sorted(os.path.basename(x)[: -len(".lmdb")] for x in glob.glob(os.path.join(d, "*.lmdb")))
        if not splits:
            raise FileNotFoundError(f"no *.lmdb under {d}")
        for split in splits:
            if split == "train":
                continue
            reader = LMDBReader(os.path.join(d, f"{split}.lmdb"))
            n0 = len(sha)
            for k in load_meta(d, split)["keys"]:
                rec = reader.get(k)
                for b in [rec.image] if rec.image else (rec.frames or []):
                    sha.add(sha1_bytes(b))
                    dh.add(f"{dhash(b):016x}")
                src = ID_SOURCES.get(rec.dataset)
                if src and rec.meta.get("image_id"):
                    ids.setdefault(src, set()).add(str(rec.meta["image_id"]))
            counts[f"{name}/{split}"] = len(sha) - n0
            print(f"{name}/{split}: +{len(sha) - n0} images/frames (total {len(sha)})", flush=True)
    for d in args.corpus_dirs:
        import numpy as np

        from onboard_vlm.data.corpus import list_shards, load_index

        name = os.path.basename(os.path.normpath(d))
        n0 = len(sha)
        for shard in list_shards(d):
            idx = load_index(d, shard)
            sha.update(b.ljust(20, b"\0").hex() for b in idx["sha1"].tolist())  # numpy S20 strips trailing NULs
            dh.update(f"{int(h):016x}" for h in np.asarray(idx["dhash"], dtype=np.uint64).tolist())
        counts[f"{name}/all"] = len(sha) - n0
        print(f"{name}: +{len(sha) - n0} images (total {len(sha)})", flush=True)
    if not args.lmdb_dirs and not args.corpus_dirs:
        raise SystemExit("give --lmdb_dirs and/or --corpus_dirs")
    vids, spans = set(), []
    if args.egoschema_uid_map:
        with open(args.egoschema_uid_map) as f:
            for m in json.load(f).values():
                vids.add(m["video_uid"])
                spans.append([m["video_uid"], m["starting_sec"], m["ending_sec"]])
    out = {
        "sha1": sorted(sha),
        "dhash": sorted(dh),
        "source_ids": {k: sorted(v) for k, v in ids.items()},
        "video_ids": sorted(vids),
        "video_spans": spans,
        "counts": counts,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.output}: {len(sha)} images/frames, {sum(map(len, ids.values()))} source ids, {len(vids)} Ego4D videos")


if __name__ == "__main__":
    main()
