"""Fingerprint every evaluation image / frame so distillation corpora can be decontaminated.

Writes ``{"sha1": [...], "dhash": [...], "video_ids": [...], "video_spans": [[uid, start, end], ...]}``:

* sha1 / dHash of every image and stored frame in the given benchmark LMDB directories (all splits);
* Ego4D ``video_uid`` (+ clip seconds) behind every EgoSchema clip (``uid_to_ego4d.json`` from the
  EgoSchema repo). EgoTextVQA-Indoor reuses EgoSchema clips, so its videos are covered too.

Example:
  python make_datasets/corpus/make_eval_blocklist.py --lmdb_dirs $LMDB_ROOT/{vizwiz,textvqa,wearvqa,egoblind,egotextvqa_outdoor_high,egotextvqa_indoor,egoschema,cvbench} \\
      --egoschema_uid_map $RAW_ROOT/egoschema/repo/uid_to_ego4d.json --output $CORPUS_ROOT/eval_blocklist.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from onboard_vlm.data.hashing import dhash, sha1_bytes
from onboard_vlm.data.lmdb_io import LMDBReader, load_meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lmdb_dirs", nargs="+", required=True)
    p.add_argument("--splits", nargs="*", default=None, help="default: every split found (include train: train images are fine to distil on, but this keeps val/test honest)")
    p.add_argument("--egoschema_uid_map", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    sha, dh = set(), set()
    for d in args.lmdb_dirs:
        splits = args.splits or [os.path.basename(x)[: -len(".lmdb")] for x in glob.glob(os.path.join(d, "*.lmdb"))]
        for split in splits:
            if split == "train":  # benchmark train splits are legitimate distillation data
                continue
            reader = LMDBReader(os.path.join(d, f"{split}.lmdb"))
            for k in load_meta(d, split)["keys"]:
                rec = reader.get(k)
                for b in [rec.image] if rec.image else (rec.frames or []):
                    sha.add(sha1_bytes(b))
                    dh.add(f"{dhash(b):016x}")
            print(f"{d}/{split}: total {len(sha)} fingerprints")
    vids, spans = set(), []
    if args.egoschema_uid_map:
        for m in json.load(open(args.egoschema_uid_map)).values():
            vids.add(m["video_uid"])
            spans.append([m["video_uid"], m["starting_sec"], m["ending_sec"]])
    out = {"sha1": sorted(sha), "dhash": sorted(dh), "video_ids": sorted(vids), "video_spans": spans}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.output}: {len(sha)} images/frames, {len(vids)} Ego4D source videos")


if __name__ == "__main__":
    main()
