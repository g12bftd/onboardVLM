"""Download and store the evaluation image sets of ``eval_sets.yaml`` in the corpus format.

    python make_datasets/eval_sets/build_eval_sets.py --raw_root $RAW_ROOT/eval_sets --eval_root $EVAL_ROOT [--only coco_val2017 pope]

Images are stored unfiltered and with their original bytes (``--max_side 0``), so SHA-1 fingerprints
equal the benchmark files'. Gated sets (ImageNet-1k) need ``hf auth login`` after accepting the terms.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BUILDER = os.path.join(HERE, "..", "corpus", "make_image_corpus.py")
KEEP_ALL = ["--min_side", "1", "--max_aspect", "1000", "--min_std", "0", "--max_side", "0"]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_root", required=True)
    p.add_argument("--eval_root", required=True)
    p.add_argument("--registry", default=os.path.join(HERE, "eval_sets.yaml"))
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    sets = yaml.safe_load(open(args.registry))["sets"]
    for name, spec in sets.items():
        if args.only and name not in args.only:
            continue
        raw, out = os.path.join(args.raw_root, name), os.path.join(args.eval_root, name)
        os.makedirs(raw, exist_ok=True)
        if "url" in spec:
            archive = os.path.join(raw, os.path.basename(spec["url"]))
            if not os.path.isfile(archive):
                run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", archive, spec["url"]])
            if not os.path.isdir(os.path.join(raw, spec["folder"])):
                run(["unzip", "-q", "-o", archive, "-d", raw])
            source = ["--source", "folder", "--input_path", os.path.join(raw, spec["folder"])]
        else:
            inc = [a for pat in spec["include"] for a in ("--include", pat)]
            run(["hf", "download", spec["hf"], "--repo-type", "dataset", "--local-dir", raw, *inc])
            if "tars" in spec:
                source = ["--source", "webdataset", "--input_path", os.path.join(raw, spec["tars"])]
            else:
                source = ["--source", "parquet", "--input_path", os.path.join(raw, spec["parquet"]), "--image_column", spec["column"]]
        procs = [subprocess.Popen([sys.executable, BUILDER, *source, "--name", name, "--output_path", out, *KEEP_ALL,
                                   "--worker_id", str(i), "--num_workers", str(args.workers)]) for i in range(args.workers)]
        if any(pr.wait() for pr in procs):
            raise RuntimeError(f"building {name} failed")
        print(f"[eval_sets] {name} -> {out}", flush=True)


if __name__ == "__main__":
    main()
