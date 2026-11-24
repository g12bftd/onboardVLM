"""What is in place? Teacher weights, benchmark LMDBs, evaluation image sets, probe data and the corpus.

    source scripts/paths.sh && python scripts/check_data.py [--strict]

Reads only metadata (meta.json / index.npz / markers), so it runs in seconds. ``--strict`` exits
non-zero if anything is missing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BENCHMARKS = ["vizwiz", "textvqa", "wearvqa", "egoblind", "egotextvqa_outdoor_high", "egotextvqa_outdoor_low", "egotextvqa_indoor", "egoschema", "cvbench"]
missing: list[str] = []


def ok(flag: bool, what: str, detail: str = "", optional: bool = False) -> None:
    print(f"  {'OK ' if flag else '-- '} {what:<42} {detail}")
    if not flag and not optional:
        missing.append(what)


def corpus_count(d: str) -> int:
    return sum(len(np.load(p)["sizes"]) for p in glob.glob(os.path.join(d, "shard_*.index.npz")))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    env = {k: os.environ.get(k) for k in ("RAW_ROOT", "LMDB_ROOT", "CORPUS_ROOT", "EVAL_ROOT", "HF_HOME")}
    if not all(env.values()):
        sys.exit("source scripts/paths.sh first")
    raw, lmdb_root, corpus, evals = env["RAW_ROOT"], env["LMDB_ROOT"], env["CORPUS_ROOT"], env["EVAL_ROOT"]

    print("Teacher weights (pinned revisions in the HF cache)")
    from huggingface_hub import try_to_load_from_cache

    from onboard_vlm.teachers.gemma4 import SUPPORTED_TEACHERS

    for model_id, rev in SUPPORTED_TEACHERS.items():
        hit = try_to_load_from_cache(model_id, "config.json", revision=rev)
        base = not model_id.endswith("-it")
        ok(isinstance(hit, str), model_id, rev[:7] + ("  (optional: base-model ablations)" if base else ""), optional=base)

    print(f"\nBenchmarks: processed {lmdb_root}")
    for b in BENCHMARKS:
        metas = sorted(glob.glob(os.path.join(lmdb_root, b, "*.meta.json")))
        counts = {os.path.basename(m)[: -len(".meta.json")]: json.load(open(m))["num_samples"] for m in metas}
        ok(bool(counts), b, ", ".join(f"{k} {v:,}" for k, v in counts.items()))

    print(f"\nEvaluation image sets: processed {evals}")
    reg = yaml.safe_load(open(os.path.join(ROOT, "make_datasets", "eval_sets", "eval_sets.yaml")))["sets"]
    for name, spec in reg.items():
        d = os.path.join(evals, name)
        ok(os.path.isfile(os.path.join(d, ".done")), name, f"{corpus_count(d):,} images | roles {spec['roles']}" if os.path.isdir(d) else "")
    for split in ("train", "test"):
        n = len(glob.glob(os.path.join(evals, "nyuv2", split, "*[0-9].png")))
        ok(n > 0, f"nyuv2/{split}", f"{n} images (expected {795 if split == 'train' else 654})")

    print(f"\nProbe sources: raw {raw}")
    n_in1k = len(glob.glob(os.path.join(raw, "imagenet1k", "imagenet1k-train-*.tar")))
    ok(n_in1k > 0, "imagenet1k train tars", f"{n_in1k} tars")
    n_val = len(glob.glob(os.path.join(raw, "eval_sets", "imagenet1k_val", "imagenet1k-validation-*.tar")))
    ok(n_val > 0, "imagenet1k val tars", f"{n_val} tars")
    ade = os.path.join(raw, "eval_sets", "ade20k_val", "ADEChallengeData2016")
    ok(os.path.isdir(os.path.join(ade, "annotations", "training")), "ADE20K images + annotations", ade)

    print(f"\nCorpus: processed {corpus}")
    ok(os.path.isfile(os.path.join(corpus, "eval_blocklist.json")), "eval_blocklist.json")
    spec = yaml.safe_load(open(os.path.join(ROOT, "make_datasets", "corpus", "sources.yaml")))["sources"]
    for src in spec:
        dirs = sorted(d for d in glob.glob(os.path.join(corpus, src["dir"])) if os.path.isdir(d))
        n = sum(corpus_count(d) for d in dirs)
        done = bool(dirs) and all(os.path.isfile(os.path.join(d, ".done")) or os.path.isfile(os.path.join(os.path.dirname(d), ".done")) for d in dirs)
        finalized = bool(dirs) and all(os.path.isfile(os.path.join(d, "keep.npz")) for d in dirs)
        ok(done and finalized, src["dir"], f"{n:,} images | built {done} | finalized {finalized}")
    man = os.path.join(corpus, "corpus_manifest.json")
    if os.path.isfile(man):
        m = json.load(open(man))
        print(f"  manifest: {m['total_built']:,} built -> {m['total_kept']:,} kept | splits {m['splits']} | see {corpus}/CORPUS_CARD.md")

    print(f"\n{len(missing)} item(s) missing" + (": " + ", ".join(missing) if missing else ""))
    if args.strict and missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
