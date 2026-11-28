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
E3_MINI = {"textvqa", "cvbench"}  # the per-checkpoint downstream benchmarks; the others are an application check only
missing: list[str] = []


def ok(flag: bool, what: str, detail: str = "", optional: bool = False) -> None:
    print(f"  {'OK ' if flag else '-- '} {what:<42} {detail}")
    if not flag and not optional:
        missing.append(what)


def step_done(d: str, root: str) -> bool:
    """A source is built when the build step that wrote it finished: ``.done`` in its directory or in an enclosing one
    up to the corpus root (``text_rich/.done`` covers ``text_rich/pixmo_docs`` and ``text_rich/cauldron/<subset>``)."""
    root, d = os.path.abspath(root), os.path.abspath(d)
    while d.startswith(root) and d != root:
        if os.path.isfile(os.path.join(d, ".done")):
            return True
        d = os.path.dirname(d)
    return False


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
        ok(bool(counts), b, ", ".join(f"{k} {v:,}" for k, v in counts.items()) + ("" if b in E3_MINI else "  (optional: application check)"),
           optional=b not in E3_MINI)

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
    ok(n_in1k == 1024, "imagenet1k train tars (probes + corpus)", f"{n_in1k} / 1024 tars")
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
        done = bool(dirs) and all(step_done(d, corpus) for d in dirs)
        finalized = bool(dirs) and all(os.path.isfile(os.path.join(d, "keep.npz")) for d in dirs)
        ok(done and finalized, src["dir"], f"{n:,} images | built {done} | finalized {finalized}" + ("  (optional)" if src.get("optional") else ""),
           optional=bool(src.get("optional")))
    man = os.path.join(corpus, "corpus_manifest.json")
    if os.path.isfile(man):
        m = json.load(open(man))
        if "total_built" in m:
            print(f"  manifest: {m['total_built']:,} built -> {m['total_kept']:,} kept | splits {m.get('splits')} | see {corpus}/CORPUS_CARD.md")

    readiness(lmdb_root, evals, corpus, raw, reg, spec)
    print(f"\n{len(missing)} item(s) missing" + (": " + ", ".join(missing) if missing else ""))
    if args.strict and missing:
        sys.exit(1)


def readiness(lmdb_root: str, evals: str, corpus: str, raw: str, reg: dict, spec: list) -> None:
    """Is the data right for the first experiments (docs/methods.md): decontamination complete and applied, the training
    corpus finalized, and every evaluation input in place?"""
    import hashlib
    import subprocess

    print("\nReadiness for the first experiments (references, identity, lr, grid)")
    issues: list[str] = []
    built_bench = sorted(b for b in BENCHMARKS if glob.glob(os.path.join(lmdb_root, b, "*.meta.json")))
    built_sets = sorted(n for n, sp in reg.items() if "blocklist" in sp["roles"] and os.path.isfile(os.path.join(evals, n, ".done")))
    bl_path = os.path.join(corpus, "eval_blocklist.json")
    bl = json.load(open(bl_path)) if os.path.isfile(bl_path) else None
    if bl is None:
        issues.append("no blocklist: run the blocklist step, then corpus_finalize")
        ok(False, "decontamination: blocklist", "missing")
    else:
        uncovered = [b for b in built_bench if b not in (bl.get("benchmarks") or [])] + [n for n in built_sets if n not in (bl.get("eval_sets") or [])]
        ok(not uncovered, "decontamination: blocklist covers what is built",
           f"{len(bl.get('benchmarks') or [])} benchmarks + {len(bl.get('eval_sets') or [])} eval sets" + (f"; NOT covered: {', '.join(uncovered)}" if uncovered else ""))
        if uncovered:
            issues.append("rebuild the blocklist and re-run corpus_finalize (new evaluation data since the last build)")
    not_built = [n for n, sp in reg.items() if "blocklist" in sp["roles"] and n not in built_sets]
    ok(not not_built, "decontamination: every evaluation set built", "" if not not_built else
       f"not built (not blocklisted): {', '.join(not_built)} -- fine for preliminary runs, required for paper numbers", optional=True)
    man_path = os.path.join(corpus, "corpus_manifest.json")
    man = json.load(open(man_path)) if os.path.isfile(man_path) else None
    if man is None:
        issues.append("corpus not finalized: run corpus_finalize")
        ok(False, "corpus finalized", "no corpus_manifest.json")
    else:
        dec = man.get("decontamination") or {}
        current = bl is not None and dec.get("sha256") == hashlib.sha256(open(bl_path, "rb").read()).hexdigest()
        ok(current, "corpus finalized with the current blocklist", f"{dec.get('removed', 0):,} images removed" if current else
           "the blocklist changed after finalize: re-run corpus_finalize")
        if not current:
            issues.append("re-run corpus_finalize (the blocklist changed after it)")
        if man.get("preliminary"):
            print(f"  ..  preliminary corpus: sources not built: {', '.join(man.get('missing_sources') or [])} (train with allow_missing_sources=true)")
    lite = {"cc12m", "imagenet1k", "openimages", "sa1b", "text_rich/pixmo_docs", "text_rich/synthdog_en", "text_rich/cauldron/*"}
    for src in spec:
        if src["dir"] not in lite:
            continue
        dirs = sorted(d for d in glob.glob(os.path.join(corpus, src["dir"])) if os.path.isdir(d))
        fin = bool(dirs) and all(os.path.isfile(os.path.join(d, "keep.npz")) for d in dirs)
        if not fin and src["dir"] != "sa1b":
            issues.append(f"training source {src['dir']} not built / finalized")
    ok(all(os.path.isfile(os.path.join(evals, n, ".done")) for n, sp in reg.items() if "fidelity" in sp["roles"] and n in
           {"coco_val2017", "ade20k_val", "imagenet1k_val", "docvqa_val", "infovqa_val", "chartqa_test", "realworldqa"}),
       "E1: the 7 held-out fidelity sets", "", optional=True)
    e2 = len(glob.glob(os.path.join(raw, "imagenet1k", "imagenet1k-train-*.tar"))) > 0 and len(glob.glob(os.path.join(evals, "nyuv2", "test", "*[0-9].png"))) > 0
    ok(e2, "E2: ImageNet-1k train tars + NYUv2", "", optional=True)
    e3 = all(b in built_bench for b in E3_MINI)
    ok(e3, "E3 per checkpoint: TextVQA + CV-Bench LMDBs", "", optional=True)
    from onboard_vlm.utils.envs import conda_env_python

    env = os.environ.get("ONBOARD_VLM_LMMS_ENV", "onboardvlm-lmms")
    py = os.environ.get("ONBOARD_VLM_LMMS_PYTHON") or conda_env_python(env) or ""
    has_lmms = bool(py) and subprocess.run([py, "-c", "import lmms_eval"], capture_output=True).returncode == 0
    ok(has_lmms, "general suite: lmms-eval environment", py if has_lmms else
       f"conda env '{env}' {'cannot import lmms_eval' if py else 'not found'} (bash scripts/setup_envs.sh; or run with --evals e1,e2,e3)",
       optional=True)
    for x, what in ((e2, "E2 inputs"), (e3, "E3 benchmark LMDBs"), (has_lmms, "the lmms-eval env")):
        if not x:
            issues.append(f"evaluation: {what} missing")
    print("  => " + ("READY for training and evaluation" if not issues else "NOT READY:\n     - " + "\n     - ".join(issues)))


if __name__ == "__main__":
    main()
