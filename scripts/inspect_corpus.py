"""Audit what the corpus builders wrote, source by source, before training on it.

    source scripts/paths.sh && python scripts/inspect_corpus.py [--samples 300] [--out $ONBOARD_VLM_OUT/corpus_inspect]

For every source of make_datasets/corpus/sources.yaml that has shards:

* completeness: the build step finished (``.done``); every builder worker that wrote shards also wrote its
  statistics (a worker that died leaves none); images written by the builders == rows in the shard indices;
* coverage: images *seen* against the size of the published release (a shortfall means input files were missed);
* filtering: what was dropped and why (decode error, short side < 224, aspect > 4, near-uniform, benchmark
  duplicate) as a share of what was seen;
* geometry, from the index: short / long side and aspect-ratio percentiles, share stored at the 1536-px cap;
* integrity, on ``--samples`` random images per source: the stored bytes decode, their size is the indexed size,
  and their dHash is within a few bits of the indexed one (fingerprints belong to the right image);
* a contact sheet of random images per source (``<out>/sheets/<source>.jpg``): look at them.

Writes ``<out>/report.md`` and prints the summary; exits non-zero if a check fails. Reads only; nothing is changed.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# images in the published release that the build reads (docs/corpus.md); None = not checked (capped subsets)
EXPECTED = {"cc12m": 10_968_539, "imagenet1k": 1_281_167, "openimages": 1_743_042, "text_rich/pixmo_docs": 255_261,
            "text_rich/synthdog_en": 500_000}
DROPS = ("decode_error", "too_small", "bad_aspect", "near_uniform", "benchmark_dup")


def step_done(d: str, root: str) -> bool:
    root, d = os.path.abspath(root), os.path.abspath(d)
    while d.startswith(root) and d != root:
        if os.path.isfile(os.path.join(d, ".done")):
            return True
        d = os.path.dirname(d)
    return False


def expand(root: str, pattern: str) -> list[str]:
    full = os.path.join(root, pattern)
    return sorted(p for p in glob.glob(full) if os.path.isdir(p)) if glob.has_magic(full) else ([full] if os.path.isdir(full) else [])


def pct(a: np.ndarray, qs=(1, 50, 99), fmt: str = ".0f") -> str:
    return " / ".join(f"{np.percentile(a, q):{fmt}}" for q in qs) if len(a) else "-"


def audit_dir(d: str, rng: np.random.Generator, samples: int, sheet: int):
    from onboard_vlm.data.corpus import corpus_key, list_shards, load_index, popcount64

    shards = list_shards(d)
    stats_files = sorted(glob.glob(os.path.join(d, "stats_*.json")))
    stats = {k: 0 for k in ("seen", "written", *DROPS)}
    for p in stats_files:
        for k, v in json.load(open(p)).items():
            stats[k] = stats.get(k, 0) + int(v)
    workers_with_shards = {m.group(1) for s in shards if (m := re.match(r"shard_(\d+)_", s))}
    workers_with_stats = {m.group(1) for p in stats_files if (m := re.match(r"stats_(\d+)\.json", os.path.basename(p)))}
    sizes, dh, owners = [], [], []
    for k, s in enumerate(shards):
        idx = load_index(d, s)
        sizes.append(idx["sizes"])
        dh.append(np.asarray(idx["dhash"], dtype=np.uint64))
        owners.append(np.stack([np.full(len(idx["sizes"]), k), np.arange(len(idx["sizes"]))], 1))
    sizes = np.concatenate(sizes) if sizes else np.zeros((0, 2), np.int64)
    dh = np.concatenate(dh) if dh else np.zeros(0, np.uint64)
    owners = np.concatenate(owners) if owners else np.zeros((0, 2), np.int64)
    out = {"dir": d, "shards": len(shards), "stats": stats, "indexed": len(sizes),
           "missing_worker_stats": sorted(workers_with_shards - workers_with_stats), "sizes": sizes}
    # integrity on random images
    from PIL import Image

    from onboard_vlm.data.hashing import dhash
    from onboard_vlm.data.lmdb_io import VQARecord, open_readonly

    pick = rng.choice(len(sizes), size=min(samples, len(sizes)), replace=False) if len(sizes) else np.zeros(0, int)
    bad_decode, bad_size, far_hash, bad_any, thumbs = 0, 0, 0, 0, []
    for n, i in enumerate(pick):
        k, j = owners[i]
        with open_readonly(os.path.join(d, f"{shards[k]}.lmdb")).begin(write=False) as txn:
            b = txn.get(corpus_key(int(j)))
        try:
            img = Image.open(io.BytesIO(VQARecord.loads(b).image))
            img.load()
        except Exception:
            bad_decode += 1
            bad_any += 1
            continue
        size_ok = (img.height, img.width) == tuple(int(x) for x in sizes[i])
        hash_ok = int(popcount64(np.array([np.uint64(dhash(img)) ^ dh[i]]))[0]) <= 10  # re-encoding moves a few bits, not 10+
        bad_size += not size_ok
        far_hash += not hash_ok
        bad_any += not (size_ok and hash_ok)
        if n < sheet:
            t = img.convert("RGB")
            t.thumbnail((160, 160))
            thumbs.append(t)
    out.update(checked=len(pick), bad_decode=bad_decode, bad_size=bad_size, far_hash=far_hash, bad_any=bad_any, thumbs=thumbs)
    return out


def contact_sheet(thumbs, path: str, cols: int = 8) -> None:
    from PIL import Image

    if not thumbs:
        return
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 164, rows * 164), (255, 255, 255))
    for n, t in enumerate(thumbs):
        x, y = (n % cols) * 164 + (164 - t.width) // 2, (n // cols) * 164 + (164 - t.height) // 2
        sheet.paste(t, (x, y))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path, quality=85)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus_root", default=os.environ.get("CORPUS_ROOT"))
    p.add_argument("--out", default=os.path.join(os.environ.get("ONBOARD_VLM_OUT", "outputs"), "corpus_inspect"))
    p.add_argument("--samples", type=int, default=300, help="random images per source decoded and checked")
    p.add_argument("--sheet", type=int, default=48, help="images per contact sheet")
    p.add_argument("--max_side", type=int, default=1536, help="the builders' --max_side")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if not a.corpus_root:
        sys.exit("source scripts/paths.sh first (or pass --corpus_root)")
    spec = yaml.safe_load(open(os.path.join(ROOT, "make_datasets", "corpus", "sources.yaml")))["sources"]
    rng = np.random.default_rng(a.seed)
    lines, problems = ["# Corpus audit", "", f"corpus root: `{a.corpus_root}`", ""], []
    lines += ["| source | done | seen | expected | kept | drop: decode / small / aspect / uniform / benchmark | short side p1/p50/p99 | "
              "long side p1/p50/p99 | aspect p1/p50/p99 | at cap | integrity (decode, size, hash) |", "|" + "---|" * 11]
    print(f"[inspect_corpus] {a.corpus_root}")
    audited = 0
    for src in spec:
        dirs = [d for d in expand(a.corpus_root, src["dir"]) if glob.glob(os.path.join(d, "shard_*.index.npz"))]
        if not dirs:
            continue
        name = src["dir"].replace("/*", "")
        audited += 1
        res = [audit_dir(d, rng, max(1, a.samples // len(dirs)), max(1, a.sheet // len(dirs))) for d in dirs]
        stats = {k: sum(r["stats"].get(k, 0) for r in res) for k in ("seen", "written", *DROPS)}
        indexed = sum(r["indexed"] for r in res)
        sizes = np.concatenate([r["sizes"] for r in res])
        short, long_ = sizes.min(1), sizes.max(1)
        aspect = long_ / np.maximum(short, 1)
        done = all(step_done(d, a.corpus_root) for d in dirs)
        exp = EXPECTED.get(src["dir"])
        checked = sum(r["checked"] for r in res)
        integ = (sum(r["bad_decode"] for r in res), sum(r["bad_size"] for r in res), sum(r["far_hash"] for r in res))
        bad = sum(r["bad_any"] for r in res)
        seen = max(stats["seen"], 1)
        drops = " / ".join(f"{100 * stats[k] / seen:.1f}%" for k in DROPS)
        lines.append(f"| {name} | {'yes' if done else 'NO'} | {stats['seen']:,} | {f'{exp:,}' if exp else '-'} | {indexed:,} | {drops} | "
                     f"{pct(short)} | {pct(long_)} | {pct(aspect, fmt=".2f")} | "
                     f"{100 * np.mean(long_ == a.max_side):.0f}% | {checked - bad}/{checked} ok |")
        # checks
        if not done:
            problems.append(f"{name}: build step not finished (no .done)")
        if stats["written"] != indexed:
            problems.append(f"{name}: builders wrote {stats['written']:,} images but the indices hold {indexed:,}")
        miss = [w for r in res for w in r["missing_worker_stats"]]
        if miss:
            problems.append(f"{name}: workers {sorted(set(miss))} wrote shards but no statistics (crashed?)")
        if exp and abs(stats["seen"] - exp) > 0.02 * exp:
            problems.append(f"{name}: {stats['seen']:,} images seen, release has {exp:,} ({100 * (stats['seen'] / exp - 1):+.1f}%)")
        if bad:
            problems.append(f"{name}: {bad} of {checked} sampled images fail integrity (decode {integ[0]}, size {integ[1]}, hash {integ[2]})")
        if len(short) and short.min() < 224:
            problems.append(f"{name}: stored images with short side < 224 (min {short.min()})")
        contact_sheet([t for r in res for t in r["thumbs"]], os.path.join(a.out, "sheets", name.replace("/", "_") + ".jpg"))
        if len(dirs) > 1:  # subsets (The Cauldron)
            for r in res:
                lines.append(f"| &nbsp;&nbsp;{os.path.basename(r['dir'])} | | {r['stats']['seen']:,} | | {r['indexed']:,} | | | | | | |")
        print(f"  {name:<24} seen {stats['seen']:>11,}  kept {indexed:>11,}  ({100 * indexed / seen:5.1f}%)  "
              f"drops {drops}  integrity {checked - bad}/{checked}")
    if not audited:
        problems.append(f"no built source under {a.corpus_root} (shard_*.index.npz): nothing was checked")
    lines += ["", "Drops are shares of the images seen. `at cap`: stored with the long side at the --max_side cap (re-encoded).",
              "Contact sheets: `sheets/<source>.jpg` (random images; look at them).", ""]
    lines += ["## Checks", ""] + ([f"- FAIL: {x}" for x in problems] or ["- all checks passed"])
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(["", *([f"FAIL: {x}" for x in problems] or ["all checks passed"]),
                     f"report: {a.out}/report.md | contact sheets: {a.out}/sheets/"]))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
