"""Finalize the corpus: global de-duplication, held-out split, manifest and corpus card.

Runs on the shard indices only (``shard_*.index.npz``: size, dHash, SHA-1 per image), so it takes
minutes and a few GB of RAM for ~30M images. For every source directory listed in ``sources.yaml``
(in order; an image seen earlier in the order wins):

1. exact duplicates: identical SHA-1 of the stored bytes, within and across sources;
2. perceptual duplicates: identical 64-bit dHash, within and across sources (catches re-encodes,
   resizes and re-hosted copies; hashes with < 8 or > 56 set bits are exempt since unrelated
   low-texture images share them);
3. held-out split: ``--holdout`` images per source directory (seeded) are removed from training and
   kept for in-distribution validation (``ImageCorpusDataset(split="holdout")``).

Writes per source ``keep.npz`` and ``holdout.npz`` (bool mask per shard), and at the corpus root
``corpus_manifest.json`` (counts, filter statistics, licences, fingerprint digest) and ``CORPUS_CARD.md``.
Benchmark decontamination already happened at build time (``--blocklist``); its counts are reported.

Example:
  python make_datasets/corpus/finalize_corpus.py --corpus_root $CORPUS_ROOT \\
      --sources_yaml make_datasets/corpus/sources.yaml --holdout 2000
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import yaml

from onboard_vlm.data.corpus import list_shards, load_index


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x).astype(np.int64)
    c = np.zeros(x.shape, dtype=np.int64)
    for s in range(0, 64, 8):
        c += np.unpackbits(((x >> np.uint64(s)) & np.uint64(0xFF)).astype(np.uint8)[..., None], axis=-1).sum(-1, dtype=np.int64)
    return c


def first_occurrence(values: np.ndarray, eligible: np.ndarray | None = None) -> np.ndarray:
    """Bool mask: True for the first occurrence of each value (ineligible entries are always True)."""
    keep = np.ones(len(values), dtype=bool)
    idx = np.arange(len(values)) if eligible is None else np.flatnonzero(eligible)
    if len(idx) == 0:
        return keep
    _, first = np.unique(values[idx], return_index=True)  # stable: index of the first occurrence
    sub = np.zeros(len(idx), dtype=bool)
    sub[first] = True
    keep[idx] = sub
    return keep


def expand(corpus_root: str, pattern: str) -> list[str]:
    full = os.path.join(corpus_root, pattern)
    return sorted(p for p in glob.glob(full) if os.path.isdir(p)) if glob.has_magic(full) else [full]


def builder_stats(d: str) -> dict[str, int]:
    tot: dict[str, int] = {}
    for p in glob.glob(os.path.join(d, "stats_*.json")):
        with open(p) as f:
            for k, v in json.load(f).items():
                tot[k] = tot.get(k, 0) + int(v)
    return tot


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus_root", required=True)
    p.add_argument("--sources_yaml", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources.yaml"))
    p.add_argument("--holdout", type=int, default=2000, help="held-out images per source directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_missing", action="store_true", help="skip sources that were not built (default: error)")
    args = p.parse_args()
    t0 = time.time()
    with open(args.sources_yaml) as f:
        spec = yaml.safe_load(f)

    # ---- gather indices in priority order ---------------------------------------------------
    dirs: list[tuple[dict, str]] = []
    for src in spec["sources"]:
        found = [d for d in expand(args.corpus_root, src["dir"]) if list_shards(d)]
        if not found:
            if args.allow_missing:
                print(f"[finalize] missing source {src['dir']} (skipped)")
                continue
            raise FileNotFoundError(f"source {src['dir']} has no shards under {args.corpus_root}; build it or pass --allow_missing")
        dirs.extend((src, d) for d in found)
    sha_parts, dh_parts, owner, shard_rows = [], [], [], []
    for k, (_, d) in enumerate(dirs):
        for name in list_shards(d):
            idx = load_index(d, name)
            n = len(idx["dhash"])
            sha_parts.append(idx["sha1"])
            dh_parts.append(idx["dhash"])
            owner.append(np.full(n, k, dtype=np.int32))
            shard_rows.append((k, name, n))
        print(f"[finalize] indexed {os.path.relpath(d, args.corpus_root)}", flush=True)
    sha = np.concatenate(sha_parts)
    dh = np.concatenate(dh_parts)
    own = np.concatenate(owner)
    n_total = len(sha)
    print(f"[finalize] {n_total:,} images in {len(dirs)} source directories")

    # ---- global de-duplication ---------------------------------------------------------------
    keep_sha = first_occurrence(sha)
    pc = popcount64(dh)
    keep_dh = first_occurrence(dh, eligible=keep_sha & (pc >= 8) & (pc <= 56))
    keep = keep_sha & keep_dh

    # which source "won" each removed image (to separate within- from cross-source duplicates)
    def winners(values: np.ndarray, eligible: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(eligible)
        _, first, inv = np.unique(values[idx], return_index=True, return_inverse=True)
        w = np.full(len(values), -1, dtype=np.int64)
        w[idx] = own[idx][first][inv]
        return w

    win_sha = winners(sha, np.ones(n_total, dtype=bool))
    win_dh = winners(dh, keep_sha & (pc >= 8) & (pc <= 56))

    # ---- held-out split (per source directory, seeded) ---------------------------------------
    rng = np.random.default_rng(args.seed)
    holdout = np.zeros(n_total, dtype=bool)
    for k in range(len(dirs)):
        cand = np.flatnonzero(keep & (own == k))
        if len(cand):
            holdout[rng.choice(cand, size=min(args.holdout, len(cand) // 10), replace=False)] = True
    train = keep & ~holdout

    # ---- write masks + per-source report -----------------------------------------------------
    report = []
    off = 0
    per_dir_rows: dict[int, list[tuple[str, int, int]]] = {}
    for k, name, n in shard_rows:
        per_dir_rows.setdefault(k, []).append((name, off, n))
        off += n
    for k, (src, d) in enumerate(dirs):
        rows = per_dir_rows[k]
        np.savez(os.path.join(d, "keep.npz"), **{name: train[o : o + n] for name, o, n in rows})
        np.savez(os.path.join(d, "holdout.npz"), **{name: holdout[o : o + n] for name, o, n in rows})
        m = own == k
        rm_sha, rm_dh = m & ~keep_sha, m & keep_sha & ~keep_dh
        report.append({
            "source": src["dir"], "directory": os.path.relpath(d, args.corpus_root), "group": src.get("group"),
            "name": src.get("name"), "license": src.get("license"), "url": src.get("url"),
            "built": int(m.sum()), "dup_exact_within": int((rm_sha & (win_sha == k)).sum()),
            "dup_exact_cross": int((rm_sha & (win_sha != k)).sum()), "dup_dhash_within": int((rm_dh & (win_dh == k)).sum()),
            "dup_dhash_cross": int((rm_dh & (win_dh != k)).sum()), "train": int((train & m).sum()), "holdout": int((holdout & m).sum()),
            "builder": builder_stats(d),
        })
        print(f"[finalize] {report[-1]['directory']}: built {report[-1]['built']:,} -> train {report[-1]['train']:,}, holdout {report[-1]['holdout']:,}")

    digest = hashlib.sha256(np.sort(sha[train]).tobytes()).hexdigest()
    manifest = {
        "corpus": spec.get("name", "corpus"), "version": spec.get("version"), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_built": n_total, "total_train": int(train.sum()), "total_holdout": int(holdout.sum()),
        "dedup": {"exact_sha1_removed": int((~keep_sha).sum()), "dhash_removed": int((keep_sha & ~keep_dh).sum())},
        "train_sha1_digest": digest, "seed": args.seed, "sources": report,
    }
    with open(os.path.join(args.corpus_root, "corpus_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    write_card(os.path.join(args.corpus_root, "CORPUS_CARD.md"), manifest)
    print(f"[finalize] train {manifest['total_train']:,} | holdout {manifest['total_holdout']:,} | digest {digest[:16]} | {time.time() - t0:.0f}s")


def write_card(path: str, m: dict) -> None:
    lines = [
        f"# {m['corpus']} (v{m['version']}) - corpus card",
        "",
        f"Generated by finalize_corpus.py on {m['created']}. Train-set SHA-1 digest: `{m['train_sha1_digest']}`.",
        "",
        "| directory | group | built | benchmark dups removed at build | exact dups (within / cross) | dHash dups (within / cross) | train | holdout | licence |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in m["sources"]:
        b = s["builder"]
        lines.append(
            f"| {s['directory']} | {s['group']} | {s['built']:,} | {b.get('benchmark_dup', 0):,} | {s['dup_exact_within']:,} / {s['dup_exact_cross']:,} "
            f"| {s['dup_dhash_within']:,} / {s['dup_dhash_cross']:,} | {s['train']:,} | {s['holdout']:,} | {s['license']} |"
        )
    lines += [
        f"| **total** | | {m['total_built']:,} | | {m['dedup']['exact_sha1_removed']:,} | {m['dedup']['dhash_removed']:,} | **{m['total_train']:,}** | {m['total_holdout']:,} | |",
        "",
        "Build-time filters per source (decode errors, short side, aspect ratio, near-uniform, benchmark duplicates) are in",
        "`corpus_manifest.json` (`sources[*].builder`).",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
