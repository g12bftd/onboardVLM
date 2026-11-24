"""Finalize the corpus: global de-duplication, per-source metadata, manifest and corpus card.

Runs on the shard indices only (``shard_*.index.npz``: size, dHash, SHA-1 per image), so it takes
minutes and a few GB of RAM for tens of millions of images. For every source directory listed in ``sources.yaml``
(in order; an image seen earlier in the order wins):

1. exact duplicates: identical SHA-1 of the stored bytes, within and across sources;
2. perceptual duplicates: identical 64-bit dHash, within and across sources (catches re-encodes,
   resizes and re-hosted copies; hashes with < 8 or > 56 set bits are exempt since unrelated
   low-texture images share them).

Writes per source ``keep.npz`` (bool mask per shard) and ``dataset.json`` (name, group, licence,
URL, counts), and at the corpus root ``corpus_manifest.json`` and ``CORPUS_CARD.md``.

Nothing here fixes train / val / test: the loader derives splits from a seeded hash of each image
(:func:`onboard_vlm.data.corpus.assign_split`), so split sizes are experiment options. The card
reports the split sizes for the fractions given here (the defaults of ``general_vision.yaml``).

Example:
  python make_datasets/corpus/finalize_corpus.py --corpus_root $CORPUS_ROOT --sources_yaml make_datasets/corpus/sources.yaml
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

from onboard_vlm.data.corpus import KEEP_FILE, SPLITS, assign_split, degenerate_dhash, list_shards, load_index


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


def winners(values: np.ndarray, eligible: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """Owner (source index) of the first occurrence of each entry's value (-1 when ineligible)."""
    idx = np.flatnonzero(eligible)
    w = np.full(len(values), -1, dtype=np.int64)
    if len(idx):
        _, first, inv = np.unique(values[idx], return_index=True, return_inverse=True)
        w[idx] = owner[idx][first][inv]
    return w


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
    p.add_argument("--val_fraction", type=float, default=0.002, help="only for the split sizes reported in the card")
    p.add_argument("--test_fraction", type=float, default=0.002, help="only for the split sizes reported in the card")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--allow_missing", action="store_true", help="skip sources that were not built (default: error)")
    args = p.parse_args()
    t0 = time.time()
    with open(args.sources_yaml) as f:
        spec = yaml.safe_load(f)

    # ---- gather indices in priority order ---------------------------------------------------
    dirs: list[tuple[dict, str]] = []
    for src in spec["sources"]:
        found = [d for d in expand(args.corpus_root, src["dir"]) if list_shards(d)]
        if not found and src.get("optional"):
            print(f"[finalize] optional source {src['dir']} not built (skipped)")
            continue
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
            sha_parts.append(idx["sha1"])
            dh_parts.append(idx["dhash"])
            owner.append(np.full(len(idx["dhash"]), k, dtype=np.int32))
            shard_rows.append((k, name, len(idx["dhash"])))
        print(f"[finalize] indexed {os.path.relpath(d, args.corpus_root)}", flush=True)
    sha, dh, own = np.concatenate(sha_parts), np.concatenate(dh_parts), np.concatenate(owner)
    n_total = len(sha)
    print(f"[finalize] {n_total:,} images in {len(dirs)} source directories")

    # ---- global de-duplication ---------------------------------------------------------------
    keep_sha = first_occurrence(sha)
    dh_ok = keep_sha & ~degenerate_dhash(dh)
    keep_dh = first_occurrence(dh, eligible=dh_ok)
    keep = keep_sha & keep_dh
    win_sha = winners(sha, np.ones(n_total, dtype=bool), own)
    win_dh = winners(dh, dh_ok, own)
    codes = assign_split(dh, sha, args.val_fraction, args.test_fraction, args.split_seed)

    # ---- write masks + per-source metadata ---------------------------------------------------
    report = []
    off, rows_of = 0, {}
    for k, name, n in shard_rows:
        rows_of.setdefault(k, []).append((name, off, n))
        off += n
    for k, (src, d) in enumerate(dirs):
        np.savez(os.path.join(d, KEEP_FILE), **{name: keep[o : o + n] for name, o, n in rows_of[k]})
        m = own == k
        rm_sha, rm_dh = m & ~keep_sha, m & keep_sha & ~keep_dh
        entry = {
            "source": src["dir"], "directory": os.path.relpath(d, args.corpus_root), "group": src.get("group"),
            "name": src.get("name"), "license": src.get("license"), "url": src.get("url"),
            "built": int(m.sum()), "kept": int((keep & m).sum()),
            "dup_exact_within": int((rm_sha & (win_sha == k)).sum()), "dup_exact_cross": int((rm_sha & (win_sha != k)).sum()),
            "dup_dhash_within": int((rm_dh & (win_dh == k)).sum()), "dup_dhash_cross": int((rm_dh & (win_dh != k)).sum()),
            "splits": {sp: int((keep & m & (codes == c)).sum()) for c, sp in enumerate(SPLITS)},
            "builder": builder_stats(d),
        }
        with open(os.path.join(d, "dataset.json"), "w") as f:
            json.dump({**entry, "split_definition": {"val_fraction": args.val_fraction, "test_fraction": args.test_fraction, "split_seed": args.split_seed}}, f, indent=1)
        report.append(entry)
        print(f"[finalize] {entry['directory']}: built {entry['built']:,} -> kept {entry['kept']:,} ({entry['splits']})")

    digest = hashlib.sha256(np.sort(sha[keep]).tobytes()).hexdigest()
    manifest = {
        "corpus": spec.get("name", "corpus"), "version": spec.get("version"), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_built": n_total, "total_kept": int(keep.sum()),
        "dedup": {"exact_sha1_removed": int((~keep_sha).sum()), "dhash_removed": int((keep_sha & ~keep_dh).sum())},
        "split_definition": {"val_fraction": args.val_fraction, "test_fraction": args.test_fraction, "split_seed": args.split_seed},
        "splits": {sp: int((keep & (codes == c)).sum()) for c, sp in enumerate(SPLITS)},
        "kept_sha1_digest": digest, "sources": report,
    }
    with open(os.path.join(args.corpus_root, "corpus_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    write_card(os.path.join(args.corpus_root, "CORPUS_CARD.md"), manifest)
    print(f"[finalize] kept {manifest['total_kept']:,} of {n_total:,} | splits {manifest['splits']} | digest {digest[:16]} | {time.time() - t0:.0f}s")


def write_card(path: str, m: dict) -> None:
    sd = m["split_definition"]
    lines = [
        f"# {m['corpus']} (v{m['version']}) - corpus card",
        "",
        f"Generated by finalize_corpus.py on {m['created']}. SHA-256 over the sorted SHA-1 of all kept images: `{m['kept_sha1_digest']}`.",
        f"Split sizes are for val_fraction={sd['val_fraction']}, test_fraction={sd['test_fraction']}, split_seed={sd['split_seed']}"
        " (splits are assigned in the loader; other values give other sizes without rebuilding).",
        "",
        "| directory | group | built | benchmark dups removed at build | exact dups (within / cross) | dHash dups (within / cross) | kept | train / val / test | licence |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in m["sources"]:
        b, sp = s["builder"], s["splits"]
        lines.append(
            f"| {s['directory']} | {s['group']} | {s['built']:,} | {b.get('benchmark_dup', 0):,} | {s['dup_exact_within']:,} / {s['dup_exact_cross']:,} "
            f"| {s['dup_dhash_within']:,} / {s['dup_dhash_cross']:,} | {s['kept']:,} | {sp['train']:,} / {sp['val']:,} / {sp['test']:,} | {s['license']} |"
        )
    sp = m["splits"]
    lines += [
        f"| **total** | | {m['total_built']:,} | | {m['dedup']['exact_sha1_removed']:,} | {m['dedup']['dhash_removed']:,} | **{m['total_kept']:,}** "
        f"| {sp['train']:,} / {sp['val']:,} / {sp['test']:,} | |",
        "",
        "Build-time filters per source (decode errors, short side, aspect ratio, near-uniform, benchmark duplicates) are in",
        "`corpus_manifest.json` (`sources[*].builder`) and in each source's `dataset.json`.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
