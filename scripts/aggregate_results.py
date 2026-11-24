"""Paper tables with uncertainty: per-item bootstrap CIs, seed mean +- std, paired tests vs a reference.

    python scripts/aggregate_results.py --out outputs/tables/main --reference teacher \\
        --eval teacher=outputs/eval/teacher \\
        --eval g4vit_l8_d384=outputs/eval/g4vit_l8_d384_s0,outputs/eval/g4vit_l8_d384_s1,outputs/eval/g4vit_l8_d384_s2 \\
        --lmms teacher=outputs/lmms_eval/teacher_e2b_b280 --lmms g4vit_l8_d384=outputs/lmms_eval/g4vit_l8_d384_e2b_b280 \\
        --fidelity g4vit_l8_d384=outputs/fidelity/g4vit_l8_d384 --probe g4vit_l8_d384=outputs/probes/g4vit

Inputs (several comma-separated directories = independent training seeds of one model):

* ``--eval``: ``run_eval.py`` output roots (``<root>/<benchmark>/scored.jsonl`` with per-item scores);
* ``--lmms``: lmms-eval ``--log_samples`` outputs (``*_samples_<task>.jsonl``); the per-item score is
  read from the sample record (best effort: the first numeric metric field, or ``score`` / ``acc`` /
  ``correct`` inside a dict-valued metric field);
* ``--fidelity``: ``run_fidelity.py`` outputs (``summary.json``, ``per_image.jsonl``);
* ``--probe``: ``run_probe.py`` outputs (``results.json``).

For every benchmark and model: mean over items with a 95% bootstrap CI (first seed), mean +- std over
seeds, retention relative to ``--reference`` and a paired bootstrap test of the difference (Holm-
corrected across benchmarks). Writes ``tables.md`` and ``tables.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onboard_vlm.eval.stats import bootstrap_ci, holm, paired_bootstrap, seed_summary

NON_METRIC = {"doc_id", "doc", "target", "arguments", "resps", "filtered_resps", "doc_hash", "prompt_hash", "target_hash",
              "input", "filter", "metrics", "id", "question_id", "task", "split"}


def _pairs(specs: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in specs or []:
        name, dirs = s.split("=", 1)
        out[name] = [d for d in dirs.split(",") if d]
    return out


def load_eval_scores(root: str) -> dict[str, dict[str, float]]:
    """``{benchmark: {sample_id: item_score}}`` from a run_eval output root."""
    out: dict[str, dict[str, float]] = {}
    for path in glob.glob(os.path.join(root, "*", "scored.jsonl")):
        bench = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        out[bench] = {str(r["sample_id"]): float(r["item_score"]) for r in rows if r.get("item_score") is not None}
    return out


def _lmms_item(rec: dict) -> float | None:
    for k, v in rec.items():
        if k in NON_METRIC:
            continue
        if isinstance(v, bool | int | float):
            return float(v)
        if isinstance(v, dict):
            for kk in ("score", "acc", "accuracy", "correct", "exact_match", "anls"):
                if isinstance(v.get(kk), bool | int | float):
                    return float(v[kk])
    return None


def load_lmms_scores(root: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path in glob.glob(os.path.join(root, "**", "*_samples_*.jsonl"), recursive=True):
        task = re.sub(r"\.jsonl$", "", path.rsplit("_samples_", 1)[1])
        scores = {}
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                s = _lmms_item(rec)
                if s is not None:
                    scores[str(rec.get("doc_id"))] = s
        if scores:
            out[f"lmms/{task}"] = scores
    return out


def benchmark_table(runs: dict[str, list[dict[str, dict[str, float]]]], reference: str | None) -> dict:
    """``runs[model] = [per-seed {benchmark: {item: score}}]`` -> table entries."""
    benches = sorted({b for seeds in runs.values() for s in seeds for b in s})
    table: dict = {}
    for model, seeds in runs.items():
        for b in benches:
            per_seed = [s[b] for s in seeds if b in s]
            if not per_seed:
                continue
            e = {"ci_first_seed": bootstrap_ci(list(per_seed[0].values())), "seeds": seed_summary([sum(v.values()) / len(v) for v in per_seed])}
            if reference and model != reference and reference in runs and any(b in s for s in runs[reference]):
                ref = next(s[b] for s in runs[reference] if b in s)
                common = sorted(set(ref) & set(per_seed[0]))
                if common:
                    e["vs_reference"] = paired_bootstrap([per_seed[0][i] for i in common], [ref[i] for i in common])
                    rm = sum(ref[i] for i in common) / len(common)
                    e["retention"] = (sum(per_seed[0][i] for i in common) / len(common)) / rm if rm else float("nan")
            table.setdefault(b, {})[model] = e
    for model in runs:  # Holm correction across benchmarks, per model
        ps = {b: t[model]["vs_reference"]["p"] for b, t in table.items() if model in t and "vs_reference" in t[model]}
        for b, p in holm(ps).items():
            table[b][model]["vs_reference"]["p_holm"] = p
    return table


def fidelity_table(fid: dict[str, list[str]], reference: str | None) -> dict:
    out: dict = {}
    per_image: dict[str, dict[tuple, float]] = {}
    for model, dirs in fid.items():
        with open(os.path.join(dirs[0], "summary.json")) as f:
            summ = json.load(f)
        for e in summ["sets"]:
            out.setdefault(f"{e['set']}@{e['budget']}", {})[model] = {k: e.get(k) for k in ("lm_cos", "lm_rel_err", "vis_cos", "cka", "num_images")}
        pi = os.path.join(dirs[0], "per_image.jsonl")
        if os.path.isfile(pi):
            with open(pi) as f:
                per_image[model] = {(r["set"], r["budget"], r["sample_id"]): r.get("lm_cos") for r in map(json.loads, f)}
    if reference in per_image:
        for model, vals in per_image.items():
            if model == reference:
                continue
            for key in out:
                name, budget = key.rsplit("@", 1)
                ids = [k for k in vals if k[0] == name and str(k[1]) == budget and k in per_image[reference]]
                if ids and model in out[key]:
                    out[key][model]["vs_reference"] = paired_bootstrap([vals[k] for k in ids], [per_image[reference][k] for k in ids])
    return out


def _fmt(ci: dict | None) -> str:
    return f"{ci['mean']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]" if isinstance(ci, dict) and "mean" in ci else "-"


def write_markdown(path: str, tables: dict, reference: str | None) -> None:
    lines = ["# Results", ""]
    if tables.get("benchmarks"):
        lines += ["## Benchmarks", "", f"mean [95% bootstrap CI] | seeds mean +- std | retention and paired diff vs `{reference}` (Holm p)", "",
                  "| benchmark | model | score | seeds | retention | diff vs ref [CI] | p (Holm) |", "|---|---|---|---|---:|---|---:|"]
        for b, models in tables["benchmarks"].items():
            for m, e in models.items():
                vs = e.get("vs_reference")
                diff = f"{vs['diff']:+.4f} [{vs['lo']:+.4f}, {vs['hi']:+.4f}]" if vs else "-"
                p = f"{vs.get('p_holm', vs['p']):.3g}" if vs else "-"
                ret = f"{e['retention']:.3f}" if "retention" in e else "-"
                s = e["seeds"]
                lines.append(f"| {b} | {m} | {_fmt(e['ci_first_seed'])} | {s['mean']:.4f} +- {s['std']:.4f} (n={s['n_seeds']}) | {ret} | {diff} | {p} |")
        lines.append("")
    if tables.get("fidelity"):
        lines += ["## E1 fidelity (LM-space cosine)", "", "| set@budget | model | LM cos [CI] | LM rel. err [CI] | CKA [CI] |", "|---|---|---|---|---|"]
        for k, models in tables["fidelity"].items():
            for m, e in models.items():
                lines.append(f"| {k} | {m} | {_fmt(e.get('lm_cos'))} | {_fmt(e.get('lm_rel_err'))} | {_fmt(e.get('cka'))} |")
        lines.append("")
    if tables.get("probes"):
        lines += ["## E2 probes", "", "| model | ImageNet k-NN | ImageNet linear | ADE20K mIoU | NYUv2 RMSE |", "|---|---|---|---|---|"]
        for m, r in tables["probes"].items():
            im, ade, nyu = r.get("imagenet", {}), r.get("ade20k", {}), r.get("nyuv2", {})
            lines.append(f"| {m} | {_fmt(im.get('knn_top1'))} | {_fmt(im.get('linear_top1'))} | {ade.get('miou', float('nan')):.4f} | {_fmt(nyu.get('rmse'))} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", action="append", default=[])
    p.add_argument("--lmms", action="append", default=[])
    p.add_argument("--fidelity", action="append", default=[])
    p.add_argument("--probe", action="append", default=[])
    p.add_argument("--reference", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    runs: dict[str, list[dict]] = defaultdict(list)
    for model, dirs in _pairs(args.eval).items():
        runs[model] = [load_eval_scores(d) for d in dirs]
    for model, dirs in _pairs(args.lmms).items():
        lm = [load_lmms_scores(d) for d in dirs]
        if model in runs:
            for i, s in enumerate(lm):
                runs[model][i].update(s) if i < len(runs[model]) else runs[model].append(s)
        else:
            runs[model] = lm
    tables = {"benchmarks": benchmark_table(dict(runs), args.reference) if runs else {},
              "fidelity": fidelity_table(_pairs(args.fidelity), args.reference) if args.fidelity else {},
              "probes": {m: json.load(open(os.path.join(d[0], "results.json"))) for m, d in _pairs(args.probe).items()}}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "tables.json"), "w") as f:
        json.dump(tables, f, indent=1, default=str)
    write_markdown(os.path.join(args.out, "tables.md"), tables, args.reference)
    print(f"wrote {args.out}/tables.md")


if __name__ == "__main__":
    main()
