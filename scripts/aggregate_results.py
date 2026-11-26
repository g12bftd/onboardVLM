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
corrected across benchmarks). Writes ``tables.md``, ``tables.tex`` (booktabs) and ``tables.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onboard_vlm.eval.stats import bootstrap_ci, hierarchical_paired_bootstrap, holm, paired_bootstrap, seed_summary

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
    from onboard_vlm.eval.suites import item_score

    out: dict[str, dict[str, float]] = {}
    for path in glob.glob(os.path.join(root, "**", "*_samples_*.jsonl"), recursive=True):
        task = re.sub(r"\.jsonl$", "", path.rsplit("_samples_", 1)[1])
        scores = {}
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                s = item_score(task, rec)
                s = _lmms_item(rec) if s is None else s
                if s is not None:
                    scores[str(rec.get("doc_id"))] = s
        if scores:
            out[f"lmms/{task}"] = scores
    return out


def load_lmms_headlines(root: str) -> dict[str, float]:
    """``{task: headline score on 0-100}`` from lmms-eval's ``*_results.json`` (``onboard_vlm.eval.suites.headlines``)."""
    from onboard_vlm.eval.suites import headlines

    return headlines(root)


def suite_table(heads: dict[str, list[dict[str, float]]], reference: str | None, blind: str | None) -> dict:
    """Cambrian-1-style table: per task and category (seed mean +- std, 0-100), overall mean, retention vs ``reference``
    and the share of the reference's advantage over ``blind`` (no image) that each model keeps."""
    from onboard_vlm.eval.suites import CATEGORIES, EXTRA

    groups = {**CATEGORIES, **EXTRA}
    out: dict = {"models": {}, "categories": list(groups), "reference": reference, "blind": blind}

    def stats(vals: list[float]) -> dict[str, float]:
        a = np.array(vals, dtype=float)
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n_seeds": len(a)}

    for model, seeds in heads.items():
        tasks = sorted({t for sd in seeds for t in sd})
        e: dict = {"tasks": {t: stats([sd[t] for sd in seeds if t in sd]) for t in tasks}, "categories": {}}
        for c, ts in groups.items():
            have = [t for t in ts if all(t in sd for sd in seeds)]
            if have:  # per seed: mean over the category's benchmarks, then over seeds
                e["categories"][c] = {**stats([float(np.mean([sd[t] for t in have])) for sd in seeds]), "benchmarks": have,
                                      "missing": [t for t in ts if t not in have]}
        core = [t for ts in CATEGORIES.values() for t in ts if all(t in sd for sd in seeds)]
        if core:
            e["overall"] = {**stats([float(np.mean([sd[t] for t in core])) for sd in seeds]), "benchmarks": core}
        out["models"][model] = e
    ref = out["models"].get(reference) if reference else None
    bl = out["models"].get(blind) if blind else None
    for e in out["models"].values():
        for scope, key in [(e["categories"], c) for c in e["categories"]] + ([(e, "overall")] if "overall" in e else []):
            v = scope[key]["mean"]
            r = (ref["categories"].get(key) if key != "overall" else ref.get("overall")) if ref else None
            b = (bl["categories"].get(key) if key != "overall" else bl.get("overall")) if bl else None
            if r:
                scope[key]["retention"] = v / r["mean"] if r["mean"] else float("nan")
                if b and r["mean"] != b["mean"]:
                    scope[key]["visual_retention"] = (v - b["mean"]) / (r["mean"] - b["mean"])
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
                refs = [s[b] for s in runs[reference] if b in s]
                e["vs_reference"], e["retention"] = compare(per_seed, refs)
            table.setdefault(b, {})[model] = e
    for model in runs:  # Holm correction across benchmarks, per model
        ps = {b: t[model]["vs_reference"]["p"] for b, t in table.items() if model in t and "vs_reference" in t[model]}
        for b, p in holm(ps).items():
            table[b][model]["vs_reference"]["p_holm"] = p
    return table


def compare(model_seeds: list[dict[str, float]], ref_seeds: list[dict[str, float]]) -> tuple[dict, float]:
    """Paired test of a model against the reference on the items every seed of both has: a plain paired
    bootstrap with one seed each, else seeds and items resampled together; and the retention model / reference."""
    common = sorted(set.intersection(*(set(s) for s in model_seeds + ref_seeds)))
    if not common:
        return {}, float("nan")
    A = [[s[i] for i in common] for s in model_seeds]
    B = [[s[i] for i in common] for s in ref_seeds]
    if len(A) == 1 and len(B) == 1:
        test = {**paired_bootstrap(A[0], B[0]), "method": "paired bootstrap over items"}
    else:
        test = {**hierarchical_paired_bootstrap(A, B), "method": f"paired bootstrap over items and seeds ({len(A)} vs {len(B)})"}
    rm = sum(map(sum, B)) / (len(B) * len(common))
    return test, (sum(map(sum, A)) / (len(A) * len(common))) / rm if rm else float("nan")


def fidelity_table(fid: dict[str, list[str]], reference: str | None) -> dict:
    """E1 per set@budget: the summary of the first seed, and a paired comparison with ``reference`` over all seeds."""
    out: dict = {}
    per_image: dict[str, list[dict[tuple, float]]] = {}
    for model, dirs in fid.items():
        with open(os.path.join(dirs[0], "summary.json")) as f:
            summ = json.load(f)
        for e in summ["sets"]:
            out.setdefault(f"{e['set']}@{e['budget']}", {})[model] = {k: e.get(k) for k in ("lm_cos", "lm_rel_err", "lm_mse", "vis_cos", "vis_mse", "cka", "num_images")}
        for d in dirs:
            pi = os.path.join(d, "per_image.jsonl")
            if os.path.isfile(pi):
                with open(pi) as f:
                    per_image.setdefault(model, []).append({(r["set"], r["budget"], r["sample_id"]): r.get("lm_cos") for r in map(json.loads, f)})
    if reference in per_image:
        for model, seeds in per_image.items():
            if model == reference:
                continue
            for key in out:
                name, budget = key.rsplit("@", 1)
                sel = lambda d, name=name, budget=budget: {k: v for k, v in d.items() if k[0] == name and str(k[1]) == budget}  # noqa: E731
                if model in out[key]:
                    test, _ = compare([sel(s) for s in seeds], [sel(s) for s in per_image[reference]])
                    if test:
                        out[key][model]["vs_reference"] = test
    return out


def _fmt(ci: dict | None) -> str:
    return f"{ci['mean']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]" if isinstance(ci, dict) and "mean" in ci else "-"


def write_markdown(path: str, tables: dict, reference: str | None) -> None:
    lines = ["# Results", ""]
    st = tables.get("suite") or {}
    if st.get("models"):
        cats = [c for c in st["categories"] if any(c in e["categories"] for e in st["models"].values())]
        lines += ["## General capabilities (Cambrian-1 categories, 0-100, seed mean +- std)", "",
                  "| model | " + " | ".join(cats) + " | overall | retention | visual retention |", "|---" * (len(cats) + 4) + "|"]
        for m, e in st["models"].items():
            cells = [f"{e['categories'][c]['mean']:.1f} +- {e['categories'][c]['std']:.1f}" if c in e["categories"] else "-" for c in cats]
            o = e.get("overall", {})
            lines.append(f"| {m} | " + " | ".join(cells) + f" | {o.get('mean', float('nan')):.1f} | {o.get('retention', float('nan')):.3f} | "
                         f"{o.get('visual_retention', float('nan')):.3f} |")
        lines.append("")
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


def _tex(x: str) -> str:
    return str(x).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def _cell(mean: float, spread: str, best: bool, mark: bool) -> str:
    v = f"{mean:.3f}"
    v = f"\\textbf{{{v}}}" if best else v
    return f"{v}{spread}" + ("$^\\dagger$" if mark else "")


def latex_table(caption: str, label: str, cols: list[str], rows: list[tuple[str, list[str]]]) -> str:
    """A booktabs table (needs \\usepackage{booktabs}); cells are pre-formatted strings."""
    spec = "l" + "c" * len(cols)
    out = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(["model", *map(_tex, cols)]) + " \\\\", "\\midrule"]
    out += [" & ".join([_tex(name), *cells]) + " \\\\" for name, cells in rows]
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def write_latex(path: str, tables: dict, reference: str | None, higher_is_better: dict[str, bool] | None = None) -> None:
    """Paper tables: models x benchmarks (seed mean +- std, or mean with its 95% CI for one seed; best per column in
    bold; $^\\dagger$: differs from ``reference`` at Holm-corrected p < 0.05), E1 fidelity per budget, E2 probes."""
    hib = higher_is_better or {}
    parts = ["% generated by scripts/aggregate_results.py (\\usepackage{booktabs})", ""]
    st = tables.get("suite") or {}
    if st.get("models"):
        cats = [c for c in st["categories"] if any(c in e["categories"] for e in st["models"].values())]
        models = list(st["models"])
        if reference in models:
            models = [reference] + [m for m in models if m != reference]
        cols = [*cats, "Overall", "Ret.", "Vis. ret."]
        best = {c: max(e["categories"][c]["mean"] for e in st["models"].values() if c in e["categories"]) for c in cats}
        rows = []
        for m in models:
            e = st["models"][m]
            cells = [_cell(e["categories"][c]["mean"], f" $\\pm$ {e['categories'][c]['std']:.1f}" if e["categories"][c]["n_seeds"] > 1 else "",
                           abs(e["categories"][c]["mean"] - best[c]) < 1e-9, False).replace(".000", ".0") if c in e["categories"] else "--" for c in cats]
            o = e.get("overall", {})
            cells += [f"{o['mean']:.1f}" if o else "--", f"{o['retention']:.3f}" if "retention" in o else "--",
                      f"{o['visual_retention']:.3f}" if "visual_retention" in o else "--"]
            rows.append((m, cells))
        parts.append(latex_table("General vision-language capabilities with the student in the unchanged Gemma 4 (Cambrian-1 "
                                 "categories, 0--100, seed mean $\\pm$ std). Ret.: overall / reference. Vis. ret.: (model $-$ blind) / "
                                 "(reference $-$ blind), blind = every image grey.", "tab:general", cols, rows))
    bench = tables.get("benchmarks") or {}
    if bench:
        cols = list(bench)
        models = list(dict.fromkeys(m for b in cols for m in bench[b]))
        if reference in models:
            models = [reference] + [m for m in models if m != reference]
        means = {(b, m): bench[b][m]["seeds"]["mean"] for b in cols for m in bench[b]}
        best = {b: (max if hib.get(b, True) else min)(v for (bb, _), v in means.items() if bb == b) for b in cols}
        rows = []
        for m in models:
            cells = []
            for b in cols:
                e = bench[b].get(m)
                if e is None:
                    cells.append("--")
                    continue
                sd = e["seeds"]
                spread = f" $\\pm$ {sd['std']:.3f}" if sd["n_seeds"] > 1 else f" [{e['ci_first_seed']['lo']:.3f}, {e['ci_first_seed']['hi']:.3f}]"
                vs = e.get("vs_reference") or {}
                cells.append(_cell(sd["mean"], spread, abs(sd["mean"] - best[b]) < 1e-12, vs.get("p_holm", 1.0) < 0.05))
            rows.append((m, cells))
        parts.append(latex_table(f"Benchmarks (seed mean $\\pm$ std; one seed: 95\\% bootstrap CI). Bold: best. $^\\dagger$: differs from "
                                 f"{_tex(reference or '-')} (paired bootstrap over items and seeds, Holm-corrected $p<0.05$).",
                                 "tab:benchmarks", cols, rows))
    fid = tables.get("fidelity") or {}
    budgets = sorted({k.rsplit("@", 1)[1] for k in fid}, key=lambda b: int(b) if b.isdigit() else 0)
    for bud in budgets:
        keys = [k for k in fid if k.endswith("@" + bud)]
        sets = [k.rsplit("@", 1)[0] for k in keys]
        models = list(dict.fromkeys(m for k in keys for m in fid[k]))
        best = {k: max((e.get("lm_cos") or {}).get("mean", float("-inf")) for e in fid[k].values()) for k in keys}
        rows = []
        for m in models:
            cells = []
            for k in keys:
                e = fid[k].get(m) or {}
                c = e.get("lm_cos") or {}
                if "mean" not in c:
                    cells.append("--")
                    continue
                vs = e.get("vs_reference") or {}
                cells.append(_cell(c["mean"], "", abs(c["mean"] - best[k]) < 1e-12, vs.get("p", 1.0) < 0.05))
            rows.append((m, cells))
        parts.append(latex_table(f"E1 fidelity: LM-input-space cosine at {bud} tokens (mean over images). Bold: best.",
                                 f"tab:fidelity_b{bud}", sets, rows))
    probes = tables.get("probes") or {}
    if probes:
        def get(r, task, key):
            v = (r.get(task) or {}).get(key)
            return v.get("mean") if isinstance(v, dict) else v
        cols = [("ImageNet k-NN", "imagenet", "knn_top1", True), ("ImageNet linear", "imagenet", "linear_top1", True),
                ("ADE20K mIoU", "ade20k", "miou", True), ("NYUv2 RMSE", "nyuv2", "rmse", False)]
        cols = [c for c in cols if any(get(r, c[1], c[2]) is not None for r in probes.values())]
        best = {c[0]: (max if c[3] else min)(v for r in probes.values() if (v := get(r, c[1], c[2])) is not None) for c in cols}
        rows = [(m, [_cell(v, "", abs(v - best[c[0]]) < 1e-12, False) if (v := get(r, c[1], c[2])) is not None else "--" for c in cols])
                for m, r in probes.items()]
        parts.append(latex_table("E2 frozen-feature probes. Bold: best (lowest RMSE).", "tab:probes", [c[0] for c in cols], rows))
    with open(path, "w") as f:
        f.write("\n".join(parts))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", action="append", default=[])
    p.add_argument("--lmms", action="append", default=[])
    p.add_argument("--fidelity", action="append", default=[])
    p.add_argument("--probe", action="append", default=[])
    p.add_argument("--reference", default=None)
    p.add_argument("--blind", default=None, help="the --lmms model run with blind=True (no image): visual retention")
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
    heads = {model: [load_lmms_headlines(d) for d in dirs] for model, dirs in _pairs(args.lmms).items()}
    tables = {"suite": suite_table(heads, args.reference, args.blind) if any(any(h) for h in heads.values()) else {},
              "benchmarks": benchmark_table(dict(runs), args.reference) if runs else {},
              "fidelity": fidelity_table(_pairs(args.fidelity), args.reference) if args.fidelity else {},
              "probes": {m: json.load(open(os.path.join(d[0], "results.json"))) for m, d in _pairs(args.probe).items()}}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "tables.json"), "w") as f:
        json.dump(tables, f, indent=1, default=str)
    write_markdown(os.path.join(args.out, "tables.md"), tables, args.reference)
    write_latex(os.path.join(args.out, "tables.tex"), tables, args.reference)
    print(f"wrote {args.out}/tables.md and tables.tex")


if __name__ == "__main__":
    main()
