"""Figures of a shrinking study (docs/shrink_study.md): what distillation does over training time, per size.

Reads, for every run matched by ``--runs`` (experiment names; all their seeds are found):

* training-time validation curves: ``<root>/logs/<tag>/<run_id>/metrics.csv`` (CSVLogger), on a compute axis
  (``progress/images_seen``);
* checkpoint evaluations: ``<root>/trajectories/<run_id>/step_*/{trajectory.json, eval/e1|e2|e3}``
  (``scripts/trajectory_eval.py``), and the teacher's (``--teacher_dir``) as reference lines.

Writes PNG + PDF + CSV (the plotted values, as a table) per figure into ``--out``:

* ``curves_training``   training loss, gradient norm, clip rate, update ratio, learning rate, throughput (EMA-smoothed);
* ``curves_loss_terms`` every training loss term;
* ``curves_fidelity``   val LM-space cosine / MSE, vision-space cosine / MSE, CKA, worst-token cosine vs images seen;
* ``curves_budgets``    val LM-space cosine per token budget;   ``curves_groups``: per corpus group;
* ``trajectory_e1``     held-out E1 fidelity (fixed subsets) at the checkpoints;
* ``trajectory_e2``     ImageNet k-NN / linear, NYUv2 depth at the checkpoints (teacher dashed);
* ``trajectory_e3``     VizWiz, TextVQA, CV-Bench, EgoSchema accuracy at the checkpoints (teacher dashed);
* ``final_vs_params``   final value vs student parameters (seed mean +- std);
* ``grid_<metric>``     final value over depth x width (when both vary; per-series curves need a row or column);
* ``fidelity_vs_downstream``  E1 fidelity vs E3 accuracy over all checkpoints, Spearman rho with a 95% CI
                              from a bootstrap over runs (checkpoints of one run are not independent).

Each seed is a thin line, the seed mean a bold line with markers; ordered sweeps (``--order``) use one hue
light -> dark, other comparisons the categorical palette in a fixed order (both validated for colour-vision
deficiencies).

    python scripts/plot_study.py --root $ONBOARD_VLM_OUT --out $ONBOARD_VLM_OUT/plots/depth \\
        --runs "shrink_L*_D768_teacher_uniform-uniform" --label "{L} layers" --order L --teacher_dir $ONBOARD_VLM_OUT/trajectories/_teacher_e2b
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# validated palettes (dataviz validator): 5-step ordinal blue ramp, 4 categorical slots in fixed order
RAMP = ["#84b9f3", "#519ced", "#257de0", "#1b64be", "#0f4a96"]
CATEGORICAL = ["#257de0", "#f47230", "#00b894", "#f2a600"]
HEAT = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"
# paper sizing: total figure width in inches (NeurIPS text width 5.5; ICML full width 6.75, column 3.25), fonts in pt
PAPER = {"width": 5.5, "title": 7.5, "label": 7, "tick": 6, "legend": 7, "note": 6}
TRAIN_PANELS = [("train/total", "training loss (total)"), ("diag/grad_norm", "gradient norm (pre-clip)"),
                ("diag/grad_clipped", "fraction of steps clipped"), ("diag/update_ratio", "update ratio ||dθ|| / ||θ||"),
                ("lr", "learning rate"), ("diag/images_per_s", "throughput (images / s)")]

NAME = re.compile(r"shrink_L(?P<L>\d+)_D(?P<D>\d+)_(?P<init>[a-z]+)_(?P<ls>[^-]+)-(?P<ws>.+)$")
SEED = re.compile(r"_(?P<tag>s\d+(?:_i\d+_d\d+)?)$")
E3 = {"vizwiz_vqa": ("vqa_accuracy", "VizWiz val (VQA acc.)"), "textvqa": ("vqa_accuracy", "TextVQA val (VQA acc.)"),
      "cvbench": ("cv_bench_overall", "CV-Bench (overall)"), "egoschema_subset": ("accuracy", "EgoSchema subset (acc.)")}
NICE = {"e2/imagenet/knn_top1": "ImageNet k-NN top-1", "e2/imagenet/linear_top1": "ImageNet linear top-1",
        "e2/nyuv2/rmse": "NYUv2 depth RMSE (lower is better)", "e2/nyuv2/absrel": "NYUv2 depth AbsRel (lower is better)",
        "e2/nyuv2/delta1": "NYUv2 depth delta1", "e2/ade20k/pixel_acc": "ADE20K pixel accuracy"}
FIDELITY_PANELS = [("fid_lm_cos", "LM-space cosine"), ("fid_lm_mse", "LM-space MSE"), ("fid_vis_cos", "vision cosine"),
                   ("fid_vis_mse", "vision MSE"), ("fid_cka", "CKA"), ("fid_vis_cos_p05", "worst-5% token cosine")]


@dataclass
class Run:
    run_id: str
    config: str
    fields: dict
    traj: str
    metrics_csv: str | None
    curves: dict = field(default_factory=dict)      # column -> (images_seen[], value[])
    points: list = field(default_factory=list)      # trajectory checkpoints: dicts of metrics


# ----------------------------------------------------------------------------------------------- loading
def load_curves(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    prog = sorted({(int(r["step"]), float(r["progress/images_seen"])) for r in rows if r.get("progress/images_seen")})
    st = np.array([p[0] for p in prog] or [0], dtype=float)
    im = np.array([p[1] for p in prog] or [0], dtype=float)
    out = {}
    for col in rows[0].keys() if rows else []:
        if not col.startswith(("val", "train/", "diag/", "lr-")):
            continue
        seen, pts = set(), []
        for r in rows:  # the first value per step: later rows at the same step are the post-training re-validation
            if r.get(col) and r["step"] and int(r["step"]) not in seen:
                seen.add(int(r["step"]))
                pts.append((int(r["step"]), float(r[col])))
        if pts:
            s = np.array([p[0] for p in pts], dtype=float)
            out[col] = (np.interp(s, st, im), np.array([p[1] for p in pts]))
    lr = sorted(c for c in out if c.startswith("lr-"))
    if lr:
        out["lr"] = out[lr[0]]  # first parameter group (the student)
    return out


def load_point(step_dir: str) -> dict:
    with open(os.path.join(step_dir, "trajectory.json")) as f:
        p = dict(json.load(f))
    return {**p, **load_evals(step_dir)}


def load_evals(d: str) -> dict:
    out: dict = {}
    e1 = os.path.join(d, "eval", "e1", "summary.json")
    if os.path.isfile(e1):
        with open(e1) as f:
            sets = json.load(f)["sets"]
        for b in sorted({e["budget"] for e in sets}):
            for m in ("lm_cos", "lm_mse", "vis_cos", "vis_mse", "cka"):
                vals = [e[m]["mean"] for e in sets if e["budget"] == b and e.get("group") == "ood" and isinstance(e.get(m), dict)]
                if vals:
                    out[f"e1/ood/b{b}/{m}"] = float(np.mean(vals))
    e2 = os.path.join(d, "eval", "e2", "results.json")
    if os.path.isfile(e2):
        with open(e2) as f:
            r = json.load(f)
        for task, res in r.items():
            if isinstance(res, dict):
                for k, v in res.items():
                    if isinstance(v, dict) and "mean" in v:
                        out[f"e2/{task}/{k}"] = float(v["mean"])
    for b, (key, _) in E3.items():
        p = os.path.join(d, "eval", "e3", b, "metrics.json")
        if os.path.isfile(p):
            with open(p) as f:
                v = json.load(f).get(key)
            if isinstance(v, (int, float)):
                out[f"e3/{b}"] = float(v)
    return out


def find_runs(root: str, patterns: list[str], tag: str) -> list[Run]:
    runs = []
    for traj in sorted(glob.glob(os.path.join(root, "trajectories", "*"))):
        rid = os.path.basename(traj)
        m = SEED.search(rid)
        if not m:
            continue
        config = rid[: m.start()]
        if not any(fnmatch.fnmatch(config, p) for p in patterns):
            continue
        nm = NAME.match(config)
        fields = {k: (int(v) if v.isdigit() else v) for k, v in nm.groupdict().items()} if nm else {}
        mc = os.path.join(root, "logs", tag, rid, "metrics.csv")
        run = Run(rid, config, fields, traj, mc if os.path.isfile(mc) else None)
        if run.metrics_csv:
            run.curves = load_curves(run.metrics_csv)
        run.points = [load_point(s) for s in sorted(glob.glob(os.path.join(traj, "step_*"))) if os.path.isfile(os.path.join(s, "trajectory.json"))]
        runs.append(run)
    return runs


def params_of(run: Run) -> float | None:
    steps = sorted(glob.glob(os.path.join(run.traj, "step_*", "student.safetensors")))
    if not steps:
        return None
    from safetensors import safe_open

    with safe_open(steps[-1], "pt") as f:
        return float(sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys()))


# ----------------------------------------------------------------------------------------------- styling
def style(ax, title: str, xlabel: str | None = None) -> None:
    ax.set_title(title, fontsize=PAPER["title"], color=TEXT, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TEXT2)
    ax.tick_params(colors=TEXT2, labelsize=PAPER["tick"])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=PAPER["label"], color=TEXT2)


def series_colors(labels: list[str], ordinal: bool) -> dict[str, str]:
    if ordinal:
        if len(labels) > len(RAMP):
            raise SystemExit(f"{len(labels)} ordered series: at most {len(RAMP)} on one ramp (split the sweep or facet it)")
        idx = np.round(np.linspace(0, len(RAMP) - 1, len(labels))).astype(int) if len(labels) > 1 else [2]
        return {lab: RAMP[i] for lab, i in zip(labels, idx)}
    if len(labels) > len(CATEGORICAL):
        raise SystemExit(f"{len(labels)} series: at most {len(CATEGORICAL)} categorical colours (split the comparison)")
    return dict(zip(labels, CATEGORICAL))


def legend(fig, colors: dict[str, str], teacher: bool) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=4, label=lab) for lab, c in colors.items()]
    if teacher:
        handles.append(Line2D([], [], color=TEXT2, lw=1.2, ls="--", label="teacher (original encoder)"))
    if len(handles) >= 2:
        fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 6), frameon=False, fontsize=PAPER["legend"], labelcolor=TEXT)


# ----------------------------------------------------------------------------------------------- figures
def by_series(runs: list[Run], label: str, order: str | None) -> tuple[dict[str, list[Run]], list[str]]:
    groups: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        groups[label.format(**r.fields) if r.fields else r.config].append(r)
    keys = list(groups)
    if order:
        keys.sort(key=lambda k: groups[k][0].fields.get(order, 0))
    return groups, keys


def mean_curve(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Mean over seeds on the x values every seed has (seeds share the batch plan, so the steps coincide)."""
    common = set(np.round(curves[0][0], 3))
    for x, _ in curves[1:]:
        common &= set(np.round(x, 3))
    xs = np.array(sorted(common))
    ys = np.mean([[y[np.isclose(np.round(x, 3), v)][0] for v in xs] for x, y in curves], axis=0) if len(xs) else np.array([])
    return xs, ys


def panel_figure(path: str, panels: list[tuple[str, str]], groups, keys, colors, get, teacher_ref=None, xlabel="images seen (M)") -> None:
    """Small multiples: one panel per metric; thin seed lines, bold seed mean, dashed teacher reference."""
    import matplotlib.pyplot as plt

    present = [(k, t) for k, t in panels if any(get(r, k) is not None for lab in keys for r in groups[lab])]
    if not present:
        return
    n = len(present)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(PAPER["width"], PAPER["width"] / cols * 0.72 * rows + 0.45), squeeze=False)
    table = []
    for ax, (key, title) in zip(axes.flat, present):
        ends: list[tuple[str, float, float]] = []
        for lab in keys:
            seeds = [c for r in groups[lab] if (c := get(r, key)) is not None]
            if not seeds:
                continue
            for x, y in seeds:
                ax.plot(x / 1e6, y, color=colors[lab], lw=0.8, alpha=0.35)
            mx, my = mean_curve(seeds)
            dense = len(mx) > 12  # validation / per-step curves: a line; checkpoint curves keep their markers
            ax.plot(mx / 1e6, my, color=colors[lab], lw=1.6 if dense else 2, marker=None if dense else "o", ms=3.5,
                    markeredgecolor="white", markeredgewidth=0.6)
            if len(keys) <= 4 and len(mx):
                ends.append((lab, mx[-1] / 1e6, my[-1]))
            table += [{"metric": key, "series": lab, "images_seen": float(a), "seed_mean": float(b), "n_seeds": len(seeds)} for a, b in zip(mx, my)]
        ref = teacher_ref(key) if teacher_ref else None
        if ref is not None:
            ax.axhline(ref, color=TEXT2, lw=1.2, ls="--")
        style(ax, title, xlabel)
        direct_labels(ax, ends)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    legend(fig, colors, teacher_ref is not None and any(teacher_ref(k) is not None for k, _ in present))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, path, table)


def direct_labels(ax, ends: list[tuple[str, float, float]], gap_pt: float = 9.0) -> None:
    """Series names at the line ends, pushed apart vertically so they never overlap (text ink, not series colour)."""
    if not ends:
        return
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    gap = gap_pt * fig.dpi / 72
    pts = sorted(((to_px((x, y))[1], lab, x, y) for lab, x, y in ends))
    placed: list[float] = []
    for py, *_ in pts:
        placed.append(max(py, placed[-1] + gap) if placed else py)
    shift = (placed[-1] - pts[-1][0]) / 2 if len(placed) > 1 else 0.0  # centre the stack on the line ends
    if any(b[0] - a[0] < 0.35 * gap for a, b in zip(pts, pts[1:])) or max(abs(q - shift - p[0]) for p, q in zip(pts, placed)) > 1.5 * gap:
        return  # lines end on top of each other: a label far from its line would mislead; the legend identifies them
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + 0.22 * (x1 - x0))  # room for the labels inside the panel (vertical positions are unaffected)
    for (py, lab, x, y), q in zip(pts, placed):
        dy_pt = (q - shift - py) * 72 / fig.dpi
        ax.annotate(lab, (x, y), xytext=(5, dy_pt), textcoords="offset points", fontsize=PAPER["note"], color=TEXT, va="center")


def save(fig, path: str, table: list[dict]) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".png", dpi=200)
    fig.savefig(path + ".pdf")
    plt.close(fig)
    if table:
        with open(path + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    print(f"[plot_study] {path}.png")


def ema(curve, alpha: float):
    """Exponential moving average of a noisy per-step curve (training losses, gradient norms)."""
    if curve is None or alpha <= 0:
        return curve
    x, y = curve
    out = np.empty_like(y, dtype=float)
    acc = None
    for i, v in enumerate(y):
        acc = v if acc is None else alpha * acc + (1 - alpha) * v
        out[i] = acc
    return x, out


def curve_getter(r: Run, key: str):
    return r.curves.get(key)


def point_getter(r: Run, key: str):
    pts = [(p["images_seen"], p[key]) for p in r.points if key in p]
    return (np.array([a for a, _ in pts], dtype=float), np.array([b for _, b in pts])) if pts else None


def final_value(r: Run, key: str) -> float | None:
    c = point_getter(r, key) if "/" in key and not key.startswith("val") else curve_getter(r, key)
    return float(c[1][-1]) if c is not None and len(c[1]) else None


def final_vs_params(path: str, groups, keys, colors, metrics: list[tuple[str, str]], teacher: dict, ordinal: bool) -> None:
    import matplotlib.pyplot as plt

    params = {lab: params_of(groups[lab][0]) for lab in keys}
    if len({p for p in params.values() if p}) < 2:
        return
    present = [(k, t) for k, t in metrics if any(final_value(r, k) is not None for lab in keys for r in groups[lab])]
    if not present:
        return
    cols = min(3, len(present))
    rows = math.ceil(len(present) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(PAPER["width"], PAPER["width"] / cols * 0.75 * rows + 0.45), squeeze=False)
    table = []
    for ax, (key, title) in zip(axes.flat, present):
        xs, ys = [], []
        for lab in keys:
            vals = [v for r in groups[lab] if (v := final_value(r, key)) is not None]
            if not vals or not params[lab]:
                continue
            m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ax.errorbar(params[lab] / 1e6, m, yerr=sd, color=colors[lab], marker="o", ms=5, lw=1.5, capsize=2,
                        markeredgecolor="white", markeredgewidth=0.6)
            xs.append(params[lab] / 1e6)
            ys.append(m)
            table.append({"metric": key, "series": lab, "params_M": params[lab] / 1e6, "seed_mean": m, "seed_std": sd, "n_seeds": len(vals)})
        if ordinal and len(xs) > 1:
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], color=TEXT2, lw=0.8, zorder=0)
        if teacher.get(key) is not None:
            ax.axhline(teacher[key], color=TEXT2, lw=1.2, ls="--")
        ax.set_xscale("log")
        from matplotlib.ticker import FuncFormatter

        plain = FuncFormatter(lambda v, _: f"{v:g}")
        ax.xaxis.set_major_formatter(plain)
        ax.xaxis.set_minor_formatter(plain)
        ax.tick_params(axis="x", which="both", labelsize=PAPER["tick"], colors=TEXT2)
        style(ax, title, "student parameters (M, log)")
    for ax in list(axes.flat)[len(present):]:
        ax.set_visible(False)
    legend(fig, colors, bool(teacher))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, path, table)


def grid_heatmap(path: str, runs: list[Run], key: str = "val/fid_lm_cos", title: str | None = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    Ls = sorted({r.fields.get("L") for r in runs if r.fields})
    Ds = sorted({r.fields.get("D") for r in runs if r.fields})
    if len(Ls) < 2 or len(Ds) < 2:
        return
    grid = np.full((len(Ls), len(Ds)), np.nan)
    table = []
    for i, L in enumerate(Ls):
        for j, D in enumerate(Ds):
            vals = [v for r in runs if r.fields.get("L") == L and r.fields.get("D") == D and (v := final_value(r, key)) is not None]
            if vals:
                grid[i, j] = np.mean(vals)
                table.append({"layers": L, "width": D, "metric": key, "seed_mean": float(np.mean(vals)), "n_seeds": len(vals)})
    if np.isnan(grid).all():
        return
    fig, ax = plt.subplots(figsize=(min(PAPER["width"], 1.0 + 0.8 * len(Ds)), 0.7 + 0.45 * len(Ls)))
    im = ax.imshow(grid, cmap=LinearSegmentedColormap.from_list("blue", HEAT), aspect="auto", origin="lower")
    lo, hi = np.nanmin(grid), np.nanmax(grid)
    for i in range(len(Ls)):
        for j in range(len(Ds)):
            if not np.isnan(grid[i, j]):
                dark = (grid[i, j] - lo) / (hi - lo + 1e-12) > 0.55
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=PAPER["note"], color="white" if dark else TEXT)
    ax.set_xticks(range(len(Ds)), [str(d) for d in Ds])
    ax.set_yticks(range(len(Ls)), [str(L) for L in Ls])
    ax.set_xlabel("width (hidden size)", fontsize=PAPER["label"], color=TEXT2)
    ax.set_ylabel("depth (layers)", fontsize=PAPER["label"], color=TEXT2)
    ax.set_title(f"{title or key}: final value, seed mean", fontsize=PAPER["title"], color=TEXT, loc="left")
    ax.tick_params(colors=TEXT2, labelsize=PAPER["tick"])
    fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=PAPER["tick"], colors=TEXT2)
    fig.tight_layout()
    save(fig, path, table)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def fidelity_vs_downstream(path: str, runs: list[Run], fid_key: str, n_boot: int = 2000) -> None:
    import matplotlib.pyplot as plt

    targets = [(f"e3/{b}", t) for b, (_, t) in E3.items()]
    present = [(k, t) for k, t in targets if any(fid_key in p and k in p for r in runs for p in r.points)]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(PAPER["width"], PAPER["width"] / max(len(present), 2) * 0.85 + 0.35), squeeze=False)
    table = []
    rng = np.random.default_rng(0)
    for ax, (key, title) in zip(axes.flat, present):
        per_run = [[(p[fid_key], p[key]) for p in r.points if fid_key in p and key in p] for r in runs]
        per_run = [pr for pr in per_run if pr]
        pts = np.array([q for pr in per_run for q in pr])
        rho = spearman(pts[:, 0], pts[:, 1])
        boots = []
        for _ in range(n_boot):  # cluster bootstrap over runs
            pick = rng.integers(0, len(per_run), len(per_run))
            b = np.array([q for i in pick for q in per_run[i]])
            boots.append(spearman(b[:, 0], b[:, 1]))
        lo, hi = np.nanquantile(boots, [0.025, 0.975])
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=CATEGORICAL[0], edgecolors="white", linewidths=0.6)
        style(ax, f"{title}\nSpearman rho {rho:.2f} [{lo:.2f}, {hi:.2f}]", "E1 held-out LM-space cosine (280 tokens)")
        table += [{"metric": key, "fidelity": float(a), "downstream": float(b)} for a, b in pts]
    fig.tight_layout()
    save(fig, path, table)


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="output_root of the runs ($ONBOARD_VLM_OUT)")
    p.add_argument("--runs", nargs="+", required=True, help="experiment-name globs (all seeds of each are used)")
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="{L} layers x {D}", help="series label, from the fields L, D, init, ls, ws of the name")
    p.add_argument("--order", default=None, help="numeric field that orders the series (one hue, light -> dark)")
    p.add_argument("--teacher_dir", default=None, help="trajectory_eval.py --teacher output (reference lines)")
    p.add_argument("--tag", default="onboard_vlm", help="logger name (config `tag`)")
    p.add_argument("--fidelity_budget", type=int, default=280)
    p.add_argument("--width", type=float, default=5.5, help="figure width in inches (NeurIPS 5.5, ICML 6.75 / column 3.25)")
    p.add_argument("--smooth", type=float, default=0.9, help="EMA factor for the training-step curves (0 = raw)")
    a = p.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    # camera-ready PDFs: TrueType (Type 42) fonts, not Type 3 (rejected by several venues' format checks)
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    PAPER["width"] = a.width

    runs = find_runs(a.root, a.runs, a.tag)
    if not runs:
        raise SystemExit(f"no runs under {a.root}/trajectories match {a.runs}")
    groups, keys = by_series(runs, a.label, a.order)
    print(f"[plot_study] {len(runs)} runs in {len(keys)} series: " + ", ".join(f"{k} ({len(groups[k])} seeds)" for k in keys))
    teacher = load_evals(a.teacher_dir) if a.teacher_dir else {}
    tref = (lambda k: teacher.get(k)) if teacher else None
    fk = f"e1/ood/b{a.fidelity_budget}/lm_cos"
    heat = [("val/fid_lm_cos", "val LM-space cosine"), ("val/fid_lm_mse", "val LM-space MSE"), (fk, "held-out LM-space cosine"),
            ("e2/imagenet/knn_top1", "ImageNet k-NN top-1"), *[(f"e3/{b}", t) for b, (_, t) in E3.items()]]
    for key, title in heat:  # depth x width grids (only when both vary)
        grid_heatmap(os.path.join(a.out, "grid_" + re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")), runs, key, title)
    fidelity_vs_downstream(os.path.join(a.out, "fidelity_vs_downstream"), runs, fk)
    try:
        colors = series_colors(keys, bool(a.order))
    except SystemExit as e:  # a full grid: the heatmaps above show it; curves need a subset of the runs
        print(f"[plot_study] {e}; per-series curves skipped (plot a row or column of the grid with --runs)")
        return

    terms = sorted({c for r in runs for c in r.curves if c.startswith("train/") and c != "train/total" and not c.startswith("train/fid_")})
    smooth = lambda r, k: ema(curve_getter(r, k), a.smooth)  # noqa: E731
    panel_figure(os.path.join(a.out, "curves_training"), TRAIN_PANELS, groups, keys, colors, smooth, xlabel="images seen (M)")
    panel_figure(os.path.join(a.out, "curves_loss_terms"), [(t, f"training loss: {t[6:]}") for t in terms], groups, keys, colors, smooth)
    panel_figure(os.path.join(a.out, "curves_fidelity"), [(f"val/{k}", t) for k, t in FIDELITY_PANELS], groups, keys, colors, curve_getter)
    budgets = sorted({int(m.group(1)) for r in runs for c in r.curves if (m := re.match(r"val/b(\d+)/fid_lm_cos$", c))})
    panel_figure(os.path.join(a.out, "curves_budgets"), [(f"val/b{b}/fid_lm_cos", f"LM-space cosine, {b} tokens") for b in budgets],
                 groups, keys, colors, curve_getter)
    vgroups = sorted({m.group(1) for r in runs for c in r.curves if (m := re.match(r"val/([a-z_]+)/fid_lm_cos$", c)) and not m.group(1).startswith("b")})
    panel_figure(os.path.join(a.out, "curves_groups"), [(f"val/{g}/fid_lm_cos", f"LM-space cosine, {g}") for g in vgroups],
                 groups, keys, colors, curve_getter)
    e1 = sorted({k for r in runs for pt in r.points for k in pt if k.startswith("e1/") and k.endswith(("lm_cos", "lm_mse", "cka"))})
    panel_figure(os.path.join(a.out, "trajectory_e1"), [(k, "held-out " + k[len("e1/ood/"):].replace("/", " ")) for k in e1],
                 groups, keys, colors, point_getter)
    e2 = sorted({k for r in runs for pt in r.points for k in pt if k.startswith("e2/")})
    panel_figure(os.path.join(a.out, "trajectory_e2"), [(k, NICE.get(k, k[3:].replace("/", " "))) for k in e2], groups, keys, colors, point_getter, tref)
    panel_figure(os.path.join(a.out, "trajectory_e3"), [(f"e3/{b}", t) for b, (_, t) in E3.items()], groups, keys, colors, point_getter, tref)
    final_vs_params(os.path.join(a.out, "final_vs_params"), groups, keys, colors,
                    [("val/fid_lm_cos", "val LM-space cosine"), ("val/fid_lm_mse", "val LM-space MSE"), (fk, "held-out LM-space cosine"),
                     *[(k, NICE.get(k, k[3:].replace("/", " "))) for k in e2 if k.endswith(("knn_top1", "linear_top1"))],
                     *[(f"e3/{b}", t) for b, (_, t) in E3.items()]], teacher, bool(a.order))


if __name__ == "__main__":
    main()
