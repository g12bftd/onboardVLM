"""Statistics for the paper tables: bootstrap confidence intervals, paired comparisons, seed summaries.

Every number reported for a model is a mean over evaluation items with a 95% percentile-bootstrap
CI. Comparisons between two models on the *same* items (student vs teacher, method A vs B) use paired
tests: a paired bootstrap of the mean difference (any per-item score) and the exact McNemar test
(binary correctness). Across training seeds we report mean +- standard deviation.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _boot_means(x: np.ndarray, n_boot: int, seed: int, chunk: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        idx = rng.integers(0, len(x), size=(k, len(x)))
        out[s : s + k] = x[idx].mean(1)
    return out


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Mean and percentile-bootstrap ``1 - alpha`` CI of ``values``."""
    x = np.asarray(values, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    b = _boot_means(x, n_boot, seed)
    return {"mean": float(x.mean()), "lo": float(np.quantile(b, alpha / 2)), "hi": float(np.quantile(b, 1 - alpha / 2)), "n": int(len(x))}


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Mean of ``a - b`` over paired items, its bootstrap CI and a two-sided bootstrap p-value."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays differ in shape {a.shape} vs {b.shape}")
    d = a - b
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "n": 0}
    bd = _boot_means(d, n_boot, seed)
    centered = bd - d.mean()  # bootstrap distribution under H0: mean difference = 0
    p = float(min(1.0, (np.abs(centered) >= abs(d.mean())).mean() + 1.0 / n_boot))
    return {"diff": float(d.mean()), "lo": float(np.quantile(bd, alpha / 2)), "hi": float(np.quantile(bd, 1 - alpha / 2)), "p": p, "n": int(len(d))}


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict[str, float]:
    """Exact McNemar test on paired binary outcomes (two-sided binomial test on discordant pairs)."""
    a, b = np.asarray(a_correct, dtype=bool), np.asarray(b_correct, dtype=bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return {"a_only": n10, "b_only": n01, "p": 1.0}
    k = min(n01, n10)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return {"a_only": n10, "b_only": n01, "p": float(min(1.0, p))}


def seed_summary(values: Sequence[float]) -> dict[str, float]:
    """Mean and sample standard deviation over independent training seeds."""
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "n_seeds": int(len(x))}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control across benchmarks)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
