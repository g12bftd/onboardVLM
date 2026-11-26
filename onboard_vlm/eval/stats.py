"""Statistics for the paper tables: bootstrap confidence intervals, paired comparisons, seed summaries.

Every number reported for a model is a mean over evaluation items with a 95% percentile-bootstrap
CI. Comparisons between two models on the *same* items (student vs teacher, method A vs B) use paired
tests: a paired bootstrap of the mean difference (any per-item score) and the exact McNemar test
(binary correctness). When a model has several training seeds, the comparison resamples seeds as well
as items (:func:`hierarchical_paired_bootstrap`), so a difference must hold across training runs, not
only across evaluation items. Across seeds we also report mean +- standard deviation.
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


def hierarchical_paired_bootstrap(
    a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0, chunk: int = 250
) -> dict[str, float]:
    """Paired comparison of two *trained* models with several seeds each, on the same items.

    ``a`` is ``seeds_a x items`` and ``b`` is ``seeds_b x items`` (same items, same order; one row for a
    deterministic model such as the teacher). Every replicate resamples the items (shared by both
    models, so the comparison stays paired) and, independently, the seeds of each model, so the CI
    covers both the evaluation-set sampling and the training-seed variance. With one seed per model
    it reduces to :func:`paired_bootstrap`. Items with a NaN in any row are dropped.
    """
    A, B = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(f"need seeds x items arrays over the same items, got {A.shape} and {B.shape}")
    ok = ~(np.isnan(A).any(0) | np.isnan(B).any(0))
    A, B = A[:, ok], B[:, ok]
    n = A.shape[1]
    if n == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "n": 0, "seeds": [len(A), len(B)]}
    rng = np.random.default_rng(seed)
    obs = A.mean() - B.mean()
    bd = np.empty(n_boot, dtype=np.float64)
    for s0 in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s0)
        w = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(k)]).astype(np.float64) / n  # item weights
        ra, rb = A @ w.T, B @ w.T                         # per-seed resampled means: (seeds, k)
        sa = rng.integers(0, len(A), size=(k, len(A)))    # resampled seeds per replicate
        sb = rng.integers(0, len(B), size=(k, len(B)))
        cols = np.arange(k)
        bd[s0 : s0 + k] = ra[sa, cols[:, None]].mean(1) - rb[sb, cols[:, None]].mean(1)
    centered = bd - obs
    p = float(min(1.0, (np.abs(centered) >= abs(obs)).mean() + 1.0 / n_boot))
    return {"diff": float(obs), "lo": float(np.quantile(bd, alpha / 2)), "hi": float(np.quantile(bd, 1 - alpha / 2)), "p": p, "n": int(n),
            "seeds": [int(len(A)), int(len(B))]}


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
