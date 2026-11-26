"""The general vision-language evaluation suite (E3) and how its numbers are read.

Protocol: Cambrian-1 (Tong et al., NeurIPS 2024), the standard for comparing *vision encoders* inside a VLM: 16
benchmarks in four categories, reported per benchmark, per category (mean) and overall. Our setting is stricter than
theirs: the student replaces the vision tower of the unchanged Gemma 4 (no connector or LLM re-training), so every
difference is the encoder's. All benchmarks run through lmms-eval 0.7.4 with its own Gemma 4 model (prompts,
generation and scoring as published for Gemma 4), at a fixed soft-token budget.

* Three Cambrian benchmarks are scored by a GPT judge in lmms-eval 0.7.4 (MMBench option matching, MathVista answer
  extraction; HallusionBench likewise): they are left out unless ``judged=True``, and the category means say so.
* Two vision-indispensable controls are added: POPE (object hallucination, F1; Li et al., EMNLP 2023) and MMStar
  (samples that need the image, by construction; Chen et al., NeurIPS 2024). InfoVQA completes the OCR category.
* A *blind* run (every image replaced by a grey image of the same size, ``OnboardGemma4(blind=True)``) measures what
  the language model answers without seeing; retention is also reported relative to it:
  ``(student - blind) / (teacher - blind)``, the share of the teacher's *visual* advantage the student keeps.

Every headline number is the task's own lmms-eval aggregate, normalised to 0-100 (MME perception: /20, as in
Cambrian-1). Paired tests use per-question scores from ``--log_samples`` (``item_score``).
"""

from __future__ import annotations

import argparse
from typing import Any

CATEGORIES: dict[str, list[str]] = {
    "General": ["mme", "mmbench_en_dev", "seedbench", "gqa"],
    "Knowledge": ["scienceqa_img", "mmmu_val", "mathvista_testmini", "ai2d"],
    "OCR & Chart": ["chartqa", "ocrbench", "textvqa_val", "docvqa_val"],
    "Vision-Centric": ["mmvp", "realworldqa", "cv_bench_2d", "cv_bench_3d"],
}
EXTRA: dict[str, list[str]] = {"Hallucination & vision-indispensable": ["pope", "mmstar"], "OCR (extra)": ["infovqa_val"]}
JUDGED = {"mmbench_en_dev", "mathvista_testmini", "hallusion_bench_image"}  # need an OpenAI key in lmms-eval 0.7.4

# headline metric of each task in lmms-eval's results json ("<metric>,none"), and its scale to 0-100
HEADLINE: dict[str, tuple[str, float | None]] = {
    "mme": ("mme_perception_score", 1 / 20), "mmbench_en_dev": ("gpt_eval_score", None), "seedbench": ("seed_image", None),
    "gqa": ("exact_match", None), "scienceqa_img": ("exact_match", None), "mmmu_val": ("mmmu_acc", None),
    "mathvista_testmini": ("gpt_eval_score", None), "ai2d": ("exact_match", None), "chartqa": ("relaxed_overall", None),
    "ocrbench": ("ocrbench_accuracy", None), "textvqa_val": ("exact_match", None), "docvqa_val": ("anls", None),
    "mmvp": ("mmvp_pair_accuracy", None), "realworldqa": ("exact_match", None), "cv_bench_2d": ("cv_bench_acc", None),
    "cv_bench_3d": ("cv_bench_acc", None), "pope": ("pope_f1_score", None), "mmstar": ("average", None),
    "infovqa_val": ("anls", None), "hallusion_bench_image": ("aAcc", None),
}
ANLS_TASKS = {"docvqa_val", "infovqa_val"}


def tasks(judged: bool = False, extra: bool = True) -> list[str]:
    out = [t for ts in CATEGORIES.values() for t in ts] + ([t for ts in EXTRA.values() for t in ts] if extra else [])
    return [t for t in out if judged or t not in JUDGED]


def category_of(task: str) -> str | None:
    for c, ts in {**CATEGORIES, **EXTRA}.items():
        if task in ts:
            return c
    return None


def to_100(task: str, value: float) -> float:
    """Headline value on a 0-100 scale (fractions x 100; MME perception / 20)."""
    scale = HEADLINE.get(task, ("", None))[1]
    if scale is not None:
        return float(value) * scale
    return float(value) * 100.0 if abs(float(value)) <= 1.0 else float(value)


def headlines(root: str) -> dict[str, float]:
    """``{task: headline score on 0-100}`` from every lmms-eval ``*_results.json`` under ``root`` (later files win)."""
    import glob
    import json
    import os

    out: dict[str, float] = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "*_results.json"), recursive=True)):
        with open(path) as f:
            res = json.load(f).get("results", {})
        for task, metrics in res.items():
            key = HEADLINE.get(task, (None,))[0]
            val = metrics.get(f"{key},none", metrics.get(key)) if key else None
            if isinstance(val, (int, float)):
                out[task] = to_100(task, val)
    return out


def summarize(heads: dict[str, float], judged: bool = False) -> dict[str, float]:
    """Category means and the overall mean (the four core categories) of one model's headline scores; a category
    (or the overall) is reported only when every one of its benchmarks that the run includes is present."""
    out: dict[str, float] = {}
    for c, ts in {**CATEGORIES, **EXTRA}.items():
        need = [t for t in ts if judged or t not in JUDGED]
        if need and all(t in heads for t in need):
            out[c] = sum(heads[t] for t in need) / len(need)
    core = [t for ts in CATEGORIES.values() for t in ts if judged or t not in JUDGED]
    if all(t in heads for t in core):
        out["overall"] = sum(heads[t] for t in core) / len(core)
    return out


def anls(prediction: str, references: list[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity of one answer (Biten et al., ICCV 2019), as DocVQA scores it."""

    def lev(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    p = " ".join(str(prediction).strip().lower().split())
    best = 0.0
    for r in references:
        r = " ".join(str(r).strip().lower().split())
        n = max(len(p), len(r)) or 1
        s = 1.0 - lev(p, r) / n
        best = max(best, s if s >= threshold else 0.0)
    return best


def item_score(task: str, rec: dict[str, Any]) -> float | None:
    """Per-question score from an lmms-eval ``--log_samples`` record (for paired bootstrap / McNemar)."""
    if task in ANLS_TASKS:
        refs = rec.get("target")
        refs = refs if isinstance(refs, list) else [refs]
        pred = (rec.get("filtered_resps") or rec.get("resps") or [""])[0]
        pred = pred[0] if isinstance(pred, list) else pred
        return anls(pred, [r for r in refs if r is not None]) if refs else None
    metric = HEADLINE.get(task, ("", None))[0]
    keys = [metric] + [k for k in rec if k not in (metric,)]
    for k in keys:
        v = rec.get(k)
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, (int, float)) and k not in ("doc_id",):
            return float(v)
        if isinstance(v, dict):
            for field in ("score", "is_correct", "correct", "acc"):
                if field in v and isinstance(v[field], (int, float, bool)):
                    return float(v[field])
            if "pred" in v and "answer" in v:
                return float(str(v["pred"]).strip().lower() == str(v["answer"]).strip().lower())
            if "parsed_pred" in v and "answer" in v:  # MMMU (multiple choice exact; open questions approximated)
                ans = v["answer"] if isinstance(v["answer"], list) else [v["answer"]]
                return float(str(v["parsed_pred"]).strip().lower() in {str(a).strip().lower() for a in ans})
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="print the lmms-eval task list of the general suite")
    p.add_argument("--judged", action="store_true", help="include the GPT-judged benchmarks (needs OPENAI_API_KEY)")
    p.add_argument("--no_extra", action="store_true")
    a = p.parse_args()
    print(",".join(tasks(judged=a.judged, extra=not a.no_extra)))


if __name__ == "__main__":
    main()
