"""Benchmark specifications: prompt, generation length, official metric, submission export.

Each scorer receives a list of prediction records (the dataset sample fields + ``prediction``)
and returns a JSON-serialisable metrics dict. See ``docs/metrics.md`` for the exact definitions.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from onboard_vlm.eval.metrics import llm_judge as J
from onboard_vlm.eval.metrics.text import anls, average_precision, contains_match, normalized_exact_match, parse_choice
from onboard_vlm.eval.metrics.vqa import vqa_soft_accuracy

log = logging.getLogger(__name__)
Records = list[dict[str, Any]]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _by(records: Records, key: Callable[[dict[str, Any]], Any], value: Callable[[dict[str, Any]], float]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in records:
        groups[str(key(r))].append(value(r))
    return {k: {"total": len(v), "accuracy": _mean(v)} for k, v in sorted(groups.items())}


def strip_answer_prefix(text: str) -> str:
    """'Answer: foo' -> 'foo' (EgoTextVQA's prompt asks for that format); also trims whitespace/periods."""
    m = re.search(r"answer\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    return (m.group(1) if m else text).strip().strip(".").strip()


def _first_line(text: str) -> str:
    return text.strip().split("\n")[0].strip()


# --------------------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------------------
def score_vqa(records: Records) -> dict[str, Any]:
    """VQA soft accuracy (10 annotators) overall and per answer type."""
    scored = [r for r in records if r.get("answers")]
    for r in scored:
        r["vqa_acc"] = vqa_soft_accuracy(r["prediction"], r["answers"])
    out: dict[str, Any] = {"num_scored": len(scored), "vqa_accuracy": _mean([r["vqa_acc"] for r in scored])}
    if any(r.get("answer_type") for r in scored):
        out["by_answer_type"] = _by(scored, lambda r: r.get("answer_type"), lambda r: r["vqa_acc"])
    return out


def score_vizwiz(records: Records) -> dict[str, Any]:
    """VizWiz-VQA: VQA accuracy (+ per answer type) and answerability average precision."""
    out = score_vqa(records)
    lab = [(r["answerability_score"], int(r["answerable"])) for r in records if r.get("answerability_score") is not None and r.get("answerable") is not None]
    if lab:
        out["answerability_ap"] = average_precision([s for s, _ in lab], [y for _, y in lab])
        out["answerability_ap_num"] = len(lab)
    return out


def score_mcq(records: Records) -> dict[str, Any]:
    scored = [r for r in records if r.get("answer_index") is not None and r["answer_index"] >= 0]
    for r in records:
        r["pred_index"] = parse_choice(r["prediction"], r.get("choices"))
    for r in scored:
        r["correct"] = float(r["pred_index"] == r["answer_index"])
    return {
        "num_scored": len(scored),
        "num_unparsed": sum(r["pred_index"] < 0 for r in records),
        "accuracy": _mean([r["correct"] for r in scored]),
    }


def score_egoschema(records: Records) -> dict[str, Any]:
    """EgoSchema 5-way accuracy on questions with public answers (the 500-question subset)."""
    out = score_mcq(records)
    out["note"] = "full-set (5,031) accuracy requires the Kaggle / validation-server submission written by export"
    return out


def score_cvbench(records: Records) -> dict[str, Any]:
    """CV-Bench: per-source accuracy and the official ``((ADE20K + COCO)/2 + Omni3D) / 2`` overall score."""
    out = score_mcq(records)
    scored = [r for r in records if "correct" in r]
    src = _by(scored, lambda r: (r.get("meta") or {}).get("source"), lambda r: r["correct"])
    task = _by(scored, lambda r: r.get("category"), lambda r: r["correct"])
    acc = lambda s: src.get(s, {}).get("accuracy", float("nan"))  # noqa: E731
    acc_2d = (acc("ADE20K") + acc("COCO")) / 2
    acc_3d = acc("Omni3D")
    out.update(by_source=src, by_task=task, accuracy_2d=acc_2d, accuracy_3d=acc_3d, cv_bench_overall=(acc_2d + acc_3d) / 2)
    return out


def _judge(judge: J.CachedJudge | None, items: list[tuple[str, str, str]]) -> dict[str, dict[str, Any]] | None:
    if judge is None:
        log.warning("no judge configured: LLM-judge (official) metrics are skipped; only deterministic proxies reported")
        return None
    return judge.judge(items)


def score_egoblind(records: Records, judge: J.CachedJudge | None = None) -> dict[str, Any]:
    """Official: GPT-4o-mini judge vs. the 4 reference answers -> accuracy + mean score per question type."""
    out: dict[str, Any] = {
        "proxy_exact_match_any_ref": _mean([normalized_exact_match(r["prediction"], r["answers"]) for r in records]),
        "proxy_contains_any_ref": _mean([contains_match(r["prediction"], r["answers"]) for r in records]),
    }
    j = _judge(judge, [J.egoblind_prompts(r["sample_id"], r["question"], r["answers"], r["prediction"]) for r in records])
    if j is not None:
        out["llm_judge"] = J.egoblind_metrics(j, {r["sample_id"]: r.get("category") or "" for r in records})
    return out


def score_egotextvqa(records: Records, judge: J.CachedJudge | None = None) -> dict[str, Any]:
    """Official: GPT-4o-mini judge -> accuracy + mean 0-5 score per question type. ANLS/EM are auxiliary."""
    for r in records:
        r["clean_prediction"] = strip_answer_prefix(r["prediction"])
    out: dict[str, Any] = {
        "aux_anls": _mean([anls(r["clean_prediction"], r["answers"]) for r in records]),
        "aux_exact_match": _mean([normalized_exact_match(r["clean_prediction"], r["answers"]) for r in records]),
    }
    j = _judge(judge, [J.egotextvqa_prompts(r["sample_id"], r["question"], r["answers"][0], r["prediction"]) for r in records])
    if j is not None:
        out["llm_judge"] = J.egotextvqa_metrics(j, {r["sample_id"]: r.get("category") or "others" for r in records})
    return out


def score_wearvqa(records: Records, judge: J.CachedJudge | None = None) -> dict[str, Any]:
    """LLM-judge accuracy sliced by image-quality issue, domain and task type (+ deterministic proxies)."""
    slices = {r["sample_id"]: {k: (r.get("meta") or {}).get(k) for k in ("quality_issues", "domain", "task_type") if (r.get("meta") or {}).get(k) is not None} for r in records}
    for r in records:
        r["em"] = normalized_exact_match(r["prediction"], r["answers"])
        r["contains"] = contains_match(r["prediction"], r["answers"])
    out: dict[str, Any] = {"proxy_exact_match": _mean([r["em"] for r in records]), "proxy_contains": _mean([r["contains"] for r in records])}
    for dim in ("quality_issues", "domain", "task_type"):
        groups: dict[str, list[float]] = defaultdict(list)
        for r in records:
            vals = (r.get("meta") or {}).get(dim)
            for v in [vals] if isinstance(vals, str) or vals is None else vals:
                groups[str(v)].append(r["contains"])
        out[f"proxy_contains_by_{dim}"] = {k: {"total": len(v), "accuracy": _mean(v)} for k, v in sorted(groups.items())}
    j = _judge(judge, [J.wearvqa_prompts(r["sample_id"], r["question"], r["answers"][0] if r["answers"] else "", r["prediction"]) for r in records])
    if j is not None:
        out["llm_judge"] = J.binary_judge_metrics(j, slices)
        out["llm_judge"]["note"] = "judge prompt is not the paper's verbatim prompt; see eval/metrics/llm_judge.py"
    return out


# --------------------------------------------------------------------------------------
# Official-format exports
# --------------------------------------------------------------------------------------
def _meta(r: dict[str, Any], key: str, default: Any) -> Any:
    return (r.get("meta") or {}).get(key, default)


def export_vizwiz(records: Records, out_dir: str) -> None:
    """EvalAI VizWiz files: VQA ``[{image, answer}]`` and answerability ``[{image, answerable}]``."""
    with open(os.path.join(out_dir, "vizwiz_vqa_submission.json"), "w") as f:
        json.dump([{"image": _meta(r, "image", r["sample_id"] + ".jpg"), "answer": r["prediction"]} for r in records], f)
    if all(r.get("answerability_score") is not None for r in records):
        with open(os.path.join(out_dir, "vizwiz_answerability_submission.json"), "w") as f:
            json.dump([{"image": _meta(r, "image", r["sample_id"] + ".jpg"), "answerable": float(r["answerability_score"])} for r in records], f)


def export_textvqa(records: Records, out_dir: str) -> None:
    """EvalAI TextVQA file: ``[{question_id, answer}]``."""
    with open(os.path.join(out_dir, "textvqa_submission.json"), "w") as f:
        json.dump([{"question_id": _meta(r, "question_id", r["sample_id"]), "answer": r["prediction"]} for r in records], f)


def export_egoschema(records: Records, out_dir: str) -> None:
    """Kaggle ``egoschema-public`` CSV (``q_uid,answer``) and ``{q_uid: int}`` JSON for validate.py / the server."""
    rows = [(_meta(r, "q_uid", r["sample_id"]), max(r.get("pred_index", -1), 0)) for r in records]
    with open(os.path.join(out_dir, "egoschema_kaggle.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["q_uid", "answer"])
        w.writerows(rows)
    with open(os.path.join(out_dir, "egoschema_submission.json"), "w") as f:
        json.dump(dict(rows), f)


def export_egoblind(records: Records, out_dir: str) -> None:
    """Prediction JSONL consumed by the official ``EgoBlind/eval.py`` (``{"question_id", "pred"}`` per line)."""
    with open(os.path.join(out_dir, "egoblind_pred.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps({"question_id": _meta(r, "question_id", r["sample_id"]), "pred": r["prediction"]}) + "\n")


def export_egotextvqa(records: Records, out_dir: str, model_name: str = "onboard_vlm") -> None:
    """Prediction file in the layout read by the official ``EgoTextVQA/gpt_eval.py``."""
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_subset[_meta(r, "subset", "unknown")].append({
            "question_id": _meta(r, "question_id", r["sample_id"]), "video_id": _meta(r, "video_id", ""), "question": r["question"],
            "correct_answer": r["answers"][0], "question_type": r.get("category"), model_name: r["prediction"],
        })
    for subset, data in by_subset.items():
        with open(os.path.join(out_dir, f"egotextvqa_{subset}_pred.json"), "w") as f:
            json.dump([{"dataset": f"egotextvqa_{subset}", "split": "test", "task setting": "", "data": data}], f)


# --------------------------------------------------------------------------------------
# Registry of benchmarks
# --------------------------------------------------------------------------------------
@dataclass
class BenchmarkSpec:
    name: str
    prompt: str
    media: str
    max_new_tokens: int
    scorer: Callable[..., dict[str, Any]]
    export: Callable[[Records, str], None] | None = None
    postprocess: Callable[[str], str] = _first_line
    uses_judge: bool = False
    answerability: bool = False
    notes: list[str] = field(default_factory=list)


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "vizwiz_vqa": BenchmarkSpec("vizwiz_vqa", "vizwiz_vqa", "image", 16, score_vizwiz, export_vizwiz, answerability=True),
    "textvqa": BenchmarkSpec("textvqa", "textvqa", "image", 16, score_vqa, export_textvqa),
    "wearvqa": BenchmarkSpec("wearvqa", "wearvqa", "image", 64, score_wearvqa, None, lambda s: s.strip(), uses_judge=True),
    "egoblind": BenchmarkSpec("egoblind", "egoblind", "video", 64, score_egoblind, export_egoblind, lambda s: s.strip(), uses_judge=True),
    "egotextvqa": BenchmarkSpec("egotextvqa", "egotextvqa", "video", 64, score_egotextvqa, export_egotextvqa, lambda s: s.strip(), uses_judge=True),
    "egoschema": BenchmarkSpec("egoschema", "egoschema", "video", 8, score_egoschema, export_egoschema),
    "cvbench": BenchmarkSpec("cvbench", "cvbench", "image", 8, score_cvbench, None),
}
