"""LLM-as-judge metrics (official protocols of EgoBlind, EgoTextVQA, and WearVQA-style judging).

EgoBlind and EgoTextVQA publish their judge scripts; their prompts are reproduced *verbatim*
below (sources: https://github.com/doc-doc/EgoBlind/blob/main/eval.py and
https://github.com/zhousheng97/EgoTextVQA/blob/main/gpt_eval.py), both with ``gpt-4o-mini``.
Their aggregation rules are reproduced in :func:`egoblind_metrics` / :func:`egotextvqa_metrics`.

WearVQA (NeurIPS 2025 D&B) also uses an LLM judge, but its prompt is in the paper appendix
and was not accessible when this was written; :data:`WEARVQA_JUDGE_*` is a binary-correctness
prompt in the same style. Paste the paper's prompt there to report official numbers.

Judgements are cached in a JSONL file keyed by question id, so runs can be resumed (as the
official scripts do).
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Callable, Iterable, Protocol, Sequence

# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------
EGOBLIND_JUDGE_SYSTEM = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
    "------"
    "##INSTRUCTIONS: "
    "- Focus on meaningful matches: Assess whether the predicted answer and the correct answer have a meaningful match, not just literal word-for-word matches.\n"
    "- Criteria for Correctness: The predicted answer is considered correct if it reasonably matches any of the four standard answers, recognizing that synonyms or varied expressions that convey the same meaning are acceptable.\n"
    "- Allow for Paraphrasing: Understand that different wording that conveys the same fundamental idea is valid.\n"
    "- Flexibility in Evaluation: Use judgment to decide if variations in the predicted answer still correctly address the question, even if they do not directly replicate the correct answer's phrasing.\n"
    "Answers should be tailored for blind individuals, avoiding reliance on any visual descriptions."
)
EGOBLIND_JUDGE_USER = (
    "Please evaluate the following video-based question-answer pair:\n\n"
    "Question: {question}\n"
    "Correct Answer0: {a0}\n"
    "Correct Answer1: {a1}\n"
    "Correct Answer2: {a2}\n"
    "Correct Answer3: {a3}\n"
    "Predicted Answer: {pred}\n\n"
    "Provide your evaluation only as a yes/no and score where the score is a float value between 0 and 5, with 5 indicating the highest meaningful match. "
    "Generate the response in the form of a Python dictionary string with keys 'pred' and 'score'. For example: {{'pred': 'yes', 'score': 4}}."
)

EGOTEXTVQA_JUDGE_SYSTEM = (
    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
    "------"
    "##INSTRUCTIONS: "
    "- Focus on the meaningful match between the predicted answer and the correct answer. "
    "Please note that not only matches of noun phrases between answers, but also matches of prepositional phrases. "
    'For example, "at the car wash on your right" does not exactly match "car wash". '
    "\"at the gas station beside the sign 'gas sale'\" does not exactly match \"gas station\".\n"
    "- Consider synonyms or paraphrases as valid matches. "
    "Note that the predicted answer must be consistent with the string type of the correct answer, which may include phone numbers, email addresses, numbers, dates, etc. "
    'For example, the string types of "www.usps.com" and "visit their website" are inconsistent, '
    'and the string types of "9849041316" and "advertiser\'s contact number" are inconsistent.\n'
    "- Evaluate the correctness of the prediction compared to the answer."
)
EGOTEXTVQA_JUDGE_USER = (
    "Please evaluate the following video-based question-answer pair:\n\n"
    "Question: {question}\n"
    "Correct Answer: {answer}\n"
    "Predicted Answer: {pred}\n\n"
    "Provide your eval_code only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
    "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. "
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
    "For example, your response should look like this: {{'pred': 'yes', 'score': 5}}, {{'pred': 'no', 'score': 1}}."
)

# Not the official WearVQA prompt (see module docstring).
WEARVQA_JUDGE_SYSTEM = (
    "You are an impartial grader for a visual question answering benchmark recorded with smart glasses. "
    "Decide whether the predicted answer is correct given the ground-truth answer. The prediction is correct if it "
    "conveys the same essential information as the ground truth (paraphrases, synonyms and extra harmless detail are "
    "fine); it is incorrect if it is wrong, contradicts the ground truth, is a refusal, or is too vague to be useful."
)
WEARVQA_JUDGE_USER = (
    "Question: {question}\nGround-truth answer: {answer}\nPredicted answer: {pred}\n\n"
    "Respond only with a Python dictionary string with keys 'pred' ('yes' if correct, otherwise 'no') and "
    "'score' (integer 0-5). For example: {{'pred': 'yes', 'score': 5}}."
)


# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------
class JudgeBackend(Protocol):
    def __call__(self, system: str, user: str) -> str: ...


class OpenAICompatibleJudge:
    """Chat-completions client for OpenAI or any compatible server (vLLM, TGI, Azure ...).

    The API key is read from ``api_key_env`` (default ``OPENAI_API_KEY``); nothing is hard-coded.
    """

    def __init__(self, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1", api_key_env: str = "OPENAI_API_KEY", max_tokens: int = 1000, temperature: float | None = None, retries: int = 5, timeout: float = 60.0):
        self.model, self.base_url, self.max_tokens, self.temperature = model, base_url.rstrip("/"), max_tokens, temperature
        self.api_key = os.environ.get(api_key_env, "")
        self.retries, self.timeout = retries, timeout

    def __call__(self, system: str, user: str) -> str:
        import requests

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        err: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:  # pragma: no cover - network
                err = e
                time.sleep(min(60, 2**attempt))
        raise RuntimeError(f"judge request failed after {self.retries} attempts: {err}")


def parse_judgement(text: str) -> dict[str, Any]:
    """``"{'pred': 'yes', 'score': 4}"`` -> dict (``ast.literal_eval`` like the official scripts, regex fallback)."""
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            d = ast.literal_eval(m.group(0))
            if isinstance(d, dict):
                return {"pred": str(d.get("pred", "no")), "score": float(d.get("score", 0))}
        except (ValueError, SyntaxError):
            pass
    pred = "yes" if re.search(r"\byes\b", text, re.IGNORECASE) else "no"
    s = re.search(r"score\W+(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    return {"pred": pred, "score": float(s.group(1)) if s else 0.0}


class CachedJudge:
    """Runs a judge over items, caching ``{qid: {"pred", "score", "raw"}}`` in a JSONL file."""

    def __init__(self, backend: JudgeBackend, cache_path: str | None = None):
        self.backend, self.cache_path = backend, cache_path
        self.cache: dict[str, dict[str, Any]] = {}
        if cache_path and os.path.isfile(cache_path):
            with open(cache_path) as f:
                for line in f:
                    d = json.loads(line)
                    self.cache[d["qid"]] = d

    def judge(self, items: Iterable[tuple[str, str, str]]) -> dict[str, dict[str, Any]]:
        """items: ``(qid, system_prompt, user_prompt)``."""
        out = {}
        fh = open(self.cache_path, "a") if self.cache_path else None
        try:
            for qid, system, user in items:
                if qid not in self.cache:
                    raw = self.backend(system, user)
                    d = {"qid": qid, **parse_judgement(raw), "raw": raw}
                    self.cache[qid] = d
                    if fh:
                        fh.write(json.dumps(d) + "\n")
                        fh.flush()
                out[qid] = self.cache[qid]
        finally:
            if fh:
                fh.close()
        return out


# --------------------------------------------------------------------------------------
# Prompt builders + official aggregations
# --------------------------------------------------------------------------------------
def egoblind_prompts(qid: str, question: str, answers: Sequence[str], pred: str) -> tuple[str, str, str]:
    # The official script reads the CSV with pandas, so missing reference answers render as "nan".
    a = [str(x) if x not in (None, "") else "nan" for x in list(answers) + [None] * (4 - len(answers))][:4]
    return qid, EGOBLIND_JUDGE_SYSTEM, EGOBLIND_JUDGE_USER.format(question=question, a0=a[0], a1=a[1], a2=a[2], a3=a[3], pred=pred)


def egotextvqa_prompts(qid: str, question: str, answer: str, pred: str) -> tuple[str, str, str]:
    # Official preprocessing: remove newlines, lower-case answer and prediction.
    clean: Callable[[str], str] = lambda s: str(s).replace("\n", "").lower()
    return qid, EGOTEXTVQA_JUDGE_SYSTEM, EGOTEXTVQA_JUDGE_USER.format(question=question, answer=clean(answer), pred=clean(pred))


def wearvqa_prompts(qid: str, question: str, answer: str, pred: str) -> tuple[str, str, str]:
    return qid, WEARVQA_JUDGE_SYSTEM, WEARVQA_JUDGE_USER.format(question=question, answer=answer, pred=pred)


def egoblind_metrics(judgements: dict[str, dict[str, Any]], qid_to_type: dict[str, str]) -> dict[str, Any]:
    """EgoBlind ``calculate_metrics``: accuracy = #('yes') / total and mean score, per type and overall."""
    stats: dict[str, dict[str, float]] = defaultdict(lambda: {"total": 0, "yes": 0, "score": 0.0})
    for qid, j in judgements.items():
        t = (qid_to_type.get(qid) or "").strip().lower().rstrip(".")
        if not t:
            continue
        stats[t]["total"] += 1
        stats[t]["yes"] += int(str(j.get("pred", "")).lower() == "yes")
        stats[t]["score"] += float(j.get("score", 0))
    res = {t: {"total": int(d["total"]), "accuracy": d["yes"] / d["total"], "average_score": d["score"] / d["total"]} for t, d in stats.items()}
    tot = sum(d["total"] for d in stats.values())
    res["overall"] = {
        "total": int(tot),
        "accuracy": sum(d["yes"] for d in stats.values()) / tot if tot else 0.0,
        "average_score": sum(d["score"] for d in stats.values()) / tot if tot else 0.0,
    }
    return res


def egotextvqa_metrics(judgements: dict[str, dict[str, Any]], qid_to_type: dict[str, str]) -> dict[str, Any]:
    """EgoTextVQA ``calculate_score``: accuracy = yes / (yes + no) (substring test), mean integer score."""
    stats: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "yes": 0, "no": 0, "score": 0.0})
    for qid, j in judgements.items():
        t = (qid_to_type.get(qid) or "others").lower()
        p = str(j.get("pred", "no")).lower()
        if "yes" in p:
            stats[t]["yes"] += 1
        elif "no" in p:
            stats[t]["no"] += 1
        stats[t]["score"] += int(float(j.get("score", 0)))
        stats[t]["count"] += 1
    res = {}
    for t, d in stats.items():
        yn = d["yes"] + d["no"]
        res[t] = {"total": int(d["count"]), "accuracy": d["yes"] / yn if yn else 0.0, "average_score": d["score"] / d["count"] if d["count"] else 0.0}
    yes, no = sum(d["yes"] for d in stats.values()), sum(d["no"] for d in stats.values())
    cnt = sum(d["count"] for d in stats.values())
    res["overall"] = {"total": int(cnt), "accuracy": yes / (yes + no) if yes + no else 0.0, "average_score": sum(d["score"] for d in stats.values()) / cnt if cnt else 0.0}
    return res


def binary_judge_metrics(judgements: dict[str, dict[str, Any]], slices: dict[str, dict[str, Sequence[str] | str | None]]) -> dict[str, Any]:
    """Accuracy overall and per slice value (``slices[qid] = {"quality_issue": [...], "domain": "...", ...}``)."""
    acc = {qid: float(str(j.get("pred", "")).lower() == "yes") for qid, j in judgements.items()}
    res: dict[str, Any] = {"overall": {"total": len(acc), "accuracy": sum(acc.values()) / len(acc) if acc else 0.0}}
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for qid, a in acc.items():
        for dim, vals in (slices.get(qid) or {}).items():
            for v in [vals] if isinstance(vals, str) or vals is None else vals:
                by[dim][str(v)].append(a)
    for dim, groups in by.items():
        res[f"by_{dim}"] = {k: {"total": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(groups.items())}
    return res
