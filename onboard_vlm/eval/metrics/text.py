"""Deterministic text metrics: ANLS, exact match, MCQ parsing, average precision."""

from __future__ import annotations

import re
from typing import Sequence

from onboard_vlm.eval.metrics.vqa import evalai_normalize


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(prediction: str, references: Sequence[str], threshold: float = 0.5) -> float:
    """Average Normalised Levenshtein Similarity (Biten et al., ST-VQA), max over references.

    ``s = 1 - NL(pred, ref)`` if ``NL < threshold`` else 0, with lower-cased, whitespace-collapsed strings.
    """
    p = " ".join(prediction.lower().split())
    best = 0.0
    for r in references:
        r = " ".join(str(r).lower().split())
        if not p and not r:
            return 1.0
        nl = _levenshtein(p, r) / max(len(p), len(r), 1)
        best = max(best, 1.0 - nl if nl < threshold else 0.0)
    return best


def normalized_exact_match(prediction: str, references: Sequence[str]) -> float:
    """1 if the EvalAI-normalised prediction equals any normalised reference."""
    p = evalai_normalize(prediction)
    return float(any(p == evalai_normalize(str(r)) for r in references))


def contains_match(prediction: str, references: Sequence[str]) -> float:
    """1 if any normalised reference appears as a whole-word span of the normalised prediction (lenient proxy)."""
    p = f" {evalai_normalize(prediction)} "
    return float(any(f" {evalai_normalize(str(r))} " in p for r in references if str(r).strip()))


_LETTER_PATTERNS = [
    re.compile(r"^\s*\(?([A-J])\)?(?:[.:)\s]|$)"),  # "B", "(B)", "B." at the start
    re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-J])\)?\b", re.IGNORECASE),
    re.compile(r"\(([A-J])\)"),
    re.compile(r"\b([A-J])\b"),
]


def parse_choice(text: str, choices: Sequence[str] | None = None, num_choices: int | None = None) -> int:
    """Map a generated answer to a choice index (``-1`` if unparseable).

    Tries letter patterns first (start of answer, "answer is X", "(X)", isolated capital), then a
    digit 0..n-1 / 1..n, then an exact / substring match against the option texts.
    """
    n = num_choices or (len(choices) if choices else 5)
    letters = "ABCDEFGHIJ"[:n]
    t = text.strip()
    for pat in _LETTER_PATTERNS:
        m = pat.search(t)
        if m and m.group(1) in letters:
            return letters.index(m.group(1))
    m = re.match(r"^\s*(\d)\b", t)
    if m and int(m.group(1)) < n:
        return int(m.group(1))
    if choices:
        low = t.lower().strip(" .")
        for i, c in enumerate(choices):
            if low == str(c).lower().strip(" ."):
                return i
        hits = [i for i, c in enumerate(choices) if str(c).lower().strip(" .") in low]
        if len(hits) == 1:
            return hits[0]
    return -1


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Average precision as in ``sklearn.metrics.average_precision_score`` (ties grouped by threshold).

    ``AP = sum_n (R_n - R_{n-1}) P_n`` over distinct score thresholds in decreasing order.
    """
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    total_pos = sum(1 for _, y in pairs if y)
    if total_pos == 0:
        return float("nan")
    ap, tp, fp, prev_recall, i = 0.0, 0, 0, 0.0, 0
    while i < len(pairs):
        s = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == s:
            tp += int(bool(pairs[i][1]))
            fp += int(not pairs[i][1])
            i += 1
        recall = tp / total_pos
        ap += (recall - prev_recall) * (tp / (tp + fp))
        prev_recall = recall
    return ap
