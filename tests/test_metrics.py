"""Official metric implementations."""

import math

import pytest

from onboard_vlm.eval.benchmarks import score_cvbench, score_egoschema, strip_answer_prefix
from onboard_vlm.eval.metrics import llm_judge as J
from onboard_vlm.eval.metrics.text import anls, average_precision, parse_choice
from onboard_vlm.eval.metrics.vqa import evalai_normalize, vqa_soft_accuracy


def test_vqa_accuracy_leave_one_out():
    assert vqa_soft_accuracy("cup", ["cup"] * 3 + ["mug"] * 7) == pytest.approx(1.0 * 3 / 10 * (2 / 3) / (2 / 3) * (2 / 3) + 7 / 10)
    # 2 matching annotators: 2 subsets see 1 match (1/3), 8 subsets see 2 matches (2/3) -> 0.6
    assert vqa_soft_accuracy("cup", ["cup"] * 2 + ["mug"] * 8) == pytest.approx(0.6)
    assert vqa_soft_accuracy("cup", ["cup"] * 10) == 1.0
    assert vqa_soft_accuracy("plate", ["cup"] * 10) == 0.0


def test_vqa_normalisation():
    assert evalai_normalize("Two dogs.") == "2 dogs"
    assert evalai_normalize("The Cup!") == "cup"
    assert evalai_normalize("dont know") == "don't know"
    assert vqa_soft_accuracy("Two.", ["2"] * 5 + ["two"] * 5) == 1.0
    assert vqa_soft_accuracy("Unanswerable", ["unanswerable"] * 10) == 1.0


def test_anls():
    assert anls("barclays center", ["Barclays Center"]) == 1.0
    assert anls("barclay center", ["barclays center"]) == pytest.approx(1 - 1 / 15)
    assert anls("something else", ["barclays center"]) == 0.0
    assert strip_answer_prefix("Answer: Barclays Center.") == "Barclays Center"


def test_parse_choice():
    assert parse_choice("B") == 1
    assert parse_choice("(C) the third one") == 2
    assert parse_choice("The answer is D.") == 3
    assert parse_choice("2", num_choices=5) == 2
    assert parse_choice("a blue car", choices=["a red car", "a blue car"]) == 1
    assert parse_choice("no idea", choices=["x", "y"]) == -1


def test_average_precision_matches_sklearn():
    scores = [0.9, 0.8, 0.8, 0.4, 0.3, 0.1]
    labels = [1, 0, 1, 1, 0, 0]
    sk = pytest.importorskip("sklearn.metrics")
    assert average_precision(scores, labels) == pytest.approx(sk.average_precision_score(labels, scores))
    assert math.isnan(average_precision([0.1], [0]))


def test_cvbench_official_overall():
    recs = []
    for src, n_correct, n in [("ADE20K", 3, 4), ("COCO", 1, 2), ("Omni3D", 2, 2)]:
        for i in range(n):
            recs.append({"prediction": "A" if i < n_correct else "B", "choices": ["x", "y"], "answer_index": 0, "category": "Count", "meta": {"source": src}})
    m = score_cvbench(recs)
    assert m["accuracy_2d"] == pytest.approx((0.75 + 0.5) / 2)
    assert m["accuracy_3d"] == 1.0
    assert m["cv_bench_overall"] == pytest.approx(((0.75 + 0.5) / 2 + 1.0) / 2)


def test_egoschema_only_scores_public_answers():
    recs = [
        {"prediction": "A", "choices": list("abcde"), "answer_index": 0},
        {"prediction": "B", "choices": list("abcde"), "answer_index": -1},
    ]
    m = score_egoschema(recs)
    assert m["num_scored"] == 1 and m["accuracy"] == 1.0


def test_judge_prompts_and_official_aggregations():
    qid, sys_p, user = J.egoblind_prompts("q1", "Where?", ["Left.", "", "On the left", None], "left")
    assert "Correct Answer1: nan" in user and "{'pred': 'yes', 'score': 4}" in user and "four standard answers" in sys_p
    _, _, user2 = J.egotextvqa_prompts("q2", "Which?", "Barclays\nCenter", "BARCLAYS")
    assert "Correct Answer: barclayscenter" in user2 and "Predicted Answer: barclays" in user2
    assert J.parse_judgement("{'pred': 'yes', 'score': 4}") == {"pred": "yes", "score": 4.0}
    assert J.parse_judgement("Sure! pred: no, score: 2")["pred"] == "no"
    j = {"a": {"pred": "yes", "score": 5}, "b": {"pred": "no", "score": 1}, "c": {"pred": "yes", "score": 3}}
    eb = J.egoblind_metrics(j, {"a": "Navigation.", "b": "navigation", "c": "safety warnings"})
    assert eb["navigation"]["accuracy"] == 0.5 and eb["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert eb["overall"]["average_score"] == pytest.approx(3.0)
    et = J.egotextvqa_metrics(j, {"a": "location", "b": "location", "c": "others"})
    assert et["location"]["accuracy"] == 0.5 and et["overall"]["average_score"] == pytest.approx(3.0)


def test_cached_judge_resumes(tmp_path):
    calls = []

    def backend(system, user):
        calls.append(user)
        return "{'pred': 'yes', 'score': 5}"

    path = str(tmp_path / "cache.jsonl")
    items = [J.wearvqa_prompts("q1", "What?", "x", "x")]
    assert J.CachedJudge(backend, path).judge(items)["q1"]["pred"] == "yes"
    J.CachedJudge(backend, path).judge(items)
    assert len(calls) == 1
