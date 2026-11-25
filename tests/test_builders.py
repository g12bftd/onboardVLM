"""Builders on synthetic raw downloads in the official formats -> LMDB -> loader round trip."""

import csv
import json
import os
import subprocess
import sys

import pytest
from conftest import ROOT, write_video
from PIL import Image

from onboard_vlm.data.datasets import LMDBVQADataset

LMDB_DIR = os.path.join(ROOT, "make_datasets", "lmdb")


def run_builder(script: str, *args: str) -> None:
    env = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, os.path.join(LMDB_DIR, script), *args], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]


def _img(path: str, size=(64, 48)) -> None:
    Image.new("RGB", size, (120, 30, 200)).save(path)


def test_vizwiz_builder(tmp_path):
    raw = tmp_path / "raw"
    (raw / "train").mkdir(parents=True)
    (raw / "Annotations").mkdir()
    anns = []
    for i in range(3):
        name = f"VizWiz_train_{i:08d}.jpg"
        _img(str(raw / "train" / name))
        anns.append({
            "image": name, "question": f"what is {i}?", "answer_type": "other", "answerable": int(i != 2),
            "answers": [{"answer": "unanswerable" if i == 2 else "a cup", "answer_confidence": "yes"}] * 10,
        })
    json.dump(anns, open(raw / "Annotations" / "train.json", "w"))
    run_builder("make_vizwiz.py", "--input_path", str(raw), "--output_path", str(tmp_path / "out"), "--splits", "train")
    ds = LMDBVQADataset(str(tmp_path / "out"), "train")
    assert len(ds) == 3
    s = ds[2]
    assert s["image"].shape == (3, 48, 64) and s["answerable"] == 0 and len(s["answers"]) == 10
    assert s["meta"]["image"] == "VizWiz_train_00000002.jpg"


def test_textvqa_builder_dedups_images_in_image_mode(tmp_path):
    raw = tmp_path / "raw"
    (raw / "train_images").mkdir(parents=True)
    _img(str(raw / "train_images" / "abc.jpg"))
    data = [{"question_id": q, "question": "what?", "image_id": "abc", "answers": ["x"] * 10, "image_classes": ["Sign"]} for q in (1, 2)]
    json.dump({"data": data}, open(raw / "TextVQA_0.5.1_val.json", "w"))
    run_builder("make_textvqa.py", "--input_path", str(raw), "--output_path", str(tmp_path / "out"), "--splits", "val")
    assert len(LMDBVQADataset(str(tmp_path / "out"), "val")) == 2
    img_ds = LMDBVQADataset(str(tmp_path / "out"), "val", mode="image")
    assert len(img_ds) == 1 and img_ds[0]["sample_id"] == "abc"


def test_egoblind_builder_samples_frames_before_question(tmp_path):
    raw = tmp_path / "raw"
    (raw / "videos").mkdir(parents=True)
    write_video(str(raw / "videos" / "00923.mp4"), num_frames=40, fps=10)
    with open(raw / "test_half_release.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "video_name", "question", "answer0", "answer1", "answer2", "answer3", "type", "start-time/s"])
        w.writerow(["v_00923_1", "923", "How do I open this gate?", "Swipe.", "Card.", "", "", "tool use", "1.98"])
    run_builder("make_egoblind.py", "--input_path", str(raw), "--output_path", str(tmp_path / "out"), "--num_frames", "4")
    s = LMDBVQADataset(str(tmp_path / "out"), "test")[0]
    assert s["frames"].shape[0] == 4 and max(s["frame_timestamps"]) <= 1.98 + 1e-6
    assert s["answers"] == ["Swipe.", "Card.", "", ""] and s["category"] == "tool use"
    # frame brightness grows with time: the last sampled frame is the one at ~1.98 s (frame ~19-20)
    assert 140 <= s["frames"][-1].float().mean() <= 175


def test_egoblind_builder_full_test_and_train_csv(tmp_path):
    """test.csv (full test set) is preferred over the released half, train.csv is picked up, and questions
    without any reference answer (7 in the official test.csv) are skipped."""
    raw = tmp_path / "raw"
    (raw / "videos" / "split_1").mkdir(parents=True)
    for v in (923, 10):
        write_video(str(raw / "videos" / "split_1" / f"{v:05d}.mp4"), num_frames=20, fps=10)
    header = ["question_id", "video_name", "question", "answer0", "answer1", "answer2", "answer3", "type", "start-time/s"]
    for name, rows in {
        "test.csv": [["v_00923_1", "923", "q1?", "a", "b", "", "", "navigation", "1.0"],
                     ["v_00923_2", "923", "q2?", "", "", "", "", "navigation", "1.5"]],
        "test_half_release.csv": [["v_00923_1", "923", "q1?", "a", "b", "", "", "navigation", "1.0"]],
        "train.csv": [["v_00010_1", "10", "q3?", "c", "", "", "", "tool use", "0.5"]],
    }.items():
        with open(raw / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    run_builder("make_egoblind.py", "--input_path", str(raw), "--video_dir", str(raw), "--output_path", str(tmp_path / "out"),
                "--num_frames", "2")
    test, train = LMDBVQADataset(str(tmp_path / "out"), "test"), LMDBVQADataset(str(tmp_path / "out"), "train")
    assert [s["sample_id"] for s in test] == ["v_00923_1"] and [s["sample_id"] for s in train] == ["v_00010_1"]


def test_verify_videos_detects_incomplete_downloads(tmp_path):
    script = os.path.join(ROOT, "make_datasets", "download", "verify_videos.py")
    vids = tmp_path / "videos" / "part"
    vids.mkdir(parents=True)
    for stem in ("00923", "uid1"):
        (vids / f"{stem}.mp4").write_bytes(b"x")
    with open(tmp_path / "test.csv", "w", newline="") as f:
        csv.writer(f).writerows([["video_name"], ["923"], ["923"]])
    json.dump([{"q_uid": "uid1"}, {"q_uid": "uid2"}], open(tmp_path / "questions.json", "w"))
    json.dump({"uid1": 0}, open(tmp_path / "subset.json", "w"))

    def verify(*args: str) -> int:
        return subprocess.run([sys.executable, "-I", script, "--name", "t", "--video_dir", str(tmp_path / "videos"), *args],
                              capture_output=True, text=True).returncode

    assert verify("--ids", str(tmp_path / "test.csv"), "--key", "video_name") == 0          # zero-padded stem found
    assert verify("--ids", str(tmp_path / "questions.json"), "--key", "q_uid") == 1           # uid2 missing
    assert verify("--ids", str(tmp_path / "questions.json"), "--key", "q_uid", "--max_missing", "1",
                  "--required", str(tmp_path / "subset.json")) == 0                            # tolerated; subset complete
    json.dump({"uid2": 0}, open(tmp_path / "subset.json", "w"))
    assert verify("--ids", str(tmp_path / "questions.json"), "--key", "q_uid", "--max_missing", "1",
                  "--required", str(tmp_path / "subset.json")) == 1                            # a required video missing


def test_egoschema_builder_subset_and_full(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    qs = [{"q_uid": f"uid{i}", "google_drive_id": "x", "question": "q?", **{f"option {k}": f"opt{k}" for k in range(5)}} for i in range(2)]
    json.dump(qs, open(raw / "questions.json", "w"))
    json.dump({"uid0": 3}, open(raw / "subset_answers.json", "w"))
    for i in range(2):
        write_video(str(raw / f"uid{i}.mp4"), num_frames=20)
    run_builder("make_egoschema.py", "--input_path", str(raw), "--output_path", str(tmp_path / "out"), "--num_frames", "4")
    full = LMDBVQADataset(str(tmp_path / "out"), "test")
    sub = LMDBVQADataset(str(tmp_path / "out"), "subset")
    assert len(full) == 2 and len(sub) == 1
    assert sub[0]["answer_index"] == 3 and len(sub[0]["choices"]) == 5
    assert sorted(s["answer_index"] for s in (full[0], full[1])) == [-1, 3]


def test_egotextvqa_builder(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_video(str(raw / "3.mp4"), num_frames=40)
    with open(raw / "egotextvqa_outdoor_annotation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "question_id", "question_type", "timestamp (s)", "question", "correct_answer"])
        w.writerow(["3", "1.0", "Location", "2.5", "which center is on the left side of the road?", "barclays center"])
    run_builder("make_egotextvqa.py", "--input_path", str(raw), "--output_path", str(tmp_path / "out"), "--subset", "outdoor", "--num_frames", "3", "--resize", "96x64")
    s = LMDBVQADataset(str(tmp_path / "out"), "test")[0]
    assert s["frames"].shape == (3, 3, 64, 96) and s["category"] == "location" and s["meta"]["subset"] == "outdoor"


def test_cvbench_and_wearvqa_builders(tmp_path):
    datasets = pytest.importorskip("datasets")
    img = Image.new("RGB", (80, 60), (10, 200, 10))
    d = datasets.Dataset.from_dict({
        "idx": [0, 1], "type": ["2D", "3D"], "task": ["Count", "Depth"], "image": [img, img],
        "question": ["how many?", "closer?"], "choices": [["1", "2"], ["a", "b"]], "answer": ["(B)", "(A)"],
        "prompt": ["how many? (A) 1 (B) 2", "closer? (A) a (B) b"], "source": ["COCO", "Omni3D"],
    }).cast_column("image", datasets.Image())
    (tmp_path / "cv" / "data").mkdir(parents=True)
    d.to_parquet(str(tmp_path / "cv" / "data" / "test-00000-of-00001.parquet"))
    run_builder("make_cvbench.py", "--input_path", str(tmp_path / "cv"), "--output_path", str(tmp_path / "out_cv"))
    s = LMDBVQADataset(str(tmp_path / "out_cv"), "test")[0]
    assert s["answer_index"] == 1 and s["meta"]["source"] == "COCO" and s["category"] == "Count"

    (tmp_path / "wv").mkdir()
    _img(str(tmp_path / "wv" / "a.jpg"))
    with open(tmp_path / "wv" / "ann.jsonl", "w") as f:
        f.write(json.dumps({"id": 7, "image": "a.jpg", "question": "what brand?", "answer": "acme", "quality_issues": ["blur", "low light"], "domain": "text", "task_type": "ocr"}) + "\n")
    run_builder("make_wearvqa.py", "--input_path", str(tmp_path / "wv" / "ann.jsonl"), "--output_path", str(tmp_path / "out_wv"))
    s = LMDBVQADataset(str(tmp_path / "out_wv"), "test")[0]
    assert s["sample_id"] == "wearvqa_7" and s["meta"]["quality_issues"] == ["blur", "low light"]
