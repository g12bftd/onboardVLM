"""The experimental protocol: input domain, shape-exact multi-budget batching, fidelity metrics,
baselines, teacher-generated targets, evaluation runners and statistics (docs/protocol.md)."""

import collections
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from conftest import ROOT

from onboard_vlm.data.collator import DualViewCollator
from onboard_vlm.data.datasets import SyntheticVQADataset
from onboard_vlm.data.samplers import BucketedBatchSampler, ExactShapeBatchSampler, batch_size_for
from onboard_vlm.distill.fidelity import fidelity_metrics, linear_cka, per_image_fidelity
from onboard_vlm.eval import stats
from onboard_vlm.geometry import InputDomain, check_student_covers, processor_shapes, sample_transform, target_size
from onboard_vlm.interfaces import TOKEN_STRIDE
from onboard_vlm.registry import build_student

TINY = {"vision_hidden": 48, "vision_layers": 3}


def _run(script, *args):
    env = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, os.path.join(ROOT, script), *args], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-4000:]
    return r.stdout


# ------------------------------------------------------------------------------------------- domain
def test_input_domain_is_the_set_of_processor_outputs():
    d = InputDomain()
    assert d.max_patches_per_axis() == 198 and InputDomain(max_aspect=8).max_patches_per_axis() == 282
    rng = random.Random(0)
    for _ in range(300):  # any source within the aspect range maps into the domain at every budget
        h = rng.randint(50, 4000)
        w = int(h * np.exp(rng.uniform(np.log(1 / 4), np.log(4))))
        if not 1 / 4 <= w / h <= 4:
            continue
        for b in d.budgets:
            th, tw = target_size(h, w, b)
            assert d.contains(th, tw) and th % TOKEN_STRIDE == 0 and (th // 48) * (tw // 48) <= b
    assert not d.contains(96, 5040) and all(s in processor_shapes(280, 4.0) for s in d.shapes(280))


def test_students_must_cover_the_domain():
    small = build_student({"name": "gemma4_vit", "out_dim": 48, "hidden_size": 32, "num_hidden_layers": 1, "head_dim": 16, "position_embedding_size": 72})
    with pytest.raises(ValueError, match="position"):
        check_student_covers(small, InputDomain())
    check_student_covers(small, InputDomain(budgets=(70, 140)))  # 69 patches per axis suffice there
    pv = torch.zeros(1, 9, 768)
    pos = torch.tensor([[[i, 0] for i in range(70, 79)]])
    with pytest.raises(ValueError, match="outside its declared input domain"):
        small(pixel_values=pv, pixel_position_ids=pos)


# ------------------------------------------------------------------------------------------- batching
def test_bucketed_sampler_exact_budget_shares_shapes_and_ranks():
    kw = dict(lengths=[50, 500, 7], weights=[0.5, 0.3, 0.2], budgets=[70, 140, 280, 560, 1120], tokens_per_batch=2240,
              samples_per_epoch=3000, budget_weights=[1, 1, 2, 0, 1], seed=3)
    s0, s1 = BucketedBatchSampler(**kw, num_replicas=2, rank=0), BucketedBatchSampler(**kw, num_replicas=2, rank=1)
    plan = s0.plan()
    counts = collections.Counter(b for b, *_ in plan)
    n = len(plan)
    assert counts[560] == 0 and abs(counts[280] - 2 * n / 5) <= 1 and abs(counts[70] - n / 5) <= 1  # exact shares
    b0, b1 = list(s0), list(s1)
    assert len(b0) == len(b1) == len(s0)  # equal number of batches on every rank (no DDP hang)
    d = InputDomain()
    for batch in b0 + b1:
        shapes = {(x[2], x[3], x[4]) for x in batch}
        assert len(shapes) == 1  # one budget and one shape per batch
        b, h, w = shapes.pop()
        assert d.contains(h, w) and len(batch) == batch_size_for((h, w), 2240)
    assert list(BucketedBatchSampler(**kw, num_replicas=2, rank=0)) == b0  # deterministic
    s0.set_epoch(1)
    assert list(s0) != b0


def test_exact_shape_sampler_groups_and_excludes():
    sizes = [(480, 640), (481, 641), (300, 1200), (2000, 100), (97, 131)]
    s = ExactShapeBatchSampler(sizes, [280], 10_000, domain=InputDomain())
    batches = list(s)
    assert s.excluded == {280: 1}  # the 20:1 strip is outside the domain
    for batch in batches:
        assert len({(x[3], x[4]) for x in batch}) == 1
        assert all(target_size(*sizes[x[1]], 280) == (x[3], x[4]) for x in batch)
    r = [list(ExactShapeBatchSampler(sizes, [280], 300, num_replicas=2, rank=k)) for k in (0, 1)]
    ids = [x[1] for rank in r for b in rank for x in b]
    assert sorted(ids) == [0, 1, 2, 3, 4]  # ranks partition the evaluation set


def test_collator_produces_padding_free_batches():
    ds = SyntheticVQADataset(6, max_side=500)
    coll = DualViewCollator(train=True, crop_scale=(0.35, 1.0))
    shape = processor_shapes(140, 4.0)[3]
    batch = coll([{**ds[i], "batch_spec": ("crop", 140, *shape)} for i in range(4)])
    assert tuple(batch.student_images.shape[-2:]) == shape and bool(batch.valid_token_mask.all())
    assert all(t.target_hw == shape and t.token_budget == 140 for t in batch.transforms)
    full = DualViewCollator(train=False)
    s = ds[0]
    shp = sample_transform(tuple(s["image"].shape[-2:]), 70, train=False).target_hw
    b2 = full([{**s, "batch_spec": ("full", 70, *shp)}])
    assert b2.transforms[0] == sample_transform(tuple(s["image"].shape[-2:]), 70, train=False)


def test_data_module_trains_every_budget_and_evaluates_per_budget():
    from onboard_vlm.data.data_module import DistillDataModule

    dm = DistillDataModule(
        train=[{"name": "synthetic", "num_samples": 40, "group": "a", "weight": 1.0}],
        val=[{"name": "synthetic", "num_samples": 10, "seed": 1, "group": "a"}],
        sampling={"mode": "crop", "budgets": [70, 140], "tokens_per_batch": 280},
        evaluation={"val_budgets": [70, 140], "tokens_per_batch": 280},
        input_domain={"budgets": [70, 140], "max_aspect": 4.0}, samples_per_epoch=40, num_workers=0, pin_memory=False,
    )
    dm.setup()
    seen = collections.Counter()
    for batch in dm.train_dataloader():
        assert bool(batch.valid_token_mask.all())
        seen[batch.transforms[0].token_budget] += 1
    assert set(seen) == {70, 140}
    assert dm.val_groups == [("a", 70), ("a", 140)] and len(dm.val_dataloader()) == 2
    with pytest.raises(ValueError, match="outside the input domain"):
        DistillDataModule(train=[{"name": "synthetic"}], sampling={"budgets": [1120]}, input_domain={"budgets": [70]})


def test_preliminary_runs_leave_out_sources_not_built(tmp_path):
    """allow_missing_sources: entries whose data is not on disk are left out (logged), the mixture
    renormalises over the rest; without it a missing source is an error."""
    from PIL import Image

    from onboard_vlm.data.data_module import DistillDataModule

    src = tmp_path / "src"
    src.mkdir()
    for i in range(16):
        Image.effect_mandelbrot((200 + 8 * i, 150), (-2, -1.5 + 0.05 * i, 1, 1.5), 30 + i).convert("RGB").save(src / f"{i}.png")
    built = tmp_path / "corpus" / "built"
    _run("make_datasets/corpus/make_image_corpus.py", "--source", "folder", "--input_path", str(src), "--name", "b",
         "--output_path", str(built), "--min_side", "1", "--min_std", "0")
    absent = str(tmp_path / "corpus" / "not_built")
    kw = dict(
        train=[{"name": "image_corpus", "root": str(built), "group": "web", "weight": 0.3},
               {"name": "image_corpus", "root": absent, "group": "scenes", "weight": 0.5},
               {"name": "image_corpus", "root": [str(built), absent], "group": "mixed", "weight": 0.2},
               {"name": "target_train", "data_root": str(tmp_path / "lmdb"), "group": "target", "weight": 0.1}],
        val=[{"name": "image_corpus", "root": absent, "split": "val", "group": "scenes"},
             {"name": "image_corpus", "root": str(built), "split": "val", "group": "web"}],
        dataset_defaults={"image_corpus": {"val_fraction": 0.3, "test_fraction": 0.0}},
        sampling={"mode": "crop", "budgets": [70], "tokens_per_batch": 280}, evaluation={"val_budgets": [70], "tokens_per_batch": 280},
        input_domain={"budgets": [70], "max_aspect": 4.0}, samples_per_epoch=8, num_workers=0, pin_memory=False,
    )
    with pytest.raises(FileNotFoundError):
        DistillDataModule(**kw).setup()
    dm = DistillDataModule(**kw, allow_missing_sources=True)
    dm.setup()
    from onboard_vlm.data.corpus import ImageCorpusDataset

    n_train = len(ImageCorpusDataset(str(built), split="train", val_fraction=0.3, test_fraction=0.0))
    assert dm.groups == ["web", "mixed"] and dm.mix_weights == [0.3, 0.2] and len(dm.train_set) == 2 * n_train > 0
    assert list(dm.val_sets) == ["web"] and dm.val_groups == [("web", 70)] and 0 < len(dm.val_sets["web"]) < 16
    left_out = {(m["split"], m["group"]) for m in dm.missing_sources if m["left_out"]}
    assert left_out == {("train", "scenes"), ("train", "target"), ("val", "scenes")}
    assert next(iter(dm.train_dataloader())).valid_token_mask.all()


def test_shrink_run_records_learning_curves(tmp_path):
    """A shrink-study run: validation at step 0, MSE metrics and a compute axis in metrics.csv, and student
    exports at fixed fractions of training (with layer_map auto and the student's own init seed)."""
    import csv as _csv

    traj = tmp_path / "traj"
    _run("run_distill.py", "+experiment=debug/dryrun_embedding", "student=gemma4_shrink", "student.num_hidden_layers=2",
         "student.hidden_size=24", "layer_map=auto", "validate_at_start=true", "trainer.max_steps=4", f"output_root={tmp_path}",
         "callbacks.diagnostics.every_n_steps=2", "+trainer.gradient_clip_val=1.0",
         f"+callbacks.trajectory={{_target_: onboard_vlm.utils.callbacks.TrajectoryCallback, dirpath: {traj}, fractions: [0, 0.5, 1]}}")
    steps = sorted(p.name for p in traj.iterdir())
    assert steps == ["step_00000000", "step_00000002", "step_00000004"]
    meta = json.load(open(traj / "step_00000004" / "trajectory.json"))
    assert meta["step"] == 4 and meta["images_seen"] > 0 and (traj / "step_00000000" / "student.safetensors").exists()
    rows = list(_csv.DictReader(open(next(tmp_path.glob("logs/*/*/metrics.csv")))))
    val0 = [r for r in rows if r["step"] == "0" and r.get("val/fid_lm_mse")]
    assert val0 and float(val0[0]["val/fid_lm_cos"]) < 1.0  # step 0: before any update
    assert any(r.get("progress/images_seen") for r in rows)
    diag = [r for r in rows if r.get("diag/grad_norm")]
    assert diag and all(r.get("diag/grad_clipped") in ("0.0", "1.0") for r in diag) and any(r.get("diag/update_ratio") for r in diag)
    run_dir = next(tmp_path.glob("logs/*/*/metrics.csv")).parent
    prov = json.load(open(run_dir / "provenance.json"))
    assert prov["seeds"] == {"seed": 0, "init_seed": 0, "data_seed": 0} and prov["versions"]["torch"] and (run_dir / "requirements.freeze.txt").exists()
    # every export evaluated (E1 here), skipping finished ones on a re-run
    ev = ["scripts/trajectory_eval.py", "--run_dir", str(traj), "--evals", "e1", "--teacher_config", "gemma4_tiny", "--budgets", "70",
          "--extra", "device=cpu", "dtype=float32", "num_workers=0", "input_domain={budgets: [70, 140], max_aspect: 4.0}", "tokens_per_batch=280",
          "sets=[{name: syn, group: g, dataset: {name: synthetic, num_samples: 4, max_side: 300}}]"]
    _run(*ev)
    assert all((traj / st / "eval" / "e1" / "summary.json").exists() for st in steps)
    assert "0 evaluations to run" in _run(*ev)


def test_literature_objective_runs_end_to_end(tmp_path):
    """Manifold distillation on intermediate layers (no projection heads) + weight multiplexing + consistent mixup:
    one tiny CPU run, with each loss term logged on its own."""
    import csv as _csv

    _run("run_distill.py", "+experiment=debug/dryrun_embedding", "student=gemma4_shrink", "student.num_hidden_layers=3",
         "student.hidden_size=24", "student.share_every=2", "student.share_transform=dwconv", "+objective=fm_mdL",
         "+data_module.collator.mixup_alpha=1.0", "trainer.max_steps=2", f"output_root={tmp_path}")
    rows = list(_csv.DictReader(open(next(tmp_path.glob("logs/*/*/metrics.csv")))))
    train = [r for r in rows if r.get("train/vis_mdL_intra")]
    assert train and all(r.get("train/lm_fm_cos") and r.get("train/vis_mdL_random") for r in train)
    stats = json.load(open(next(tmp_path.glob("students/*/student_stats.json"))))
    assert stats["parameters"] < stats["parameters_with_repeats"]  # shared weights counted once


def test_vision_space_objective_trains_on_vision_tokens_and_monitors_lm_space(tmp_path):
    """fmV: the loss is on the 768-d vision tokens only; the LM-space terms are logged with weight 0."""
    import csv as _csv

    _run("run_distill.py", "+experiment=debug/dryrun_embedding", "student=gemma4_shrink", "student.num_hidden_layers=2",
         "student.hidden_size=24", "+objective=fmV", "~layer_map", "+layer_map={}", "trainer.max_steps=2", f"output_root={tmp_path}")
    rows = [r for r in _csv.DictReader(open(next(tmp_path.glob("logs/*/*/metrics.csv")))) if r.get("train/total")]
    assert rows and all(r.get("train/lm_fm_cos") for r in rows)
    for r in rows:
        assert abs(float(r["train/total"]) - float(r["train/vis_fmV_mse"]) - float(r["train/vis_fmV_cos"])) < 1e-4


# ------------------------------------------------------------------------------------------- fidelity
def test_fidelity_metrics_are_per_token_scale_invariant():
    torch.manual_seed(0)
    t = torch.randn(30, 16)
    s = t + 0.3 * torch.randn(30, 16)
    proj = torch.nn.Sequential()  # stand-in for embed_vision: scale-free RMSNorm + linear
    w = torch.randn(16, 24)

    def project(z):
        return (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6)) @ w

    m1 = fidelity_metrics(s, t, [10, 20], project)
    m2 = fidelity_metrics(s * torch.rand(30, 1) * 10 + 0.0, t, [10, 20], project)  # rescale every student token
    for k in ("fid_lm_cos", "fid_lm_rel_err", "fid_vis_cos", "fid_cka"):
        assert torch.allclose(m1[k], m2[k], atol=1e-4), k
    same = fidelity_metrics(t, t, [10, 20], project)
    assert same["fid_lm_cos"] > 0.9999 and same["fid_lm_rel_err"] < 1e-5 and same["fid_cka"] > 0.9999
    assert proj is not None and abs(float(linear_cka(t, t)) - 1) < 1e-6
    recs = per_image_fidelity(s, t, [10, 20], project)
    assert [r["num_tokens"] for r in recs] == [10, 20] and all("lm_cos" in r and "cka" in r for r in recs)


# ------------------------------------------------------------------------------------------- baselines
def test_truncated_teacher_with_all_layers_is_the_teacher(tiny_teacher):
    full = build_student({"name": "gemma4_truncated", "num_layers": 3, "out_dim": 48, "tiny": TINY})
    full.encoder.load_state_dict(tiny_teacher.vision_tower.state_dict())  # same random teacher as the fixture
    ds = SyntheticVQADataset(3, max_side=300)
    shape = target_size(*ds[0]["image"].shape[-2:], 70)
    batch = DualViewCollator(train=False)([{**ds[0], "batch_spec": ("full", 70, *shape)}])
    with torch.no_grad():
        s = full(pixel_values=batch.teacher_inputs.pixel_values, pixel_position_ids=batch.teacher_inputs.pixel_position_ids).tokens
        t = tiny_teacher.encode_vision(batch.teacher_inputs).tokens
    torch.testing.assert_close(s, t, rtol=1e-5, atol=1e-5)
    half = build_student({"name": "gemma4_truncated", "num_layers": 1, "out_dim": 48, "tiny": TINY})
    assert half.config.num_hidden_layers == 1 and not any(p.requires_grad for p in half.parameters())


def test_frozen_backbone_projector_baseline():
    s = build_student({"name": "timm_spatial", "model_name": "mobilenetv4_conv_small", "out_dim": 16, "pretrained": False,
                       "freeze_backbone": True, "head": "mlp"})
    s.train()
    assert not s.backbone.training and all(not p.requires_grad for p in s.backbone.parameters())
    assert any(p.requires_grad for p in s.head.parameters())


# ------------------------------------------------------------------------------------------- teacher targets
def test_generate_teacher_responses_and_train_on_them(tmp_path):
    _run("run_generate_targets.py", "teacher=gemma4_tiny", "device=cpu", "dtype=float32", "max_new_tokens=4", "batch_size=2",
         "num_workers=0", "budget=70", "skip_empty=false", f"output_root={tmp_path}", f"hydra.run.dir={tmp_path}/h",
         "sources=[{dataset: {name: synthetic, num_samples: 3, max_side: 300}}]")
    from onboard_vlm.registry import build_dataset

    d = build_dataset({"name": "teacher_responses", "path": str(tmp_path / "teacher_responses" / "tiny_train")})
    assert len(d) == 3 and d[0]["meta"]["budget"] == 70 and d[0]["question"] in __import__("onboard_vlm.data.teacher_targets", fromlist=["x"]).DEFAULT_TEMPLATES.values()
    prov = json.load(open(tmp_path / "teacher_responses" / "tiny_train" / "part_000.provenance.json"))
    assert prov["num_records"] == 3 and prov["generation"]["do_sample"] is False


# ------------------------------------------------------------------------------------------- E1 / stats
def test_run_fidelity_end_to_end(tmp_path):
    _run("run_fidelity.py", "teacher=gemma4_tiny", "device=cpu", "dtype=float32", "num_workers=0", "run_name=t", f"output_root={tmp_path}",
         "student={name: gemma4_truncated, num_layers: 3, out_dim: 48}", "input_domain={budgets: [70, 140], max_aspect: 4.0}",
         "tokens_per_batch=280", "bootstrap.n=200", "allow_missing_sources=true",
         f"sets=[{{name: syn, group: g, dataset: {{name: synthetic, num_samples: 6, max_side: 400}}}}, "
         f"{{name: not_built, group: g, dataset: {{name: lmdb_vqa, path: {tmp_path}/none, split: val}}}}]")
    assert [s["set"] for s in json.load(open(tmp_path / "fidelity" / "t" / "skipped_sets.json"))] == ["not_built"]
    summ = json.load(open(tmp_path / "fidelity" / "t" / "summary.json"))
    rows = [json.loads(line) for line in open(tmp_path / "fidelity" / "t" / "per_image.jsonl")]
    assert {e["budget"] for e in summ["sets"]} == {70, 140} and rows
    # gemma4_truncated with every layer, built from the same tiny config, is a *different random* tower here,
    # so only the structure is checked; exact equality is covered by test_truncated_teacher_with_all_layers_is_the_teacher.
    assert all(set(e["lm_cos"]) == {"mean", "lo", "hi", "n"} for e in summ["sets"])


def test_statistics():
    rng = np.random.default_rng(0)
    x = rng.normal(0.6, 0.1, 400)
    ci = stats.bootstrap_ci(x, n_boot=2000)
    assert ci["lo"] < x.mean() < ci["hi"] and ci["n"] == 400
    pb = stats.paired_bootstrap(x + 0.02, x, n_boot=2000)
    assert abs(pb["diff"] - 0.02) < 1e-9 and pb["p"] < 0.01
    assert stats.mcnemar([1, 1, 1, 0] * 10, [0, 0, 0, 0] * 10)["p"] < 1e-6
    assert stats.mcnemar([1, 0] * 10, [0, 1] * 10)["p"] == 1.0
    assert stats.holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_seed_and_item_bootstrap():
    """One seed each: identical to the paired bootstrap. Several seeds with a seed effect: the CI widens."""
    rng = np.random.default_rng(0)
    item = rng.normal(0, 1, 1500)
    a, b = item + 0.05 + rng.normal(0, 0.2, 1500), item + rng.normal(0, 0.2, 1500)
    h, p = stats.hierarchical_paired_bootstrap([a], [b], n_boot=2000), stats.paired_bootstrap(a, b, n_boot=2000)
    assert h["diff"] == pytest.approx(p["diff"]) and abs((h["hi"] - h["lo"]) - (p["hi"] - p["lo"])) < 0.005
    seeds = [item + 0.05 + off + rng.normal(0, 0.2, 1500) for off in (-0.05, 0.0, 0.05)]
    h3 = stats.hierarchical_paired_bootstrap(seeds, [b], n_boot=2000)
    first = stats.paired_bootstrap(seeds[0], b, n_boot=2000)
    assert h3["seeds"] == [3, 1] and (h3["hi"] - h3["lo"]) > 2 * (first["hi"] - first["lo"])
    assert h3["lo"] < 0.05 < h3["hi"]


def test_aggregate_results(tmp_path):
    random.seed(0)
    for run, p_correct, seeds in (("ref", 0.7, 1), ("stu", 0.6, 2)):
        for s in range(seeds):
            d = tmp_path / f"{run}{s}" / "vizwiz_vqa"
            d.mkdir(parents=True)
            with open(d / "scored.jsonl", "w") as f:
                for i in range(200):
                    f.write(json.dumps({"sample_id": f"q{i}", "item_score": float(random.random() < p_correct)}) + "\n")
    _run("scripts/aggregate_results.py", "--out", str(tmp_path / "t"), "--reference", "ref", "--eval", f"ref={tmp_path}/ref0",
         "--eval", f"stu={tmp_path}/stu0,{tmp_path}/stu1")
    t = json.load(open(tmp_path / "t" / "tables.json"))["benchmarks"]["vizwiz_vqa"]
    assert t["stu"]["seeds"]["n_seeds"] == 2 and "vs_reference" in t["stu"] and "p_holm" in t["stu"]["vs_reference"]
    tex = (tmp_path / "t" / "tables.tex").read_text()
    assert "\\toprule" in tex and "vizwiz\\_vqa" in tex and "$\\pm$" in tex and "\\textbf{" in tex
    rows = [ln for ln in tex.splitlines() if ln.endswith("\\\\") and not ln.startswith("model")]
    assert rows[0].startswith("ref &")  # the reference first


def test_blocklist_reads_eval_set_fingerprints(tmp_path):
    from PIL import Image

    src = tmp_path / "src"
    src.mkdir()
    img = Image.effect_mandelbrot((200, 150), (-2, -1.5, 1, 1.5), 40).convert("RGB")
    img.save(src / "a.png")
    _run("make_datasets/corpus/make_image_corpus.py", "--source", "folder", "--input_path", str(src), "--name", "ev",
         "--output_path", str(tmp_path / "ev"), "--min_side", "1", "--min_std", "0", "--max_side", "0")
    _run("make_datasets/corpus/make_eval_blocklist.py", "--corpus_dirs", str(tmp_path / "ev"), "--output", str(tmp_path / "bl.json"))
    from onboard_vlm.data.hashing import sha1_bytes

    bl = json.load(open(tmp_path / "bl.json"))
    assert bl["sha1"] == [sha1_bytes((src / "a.png").read_bytes())] and bl["counts"] == {"ev/all": 1}


def test_plot_study_on_synthetic_runs(tmp_path):
    """Curves over images seen per series (seeds averaged), trajectory panels with the teacher's reference
    lines, final value vs parameters, and the fidelity -> downstream scatter with a run-level bootstrap."""
    pytest.importorskip("matplotlib")
    import csv as _csv

    from safetensors.torch import save_file

    for L in (4, 8):
        for s in (0, 1):
            rid = f"shrink_L{L}_D768_teacher_uniform-uniform_s{s}"
            log_dir = tmp_path / "logs" / "onboard_vlm" / rid
            log_dir.mkdir(parents=True)
            with open(log_dir / "metrics.csv", "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=["step", "progress/images_seen", "val/fid_lm_cos", "val/b70/fid_lm_cos", "train/total",
                                                   "diag/grad_norm", "lr-AdamW/pg1"])
                w.writeheader()
                for step in range(0, 500, 100):
                    v = 0.9 + L / 100 - 0.3 * np.exp(-step / 100) + 0.001 * s
                    w.writerow({"step": step, "progress/images_seen": step * 8, "val/fid_lm_cos": v, "val/b70/fid_lm_cos": v,
                                "train/total": 1 - v, "diag/grad_norm": 2 - v, "lr-AdamW/pg1": 1e-3})
            for step in (0, 200, 400):
                d = tmp_path / "trajectories" / rid / f"step_{step:08d}"
                (d / "eval" / "e3" / "cvbench").mkdir(parents=True)
                (d / "eval" / "e1").mkdir()
                json.dump({"step": step, "images_seen": step * 8}, open(d / "trajectory.json", "w"))
                save_file({"w": torch.zeros(L, 4)}, str(d / "student.safetensors"))
                v = 0.6 + L / 100 + step / 2000 + 0.002 * s
                json.dump({"sets": [{"set": "coco", "group": "ood", "budget": 280, "lm_cos": {"mean": v}}]}, open(d / "eval" / "e1" / "summary.json", "w"))
                json.dump({"cv_bench_overall": 0.3 + v / 2}, open(d / "eval" / "e3" / "cvbench" / "metrics.json", "w"))
    teacher = tmp_path / "teacher" / "eval" / "e3" / "cvbench"
    teacher.mkdir(parents=True)
    json.dump({"cv_bench_overall": 0.8}, open(teacher / "metrics.json", "w"))
    out = tmp_path / "plots"
    log = _run("scripts/plot_study.py", "--root", str(tmp_path), "--out", str(out), "--runs", "shrink_L*_D768_*",
               "--label", "{L} layers", "--order", "L", "--teacher_dir", str(tmp_path / "teacher"))
    assert "2 series" in log
    for name in ("curves_fidelity", "curves_budgets", "trajectory_e1", "trajectory_e3", "final_vs_params", "fidelity_vs_downstream", "curves_training"):
        assert (out / f"{name}.png").exists() and (out / f"{name}.csv").exists(), name
    rows = list(_csv.DictReader(open(out / "curves_fidelity.csv")))
    assert {r["series"] for r in rows} == {"4 layers", "8 layers"} and all(r["n_seeds"] == "2" for r in rows)


def test_consistent_mixup_gives_teacher_and_student_the_same_pixels():
    """Function matching (Beyer et al. 2022): mixed images, and the teacher's patches are built from the very
    pixels the student sees."""
    from onboard_vlm.geometry import build_teacher_patches

    ds = SyntheticVQADataset(4, max_side=300)
    samples = [{**ds[i], "batch_spec": ("crop", 70, 288, 384)} for i in range(4)]
    plain = DualViewCollator(train=True, crop_scale=(1.0, 1.0), seed=3)(samples)
    mixed = DualViewCollator(train=True, crop_scale=(1.0, 1.0), seed=3, mixup_alpha=1.0, mixup_prob=1.0)(samples)
    assert not torch.allclose(plain.student_images, mixed.student_images)
    imgs = [mixed.student_images[i, :, :288, :384] for i in range(4)]
    pv, _ = build_teacher_patches(imgs, max_patches=70 * 9)
    assert torch.equal(pv, mixed.teacher_inputs.pixel_values)  # identical pixels for both networks
    assert torch.allclose(mixed.student_images * 255, (mixed.student_images * 255).round(), atol=1e-3)  # 8-bit pixels


def test_general_suite_tables_retention_and_visual_retention(tmp_path):
    """Cambrian-1 categories from lmms-eval results json (0-100), seed mean, retention vs the teacher, and the share
    of the teacher's advantage over the blind run (grey images) a student keeps; per-question ANLS for DocVQA."""
    from onboard_vlm.eval import suites

    assert "mmbench_en_dev" not in suites.tasks() and "mmbench_en_dev" in suites.tasks(judged=True)
    assert suites.to_100("mme", 1500) == 75.0 and suites.to_100("gqa", 0.6) == 60.0
    assert suites.anls("the cat", ["The cat"]) == 1.0 and suites.anls("dog", ["cat"]) == 0.0
    assert suites.item_score("pope", {"pope_accuracy": {"score": 1.0}}) == 1.0
    assert suites.item_score("cv_bench_2d", {"cv_bench_acc": {"is_correct": False}}) == 0.0

    def write(d, gqa, chartqa):
        d.mkdir(parents=True)
        json.dump({"results": {"gqa": {"exact_match,none": gqa}, "chartqa": {"relaxed_overall,none": chartqa},
                               "mme": {"mme_perception_score,none": 1000.0 + 1000 * gqa}}}, open(d / "x_results.json", "w"))

    write(tmp_path / "t", 0.60, 0.80)
    write(tmp_path / "s0", 0.50, 0.60)
    write(tmp_path / "s1", 0.54, 0.64)
    write(tmp_path / "b", 0.40, 0.10)
    _run("scripts/aggregate_results.py", "--out", str(tmp_path / "o"), "--reference", "teacher", "--blind", "blind",
         "--lmms", f"teacher={tmp_path}/t", "--lmms", f"student={tmp_path}/s0,{tmp_path}/s1", "--lmms", f"blind={tmp_path}/b")
    st = json.load(open(tmp_path / "o" / "tables.json"))["suite"]["models"]
    gen = st["student"]["categories"]["General"]  # mean of MME (/20) and GQA, per seed, then over seeds
    assert gen["benchmarks"] == ["mme", "gqa"] and gen["n_seeds"] == 2
    assert abs(gen["mean"] - np.mean([(75 + 50) / 2, (77 + 54) / 2])) < 1e-9
    t, s, b = (st[m]["categories"]["OCR & Chart"]["mean"] for m in ("teacher", "student", "blind"))
    assert abs(st["student"]["categories"]["OCR & Chart"]["visual_retention"] - (s - b) / (t - b)) < 1e-9
    assert "General" in (tmp_path / "o" / "tables.tex").read_text()
