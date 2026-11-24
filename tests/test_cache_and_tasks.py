"""Teacher-feature cache round trip and one-step training of the three distillation tasks."""

import pytorch_lightning as pl
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from onboard_vlm.data.collator import DualViewCollator
from onboard_vlm.data.datasets import CachedTeacherDataset, SyntheticVQADataset
from onboard_vlm.data.prompts import Gemma4ChatInputs
from onboard_vlm.tasks import EmbeddingDistillationTask, FeatureDistillationTask, LogitDistillationTask
from onboard_vlm.teachers.feature_cache import TeacherFeatureCacheWriter

TINY = {"vision_hidden": 48, "vision_layers": 3}
TEACHER = {"name": "gemma4", "tiny": TINY, "dtype": "float32"}
STUDENT = {"name": "compact_vit", "out_dim": 48, "embed_dim": 32, "depth": 4, "num_heads": 2}


def hparams(**kw):
    """The slice of the Hydra run config that tasks read (TimeFM-style ``Task(hparams, **task_cfg)``)."""
    base = {"teacher": TEACHER, "student": STUDENT, "criterion": [{"name": "token_feature"}], "layer_map": {}, "cache_root": None}
    return {**base, **kw}


def _trainer(steps=2):
    return pl.Trainer(max_steps=steps, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, num_sanity_val_steps=0)


def test_cache_roundtrip_reproduces_student_view(tiny_teacher, tmp_path):
    ds = SyntheticVQADataset(6, max_side=400)
    collate = DualViewCollator([70, 140], train=True, hflip_prob=0.5, seed=3)
    writer = TeacherFeatureCacheWriter(str(tmp_path), {"hidden_size": 48, "layers": [0, 2]}, shard_size=4)
    ref = {}
    for i in range(0, 6, 3):
        b = collate([ds[j] for j in range(i, i + 3)])
        out = tiny_teacher.encode_vision(b.teacher_inputs, [0, 2])
        writer.add_batch(b.sample_ids, b.transforms, out)
        for k, t, tok in zip(b.sample_ids, b.transforms, out.split()):
            ref[k] = (t, tok)
    writer.close()
    cached = CachedTeacherDataset({"name": "synthetic", "num_samples": 6, "max_side": 400}, str(tmp_path), random_view=False)
    samples = [cached[i] for i in range(6)]
    batch = DualViewCollator(280, train=False, build_teacher_inputs=False)(samples)
    assert batch.teacher_inputs is None and batch.cached_teacher is not None
    for s, t in zip(samples, batch.transforms):
        assert t == ref[s["sample_id"]][0]  # exact transform re-applied
        assert torch.allclose(s["teacher_tokens"].float(), ref[s["sample_id"]][1].to(torch.bfloat16).float())
    assert batch.cached_teacher.image_token_lengths == batch.image_token_lengths
    assert set(batch.cached_teacher.intermediate_tokens) == {0, 2}


def test_feature_task_trains_and_exports(tmp_path):
    cfg = hparams(criterion=[{"name": "token_feature"}, {"name": "intermediate_layer", "mode": "relational"}], layer_map={1: 0, 3: 2})
    task = FeatureDistillationTask(cfg, optimizer={"lr": 1e-3})
    dl = DataLoader(SyntheticVQADataset(8), batch_size=4, collate_fn=DualViewCollator([70, 140], train=True))
    before = [p.clone() for p in task.student.parameters()]
    tr = _trainer()
    tr.fit(task, dl)
    assert "train/total" in tr.callback_metrics and "train/fid_vis_cos" in tr.callback_metrics
    assert any(not torch.equal(a, b) for a, b in zip(before, task.student.parameters()))
    task.export(str(tmp_path))
    assert (tmp_path / "student.safetensors").exists() and (tmp_path / "student_config.json").exists()
    assert not any(k.startswith("adapter") for k in load_file(str(tmp_path / "student.safetensors")))


def test_feature_task_with_gemma4_student(tiny_teacher):
    student = {"name": "gemma4_vit", "out_dim": 48, "hidden_size": 32, "num_hidden_layers": 2, "head_dim": 16, "position_embedding_size": 256}
    task = FeatureDistillationTask(hparams(student=student, criterion=[{"name": "token_feature"}, {"name": "intermediate_layer", "mode": "feature"}],
                                           layer_map={1: 2}), optimizer={"lr": 1e-3})
    dl = DataLoader(SyntheticVQADataset(8), batch_size=4, collate_fn=DualViewCollator([70, 140], train=True, degradations={"motion_blur": 0.5}))
    before = [p.clone() for p in task.student.parameters()]
    _trainer().fit(task, dl)
    assert any(not torch.equal(a, b) for a, b in zip(before, task.student.parameters()))


def test_embedding_task_uses_frozen_embed_vision():
    task = EmbeddingDistillationTask(hparams(criterion=[{"name": "token_feature", "metric": "mse", "normalize": "none"},
                                                        {"name": "token_feature", "metric": "cosine", "label": "cos"}]),
                                     vision_criterion=[{"name": "token_feature", "metric": "cosine", "label": "vcos"}], optimizer={"lr": 1e-3})
    dl = DataLoader(SyntheticVQADataset(8), batch_size=4, collate_fn=DualViewCollator([70], train=True))
    tr = _trainer()
    tr.fit(task, dl)
    ev_before = {n: p.clone() for n, p in task.teacher.embed_vision.named_parameters()}
    tr2 = _trainer(1)
    tr2.fit(task, dl)
    assert {"train/lm_cos", "train/vis_vcos"} <= set(tr.callback_metrics)
    assert all(torch.equal(p, ev_before[n]) for n, p in task.teacher.embed_vision.named_parameters())
    assert all(not p.requires_grad for p in task.teacher.parameters())


def test_embedding_task_rejects_intermediate_terms():
    import pytest

    with pytest.raises(ValueError, match="vision_criterion"):
        EmbeddingDistillationTask(hparams(criterion=[{"name": "intermediate_layer"}], layer_map={1: 0}))


def _logit_loader(tiny_processor):
    return DataLoader(SyntheticVQADataset(8), batch_size=4,
                      collate_fn=DualViewCollator(70, build_student_view=False, text_builder=Gemma4ChatInputs(tiny_processor, "auto", with_answer=True)))


def test_logit_task_trains_only_the_student_encoder(tiny_processor, tmp_path):
    """Default: the whole VLM (embed_vision + LM) stays byte-identical; only the student learns."""
    task = LogitDistillationTask(hparams(criterion={"name": "grafted_logit_kd"}), dtype="float32", gradient_checkpointing=False)
    trainable = {n for n, p in task.named_parameters() if p.requires_grad}
    assert trainable and all(".vision_tower.student." in n for n in trainable)
    vlm_before = {n: p.clone() for n, p in task.grafted.vlm.named_parameters() if ".vision_tower.student." not in n}
    stu_before = {n: p.clone() for n, p in task.grafted.student_tower.named_parameters()}
    _trainer().fit(task, _logit_loader(tiny_processor))
    assert all(torch.equal(p, vlm_before[n]) for n, p in task.grafted.vlm.named_parameters() if n in vlm_before)
    assert any(not torch.equal(p, stu_before[n]) for n, p in task.grafted.student_tower.named_parameters())
    task.export(str(tmp_path))
    assert (tmp_path / "student.safetensors").exists() and not (tmp_path / "embed_vision.safetensors").exists()


def test_logit_task_embed_vision_ablation(tiny_processor):
    task = LogitDistillationTask(hparams(criterion={"name": "grafted_logit_kd"}), dtype="float32", train_embed_vision="full",
                                 train_student=False, gradient_checkpointing=False)
    trainable = {n for n, p in task.named_parameters() if p.requires_grad}
    assert trainable and all(".embed_vision." in n for n in trainable)
    frozen_before = {n: p.clone() for n, p in task.named_parameters() if not p.requires_grad and "language_model" in n}
    _trainer().fit(task, _logit_loader(tiny_processor))
    assert all(torch.equal(p, frozen_before[n]) for n, p in task.named_parameters() if n in frozen_before)


def test_logit_task_starts_from_feature_export(tmp_path, tiny_processor):
    feat = FeatureDistillationTask(hparams())
    feat.export(str(tmp_path / "feat"))
    task = LogitDistillationTask(hparams(criterion={"name": "grafted_logit_kd"}, student={"name": "ignored"}),
                                 student_dir=str(tmp_path / "feat"), dtype="float32", gradient_checkpointing=False)
    sd_feat = load_file(str(tmp_path / "feat" / "student.safetensors"))
    sd_task = task.student.state_dict()
    assert all(torch.equal(sd_feat[k], sd_task[k]) for k in sd_feat)
    assert task.cfg["student"]["name"] == "compact_vit"


def test_per_group_validation_and_test_metrics():
    """One dataloader per val / test group: metrics per group plus the image-weighted total and val_loss."""
    from onboard_vlm.data.data_module import DistillDataModule

    dm = DistillDataModule(
        train=[{"name": "synthetic", "num_samples": 8}],
        val=[{"name": "synthetic", "num_samples": 8, "seed": 1, "group": "web"}, {"name": "synthetic", "num_samples": 4, "seed": 2, "group": "ego"}],
        test=[{"name": "synthetic", "num_samples": 4, "seed": 3, "group": "web"}],
        collator={"token_budgets": [70], "train": True}, val_collator={"token_budgets": [70], "train": False},
        batch_size=4, num_workers=0, pin_memory=False,
    )
    task = FeatureDistillationTask(hparams(), optimizer={"lr": 1e-3})
    tr = pl.Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False,
                    enable_model_summary=False, num_sanity_val_steps=0)
    tr.fit(task, dm)
    m = {k: float(v) for k, v in tr.callback_metrics.items()}
    assert {"val/web/total", "val/ego/total", "val/total", "val_loss", "val/web/fid_vis_cos", "val/web/fid_lm_cos", "val/web/fid_cka"} <= set(m)
    assert abs(m["val/total"] - (8 * m["val/web/total"] + 4 * m["val/ego/total"]) / 12) < 1e-4 and m["val_loss"] == m["val/total"]
    tr.test(task, dm)
    assert {"test/web/total", "test/total"} <= set(tr.callback_metrics)
