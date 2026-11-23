"""Teacher-feature cache round trip and one-step Stage-1 / Stage-2 training."""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from onboard_vlm.data.collator import DualViewCollator
from onboard_vlm.data.datasets import CachedTeacherDataset, SyntheticVQADataset
from onboard_vlm.data.prompts import Gemma4ChatInputs
from onboard_vlm.distill.tasks import Stage1FeatureDistillTask, Stage2GraftedKDTask
from onboard_vlm.teachers.feature_cache import TeacherFeatureCacheWriter

TINY = {"vision_hidden": 48, "vision_layers": 3}
STUDENT = {"name": "compact_vit", "out_dim": 48, "embed_dim": 32, "depth": 4, "num_heads": 2}


def _trainer(steps=2):
    return pl.Trainer(max_steps=steps, accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)


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


def test_stage1_task_trains_and_exports(tmp_path):
    task = Stage1FeatureDistillTask(
        student=STUDENT, teacher={"name": "gemma4", "tiny": TINY, "dtype": "float32"},
        criterion=[{"name": "token_feature"}, {"name": "intermediate_layer", "mode": "relational"}],
        layer_map={1: 0, 3: 2}, optimizer={"lr": 1e-3},
    )
    dl = DataLoader(SyntheticVQADataset(8), batch_size=4, collate_fn=DualViewCollator([70, 140], train=True))
    before = [p.clone() for p in task.student.parameters()]
    _trainer().fit(task, dl)
    assert any(not torch.equal(a, b) for a, b in zip(before, task.student.parameters()))
    task.export_student(str(tmp_path))
    assert (tmp_path / "student.safetensors").exists() and (tmp_path / "student_config.json").exists()
    assert not any(k.startswith("adapter") for k in __import__("safetensors.torch", fromlist=["load_file"]).load_file(str(tmp_path / "student.safetensors")))


def _stage2_loader(tiny_processor):
    return DataLoader(SyntheticVQADataset(8), batch_size=4, collate_fn=DualViewCollator(70, build_student_view=False, text_builder=Gemma4ChatInputs(tiny_processor, "auto", with_answer=True)))


def test_stage2_default_trains_only_the_student_encoder(tiny_processor):
    """Default Stage 2: the whole VLM (embed_vision + LM) stays byte-identical; only the student learns."""
    task = Stage2GraftedKDTask(vlm_id=None, student_dir=None, tiny_config=TINY, student=STUDENT, dtype="float32", gradient_checkpointing=False)
    trainable = {n for n, p in task.named_parameters() if p.requires_grad}
    assert trainable and all(".vision_tower.student." in n for n in trainable)
    vlm_before = {n: p.clone() for n, p in task.grafted.vlm.named_parameters() if ".vision_tower.student." not in n}
    stu_before = {n: p.clone() for n, p in task.grafted.student_tower.named_parameters()}
    _trainer().fit(task, _stage2_loader(tiny_processor))
    assert all(torch.equal(p, vlm_before[n]) for n, p in task.grafted.vlm.named_parameters() if n in vlm_before)
    assert any(not torch.equal(p, stu_before[n]) for n, p in task.grafted.student_tower.named_parameters())


def test_stage2_embed_vision_ablation(tiny_processor):
    task = Stage2GraftedKDTask(vlm_id=None, student_dir=None, tiny_config=TINY, student=STUDENT, dtype="float32",
                               train_embed_vision="full", train_student=False, gradient_checkpointing=False)
    trainable = {n for n, p in task.named_parameters() if p.requires_grad}
    assert trainable and all(".embed_vision." in n for n in trainable)
    frozen_before = {n: p.clone() for n, p in task.named_parameters() if not p.requires_grad and "language_model" in n}
    _trainer().fit(task, _stage2_loader(tiny_processor))
    assert all(torch.equal(p, frozen_before[n]) for n, p in task.named_parameters() if n in frozen_before)


def test_stage1_gemma4_student_with_lm_space_loss(tiny_teacher):
    task = Stage1FeatureDistillTask(
        student={"name": "gemma4_vit", "out_dim": 48, "hidden_size": 32, "num_hidden_layers": 2, "head_dim": 24, "position_embedding_size": 64},
        teacher={"name": "gemma4", "tiny": TINY, "dtype": "float32"},
        criterion=[{"name": "token_feature"}, {"name": "token_feature", "metric": "cosine", "space": "lm", "label": "lm_cos"},
                   {"name": "intermediate_layer", "mode": "feature"}],
        layer_map={1: 2}, optimizer={"lr": 1e-3},
    )
    dl = DataLoader(SyntheticVQADataset(8), batch_size=4, collate_fn=DualViewCollator([70, 140], train=True, degradations={"motion_blur": 0.5, "low_light": 0.5}))
    before = [p.clone() for p in task.student.parameters()]
    tr = _trainer()
    tr.fit(task, dl)
    assert "train/lm_cos" in tr.callback_metrics
    assert any(not torch.equal(a, b) for a, b in zip(before, task.student.parameters()))
