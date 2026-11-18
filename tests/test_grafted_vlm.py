"""(5) End-to-end forward / generate with GraftedVLM through the real processor + eval collator."""

import pytest
import torch
from transformers import Gemma4ForConditionalGeneration

from onboard_vlm.data.collator import DualViewCollator, VLMEvalCollator
from onboard_vlm.data.datasets import SyntheticVQADataset
from onboard_vlm.data.prompts import Gemma4ChatInputs
from onboard_vlm.eval.runner import continuation_logprob, run_benchmark
from onboard_vlm.registry import build_student
from onboard_vlm.students.io import load_student, save_student
from onboard_vlm.vlm.grafted import GraftedVLM, LoRALinear

D = 48
FAMILIES = {
    "compact_vit": {"name": "compact_vit", "out_dim": D, "embed_dim": 32, "depth": 2, "num_heads": 2},
    "coarse_patch_vit": {"name": "coarse_patch_vit", "out_dim": D, "embed_dim": 32, "depth": 2, "num_heads": 2},
    "multiplexed_vit": {"name": "multiplexed_vit", "out_dim": D, "embed_dim": 32, "num_heads": 2, "num_shared_blocks": 1, "steps_per_block": 2},
    "timm_spatial": {"name": "timm_spatial", "model_name": "mobilenetv4_conv_small", "out_dim": D, "pretrained": False},
}


@pytest.fixture()
def vlm(tiny_cfg):
    return Gemma4ForConditionalGeneration(tiny_cfg).eval()


def _eval_inputs(processor, n=3, frames=0):
    ds = SyntheticVQADataset(n, max_side=500, num_frames=frames)
    collate = VLMEvalCollator(Gemma4ChatInputs(processor, prompt="vizwiz_vqa"), image_token_budget=70, video_token_budget=70)
    return collate([ds[i] for i in range(n)])


@pytest.mark.parametrize("family", list(FAMILIES))
def test_generate_with_every_student_family(vlm, tiny_processor, family):
    inputs, extras = _eval_inputs(tiny_processor)
    student = build_student(FAMILIES[family]).eval()
    calls = []
    student.register_forward_hook(lambda m, i, o: calls.append(o.tokens.shape))
    g = GraftedVLM(vlm, student, keep_teacher_tower=True)
    out = g.generate(**inputs, max_new_tokens=4, do_sample=False)
    assert out.shape[0] == 3 and out.shape[1] == inputs["input_ids"].shape[1] + 4
    assert calls, "student was not invoked by HF generate"
    n_image_tokens = int((inputs["input_ids"] == vlm.config.image_token_id).sum())
    assert sum(c[0] for c in calls[:3 if g.student_tower.per_image else 1]) == n_image_tokens
    with torch.no_grad():
        s_logits = g(**inputs).logits
        with g.teacher_vision():
            t_logits = g(**inputs).logits
    assert s_logits.shape == t_logits.shape and not torch.allclose(s_logits, t_logits)
    g.ungraft()
    assert type(vlm.model.vision_tower).__name__ == "Gemma4VisionModel"


def test_video_generate(vlm, tiny_processor):
    inputs, _ = _eval_inputs(tiny_processor, n=2, frames=3)
    assert "pixel_values_videos" in inputs
    g = GraftedVLM(vlm, build_student(FAMILIES["compact_vit"]).eval())
    out = g.generate(**inputs, max_new_tokens=3, do_sample=False)
    assert out.shape[0] == 2


def test_stage2_freezing_lora_and_export(vlm, tmp_path):
    student = build_student(FAMILIES["compact_vit"])
    g = GraftedVLM(vlm, student)
    params = g.configure_stage2("lora", lora_rank=4)
    assert isinstance(vlm.model.embed_vision.embedding_projection, LoRALinear)
    assert sum(p.numel() for p in params) == 4 * (D + vlm.config.text_config.hidden_size)
    assert all(not p.requires_grad for p in g.student_tower.parameters())
    sd = g.embed_vision_state_dict()
    assert set(sd) == {"embedding_projection.weight"}
    save_student(str(tmp_path), student, FAMILIES["compact_vit"])
    loaded, cfg = load_student(str(tmp_path))
    assert cfg["name"] == "compact_vit"
    for a, b in zip(student.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)


def test_out_dim_mismatch_rejected(vlm):
    with pytest.raises(ValueError):
        GraftedVLM(vlm, build_student({**FAMILIES["compact_vit"], "out_dim": 32}))


def test_stage2_collator_answer_mask(tiny_processor):
    ds = SyntheticVQADataset(4, max_side=400)
    collate = DualViewCollator(70, build_student_view=False, text_builder=Gemma4ChatInputs(tiny_processor, prompt="auto", with_answer=True))
    b = collate([ds[i] for i in range(4)])
    ti = b.text_inputs
    k = int(ti["logits_to_keep"])
    assert ti["kd_mask"].shape == (4, k) and ti["kd_mask"].any(1).all()
    ids = ti["input_ids"]
    # every distilled target is the next input token of the answer span
    tgt = ti["kd_targets"][ti["kd_mask"]]
    assert set(tgt.tolist()) <= set(ids[:, -(k - 1):].flatten().tolist())
    assert b.student_images.numel() == 0 and b.teacher_inputs.pixel_values.shape[0] == 4


def test_runner_end_to_end(vlm, tiny_processor, tmp_path):
    g = GraftedVLM(vlm, build_student(FAMILIES["coarse_patch_vit"]).eval())
    m = run_benchmark(g, tiny_processor, "vizwiz_vqa", SyntheticVQADataset(4, max_side=300), str(tmp_path), batch_size=2, num_workers=0, image_token_budget=70)
    assert m["num_predictions"] == 4 and 0.0 <= m["vqa_accuracy"] <= 1.0 and "answerability_ap" in m
    assert (tmp_path / "predictions.jsonl").exists() and (tmp_path / "vizwiz_vqa_submission.json").exists()
    inputs, _ = _eval_inputs(tiny_processor, n=2)
    lp = continuation_logprob(g, inputs, torch.tensor([20, 21]))
    assert lp.shape == (2,) and (lp <= 0).all()
