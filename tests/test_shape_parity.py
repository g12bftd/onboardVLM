"""(1) Shape & unpadding parity between Gemma4Teacher and all 4 student families across aspect
ratios and token budgets, through the real dual-view collator."""

import pytest
import torch

from onboard_vlm.data.collator import DualViewCollator
from onboard_vlm.geometry import pad_dense
from onboard_vlm.registry import build_student
from onboard_vlm.teachers.gemma4 import EncoderFreeTeacherError, Gemma4Teacher, pool_patch_states
from onboard_vlm.teachers.hub import UnsupportedTeacherError

D = 48  # tiny teacher vision width
STUDENTS = {
    "compact_vit": {"name": "compact_vit", "out_dim": D, "embed_dim": 32, "depth": 3, "num_heads": 2},
    "coarse_patch_vit": {"name": "coarse_patch_vit", "out_dim": D, "embed_dim": 32, "depth": 3, "num_heads": 2, "stem": "conv"},
    "multiplexed_vit": {"name": "multiplexed_vit", "out_dim": D, "embed_dim": 32, "num_heads": 2, "num_shared_blocks": 2, "steps_per_block": 2},
    "gemma4_vit": {"name": "gemma4_vit", "out_dim": D, "hidden_size": 32, "num_hidden_layers": 2, "head_dim": 16, "position_embedding_size": 128},
    "gemma4_vit_p48": {"name": "gemma4_vit", "out_dim": D, "hidden_size": 32, "num_hidden_layers": 2, "head_dim": 16, "patch_size": 48},
    "timm_mobilenetv4": {"name": "timm_spatial", "model_name": "mobilenetv4_conv_small", "out_dim": D, "pretrained": False},
    "timm_fastvit": {"name": "timm_spatial", "model_name": "fastvit_t8", "out_dim": D, "pretrained": False, "alignment": "tap16_conv3"},
}
SOURCE_SIZES = [(480, 640), (1080, 1920), (300, 1200), (2000, 500), (97, 131), (333, 333)]


def _batch(budget, train=False):
    samples = [{"sample_id": str(i), "image": torch.randint(0, 256, (3, *hw), dtype=torch.uint8)} for i, hw in enumerate(SOURCE_SIZES)]
    return DualViewCollator(budget, train=train, hflip_prob=0.5)(samples)


@pytest.mark.parametrize("budget", [70, 140, 280])
@pytest.mark.parametrize("family", list(STUDENTS))
def test_student_teacher_token_parity(tiny_teacher, family, budget):
    batch = _batch(budget, train=budget == 140)
    t = tiny_teacher.encode_vision(batch.teacher_inputs, layers=[0, 2])
    student = build_student(STUDENTS[family]).eval()
    layers = sorted(student.layer_dims)[-2:]
    with torch.no_grad():
        s = student(batch.student_images, batch.valid_token_mask, return_layers=layers)
    n = sum(batch.image_token_lengths)
    assert t.tokens.shape == (n, D) and s.tokens.shape == (n, D)
    assert s.image_token_lengths == t.image_token_lengths == batch.image_token_lengths
    assert all(v.shape[0] == n for v in list(s.intermediate_tokens.values()) + list(t.intermediate_tokens.values()))
    assert batch.valid_token_mask.sum().item() == n
    for i, tr in enumerate(batch.transforms):
        assert batch.image_token_lengths[i] == tr.num_tokens <= tr.token_budget


@pytest.mark.parametrize("family", ["compact_vit", "coarse_patch_vit", "multiplexed_vit", "gemma4_vit", "gemma4_vit_p48"])
def test_vit_students_are_batch_invariant(family):
    torch.manual_seed(0)
    student = build_student(STUDENTS[family]).eval()
    for n, p in student.named_parameters():  # break zero-inits so every transform is active
        if "dwconv" in n or "head_mix" in n:
            p.data.add_(0.1 * torch.randn_like(p))
    imgs = [torch.rand(3, 96, 144), torch.rand(3, 240, 48), torch.rand(3, 48, 48)]
    x, m, _ = pad_dense(imgs)
    with torch.no_grad():
        batched = student(x, m).tokens
        single = torch.cat([student(*pad_dense([im])[:2]).tokens for im in imgs])
    assert torch.allclose(batched, single, rtol=1e-5, atol=1e-5 * single.abs().max().item())


def test_teacher_intermediate_pooling_matches_gemma_pooler(tiny_teacher):
    batch = _batch(70)
    pos = batch.teacher_inputs.pixel_position_ids
    h = torch.randn(pos.shape[0], pos.shape[1], D)
    pad = (pos == -1).all(-1)
    ref, ref_mask = tiny_teacher.vision_tower.pooler._avg_pool_by_positions(h.masked_fill(pad[..., None], 0), pos, 70)
    mine, lengths = pool_patch_states(h, pos)
    assert lengths == batch.image_token_lengths
    assert torch.allclose(ref[ref_mask], mine, atol=1e-6)
    # final tokens == pooled last layer * sqrt(D) (standardize=False): one consistent token order
    t = tiny_teacher.encode_vision(batch.teacher_inputs, layers=[2])
    assert torch.allclose(t.tokens, t.intermediate_tokens[2] * D**0.5, atol=1e-4)


@pytest.mark.parametrize("model_id", ["google/gemma-4-12b-it", "google/gemma-4-31B-it", "google/gemma-4-26B-A4B-it", "google/gemma-3n-E2B-it"])
def test_only_e2b_e4b_teachers_are_accepted(model_id):
    with pytest.raises(UnsupportedTeacherError):
        Gemma4Teacher(model_id)


def test_pinned_revision_enforced():
    with pytest.raises(UnsupportedTeacherError):
        Gemma4Teacher("google/gemma-4-E2B-it", revision="main")


def test_encoder_free_teacher_rejected(tiny_cfg):
    cfg = tiny_cfg.to_dict()
    cfg["vision_config"] = None
    with pytest.raises(EncoderFreeTeacherError):
        Gemma4Teacher(config=cfg)
