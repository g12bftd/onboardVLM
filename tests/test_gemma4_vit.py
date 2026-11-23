"""Gemma-architecture student: exact teacher inheritance, weight selection, 48-px variant."""

import pytest
import torch

from onboard_vlm.data.collator import DualViewCollator
from onboard_vlm.students.gemma4_vit import Gemma4ViTStudent, select_indices
from onboard_vlm.teachers.gemma4 import Gemma4Teacher

TINY = {"vision_hidden": 64, "vision_layers": 4}


@pytest.fixture(scope="module")
def teacher():
    torch.manual_seed(0)
    t = Gemma4Teacher(tiny=TINY, compute_dtype="float32")
    for n, b in t.vision_tower.named_buffers():  # non-trivial clamp ranges, like the real checkpoint
        if n.endswith("_min"):
            b.fill_(-3.0)
        elif n.endswith("_max"):
            b.fill_(3.0)
    return t


@pytest.fixture(scope="module")
def batch():
    imgs = [{"sample_id": str(i), "image": torch.randint(0, 256, (3, *hw), dtype=torch.uint8)} for i, hw in enumerate([(480, 640), (300, 1200), (97, 131)])]
    return DualViewCollator(70)(imgs)


def _sd(t):
    return {k: v.float() for k, v in t.vision_tower.state_dict().items()}


def test_full_size_inheritance_is_bitwise_identical_to_teacher(teacher, batch):
    vc = teacher.config.vision_config
    s = Gemma4ViTStudent(
        out_dim=64, hidden_size=64, num_hidden_layers=4, head_dim=vc.head_dim, intermediate_size=vc.intermediate_size,
        position_embedding_size=vc.position_embedding_size, use_clipped_linears=True, init="teacher", teacher_state_dict=_sd(teacher),
    ).eval()
    ref = teacher.encode_vision(batch.teacher_inputs, layers=[1, 3])
    with torch.no_grad():
        g = s.forward_gemma(batch.teacher_inputs.pixel_values, batch.teacher_inputs.pixel_position_ids, return_layers=[1, 3])
        d = s(batch.student_images, batch.valid_token_mask)
    assert torch.equal(g.tokens, ref.tokens)
    assert all(torch.equal(g.intermediate_tokens[i], ref.intermediate_tokens[i]) for i in (1, 3))
    assert torch.allclose(d.tokens, ref.tokens, rtol=1e-5, atol=1e-5 * ref.tokens.abs().max().item())  # dense path: float order only


@pytest.mark.parametrize("sel", ["uniform", "first", "magnitude"])
def test_weight_selection_init_shapes_and_projection(teacher, batch, sel):
    vc = teacher.config.vision_config
    s = Gemma4ViTStudent(
        out_dim=64, hidden_size=32, num_hidden_layers=2, head_dim=vc.head_dim, intermediate_size=48, position_embedding_size=64,
        init="teacher", teacher_state_dict=_sd(teacher), width_selection=sel if sel != "magnitude" else "uniform",
        head_selection=sel, mlp_selection=sel, layer_selection="uniform",
    ).eval()
    assert s.init_info["layers"] == [0, 3]
    with torch.no_grad():
        out = s(batch.student_images, batch.valid_token_mask, return_layers=[0, 1])
    assert out.tokens.shape == (sum(batch.image_token_lengths), 64) and out.intermediate_tokens[1].shape[1] == 32
    w = s.head.weight
    assert (w != 0).sum() == 32  # selected dims mapped back to their teacher positions


def test_select_indices():
    assert select_indices(16, 8, "uniform").tolist() == [0, 2, 4, 6, 9, 11, 13, 15]
    assert select_indices(16, 4, "last").tolist() == [12, 13, 14, 15]
    assert select_indices(5, 2, "magnitude", torch.tensor([0.1, 5, 0.2, 9, 0])).tolist() == [1, 3]
    assert select_indices(16, 3, [15, 0, 7]).tolist() == [0, 7, 15]
    with pytest.raises(ValueError):
        select_indices(4, 2, [1, 1])


def test_invalid_configs(teacher):
    with pytest.raises(ValueError):
        Gemma4ViTStudent(hidden_size=64, num_hidden_layers=2, head_dim=16, init="teacher", teacher_state_dict=_sd(teacher))
    with pytest.raises(ValueError):
        Gemma4ViTStudent(patch_size=48, init="teacher", teacher_state_dict=_sd(teacher))
    with pytest.raises(ValueError):
        Gemma4ViTStudent(patch_size=32)


def test_patch48_variant_and_bf16_cast(batch):
    s = Gemma4ViTStudent(out_dim=64, hidden_size=32, num_hidden_layers=2, head_dim=16, patch_size=48).eval()
    assert not s.accepts_teacher_patches
    with torch.no_grad():
        out = s(batch.student_images, batch.valid_token_mask, return_layers=[1])
    assert out.image_token_lengths == batch.image_token_lengths and out.intermediate_tokens[1].shape[0] == sum(batch.image_token_lengths)
    s.to(torch.bfloat16)
    assert s.encoder.encoder.rotary_emb.inv_freq.dtype == torch.float32
