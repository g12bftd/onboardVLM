"""Gemma 4-architecture student: the unchanged transformers Gemma4VisionModel at a smaller size."""

import pytest
import torch

from onboard_vlm.data.collator import DualViewCollator
from onboard_vlm.students.gemma4_vit import Gemma4ViTStudent


@pytest.fixture(scope="module")
def batch():
    imgs = [{"sample_id": str(i), "image": torch.randint(0, 256, (3, *hw), dtype=torch.uint8)} for i, hw in enumerate([(480, 640), (300, 1200), (97, 131)])]
    return DualViewCollator(70)(imgs)


def _teacher_sized_student(teacher):
    vc = teacher.vision_tower.config
    s = Gemma4ViTStudent(
        out_dim=vc.hidden_size, hidden_size=vc.hidden_size, num_hidden_layers=vc.num_hidden_layers, num_attention_heads=vc.num_attention_heads,
        head_dim=vc.head_dim, intermediate_size=vc.intermediate_size, position_embedding_size=vc.position_embedding_size,
        use_clipped_linears=vc.use_clipped_linears, rms_norm_eps=vc.rms_norm_eps, rope_theta=vc.rope_parameters["rope_theta"],
        attn_implementation="eager",
    )
    s.encoder.load_state_dict(teacher.vision_tower.state_dict())
    return s.eval()


def test_student_is_the_gemma4_vision_architecture(tiny_teacher, batch):
    """At the teacher's size and with the teacher's weights, the student reproduces the teacher exactly:
    same module code, same patch input, same pooling, same token order."""
    s = _teacher_sized_student(tiny_teacher)
    ref = tiny_teacher.encode_vision(batch.teacher_inputs, layers=[0, 2])
    with torch.no_grad():
        out = s(pixel_values=batch.teacher_inputs.pixel_values, pixel_position_ids=batch.teacher_inputs.pixel_position_ids, return_layers=[0, 2])
        dense = s(batch.student_images, batch.valid_token_mask)
    assert out.image_token_lengths == ref.image_token_lengths
    torch.testing.assert_close(out.tokens, ref.tokens, rtol=1e-5, atol=1e-5)
    for i in (0, 2):
        torch.testing.assert_close(out.intermediate_tokens[i], ref.intermediate_tokens[i], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dense.tokens, ref.tokens, rtol=1e-4, atol=1e-4)  # dense path rebuilds the same patches


@pytest.mark.parametrize("layers,hidden,heads", [(2, 32, None), (3, 64, 2), (1, 48, 4)])
def test_smaller_students(batch, layers, hidden, heads):
    s = Gemma4ViTStudent(out_dim=48, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads, head_dim=16, position_embedding_size=64)
    assert s.config.num_hidden_layers == layers and s.config.num_attention_heads == (heads or hidden // 16)
    assert set(s.layer_dims) == set(range(layers)) and isinstance(s.head, torch.nn.Linear if hidden != 48 else torch.nn.Identity)
    out = s(batch.student_images, batch.valid_token_mask, return_layers=[layers - 1])
    n = sum(batch.image_token_lengths)
    assert out.tokens.shape == (n, 48) and out.intermediate_tokens[layers - 1].shape == (n, hidden)
    out.tokens.square().mean().backward()
    assert all(p.grad is not None for p in s.parameters())
    with pytest.raises(KeyError):
        s(batch.student_images, batch.valid_token_mask, return_layers=[layers])

