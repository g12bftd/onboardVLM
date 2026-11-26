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



# ------------------------------------------------------------------ gemma4_shrink: the teacher, made smaller
SHRINK_TINY = {"vision_hidden": 96, "vision_layers": 4}


def _shrink_batch():
    from onboard_vlm.data.collator import DualViewCollator
    from onboard_vlm.data.datasets import SyntheticVQADataset
    from onboard_vlm.geometry import target_size

    ds = SyntheticVQADataset(3, max_side=300)
    shape = target_size(*ds[0]["image"].shape[-2:], 70)
    return DualViewCollator(train=False)([{**ds[0], "batch_spec": ("full", 70, *shape)}]).teacher_inputs


def _shrink(teacher, **kw):
    from onboard_vlm.registry import build_student

    s = build_student({"name": "gemma4_shrink", "tiny": SHRINK_TINY, "out_dim": 96, "load_teacher_weights": False, **kw})
    if kw.get("init", "teacher") == "teacher":
        s.inherit_from(teacher.vision_tower.state_dict())
    return s.eval()


def test_shrink_at_full_size_is_the_teacher_and_shrinks_consistently(tmp_path):
    import torch

    from onboard_vlm.students.gemma4_shrink import select
    from onboard_vlm.students.io import load_student, save_student
    from onboard_vlm.teachers.gemma4 import Gemma4Teacher
    from onboard_vlm.utils.tiny import tiny_gemma4_config

    teacher = Gemma4Teacher(config=tiny_gemma4_config(**SHRINK_TINY), load="vision", dtype="float32")
    ti = _shrink_batch()
    run = lambda m: m(pixel_values=ti.pixel_values, pixel_position_ids=ti.pixel_position_ids).tokens  # noqa: E731
    with torch.no_grad():
        t = teacher.encode_vision(ti).tokens
        full = _shrink(teacher)
        assert full.config.to_dict() == teacher.vision_tower.config.to_dict()  # the teacher's own config
        torch.testing.assert_close(run(full), t, rtol=0, atol=0)  # weight for weight the teacher
        small = _shrink(teacher, num_hidden_layers=2, hidden_size=48)
        assert small.source_layers == [0, 3] and small.config.num_attention_heads == 1 and small.config.intermediate_size == 96
        assert run(small).shape == t.shape
        # the kept channel of a kept head is the teacher's, in every layer
        c, rows = small._sel["chans"], small._sel["heads"][0] * 48
        q_s = small.encoder.encoder.layers[1].self_attn.q_proj.linear.weight
        q_t = teacher.vision_tower.encoder.layers[3].self_attn.q_proj.linear.weight
        torch.testing.assert_close(q_s, q_t[rows : rows + 48][:, c])
        a, b = _shrink(teacher, init="random", hidden_size=48, init_seed=1), _shrink(teacher, init="random", hidden_size=48, init_seed=1)
        assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
        save_student(str(tmp_path / "exp"), small, {"name": "gemma4_shrink", "tiny": SHRINK_TINY, "out_dim": 96,
                                                    "num_hidden_layers": 2, "hidden_size": 48})
        back, _ = load_student(str(tmp_path / "exp"))
        torch.testing.assert_close(run(back.eval()), run(small))
    assert select(16, 8, "uniform") == [0, 2, 4, 6, 9, 11, 13, 15] and select(16, 4, "last") == [12, 13, 14, 15]
    with pytest.raises(ValueError):
        _shrink(teacher, hidden_size=50)  # not a whole number of heads
