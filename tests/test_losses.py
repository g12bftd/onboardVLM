"""(4) Per-image isolation of RelationalTokenLoss + other loss / adapter properties."""

import pytest
import torch

from onboard_vlm.distill.adapter import DistillationAdapter
from onboard_vlm.distill.losses import (
    CompositeLoss,
    GraftedLogitKDLoss,
    IntermediateLayerLoss,
    RelationalTokenLoss,
    TokenFeatureLoss,
)
from onboard_vlm.interfaces import StudentVisionOutput, TeacherVisionOutput


def _pair(lengths, d_s=16, d_t=16, layers=()):
    n = sum(lengths)
    s = StudentVisionOutput(torch.randn(n, d_s), lengths, {i: torch.randn(n, d_s) for i in layers})
    t = TeacherVisionOutput(torch.randn(n, d_t), lengths, {i: torch.randn(n, d_t) for i in layers})
    return s, t


def _select(out, cls, idx):
    parts = out.split()
    toks = torch.cat([parts[i] for i in idx])
    return cls(toks, [out.image_token_lengths[i] for i in idx])


@pytest.mark.parametrize("match", ["mse", "kl"])
@pytest.mark.parametrize("similarity", ["cosine", "dot"])
def test_relational_loss_is_per_image_and_order_invariant(match, similarity):
    loss = RelationalTokenLoss(similarity=similarity, match=match)
    lengths = [7, 3, 12]
    s, t = _pair(lengths)
    full = loss(s, t)
    singles = [loss(_select(s, StudentVisionOutput, [i]), _select(t, TeacherVisionOutput, [i])) for i in range(3)]
    assert torch.allclose(full, torch.stack(singles).mean(), atol=1e-6)
    perm = [2, 0, 1]
    assert torch.allclose(full, loss(_select(s, StudentVisionOutput, perm), _select(t, TeacherVisionOutput, perm)), atol=1e-6)


def test_relational_loss_never_couples_tokens_across_images():
    loss = RelationalTokenLoss()
    s, t = _pair([5, 6])
    s.tokens.requires_grad_(True)
    # only image 1's term: gradient w.r.t. image 0's tokens must be exactly zero
    term1 = loss.per_image(s.split()[1], t.split()[1])
    (g,) = torch.autograd.grad(term1, s.tokens)
    assert g[:5].abs().sum() == 0 and g[5:].abs().sum() > 0
    # and a global X X^T over the packed batch would give a different value
    glob = loss.per_image(s.tokens, t.tokens)
    assert not torch.allclose(glob, loss(s, t))


def test_token_feature_rms_is_scale_invariant_per_token():
    s, t = _pair([4, 4])
    loss = TokenFeatureLoss(metric="mse", normalize="rms")
    scaled = TeacherVisionOutput(t.tokens * torch.rand(8, 1) * 30, t.image_token_lengths)
    assert torch.allclose(loss(s, t), loss(s, scaled), atol=1e-5)
    assert TokenFeatureLoss(metric="cosine")(StudentVisionOutput(t.tokens.clone(), [4, 4]), t) < 1e-6


def test_mismatched_lengths_raise():
    s, _ = _pair([4, 4])
    _, t = _pair([5, 3])
    with pytest.raises(ValueError):
        TokenFeatureLoss()(s, t)


def test_adapter_dims_and_layer_map():
    student = type("S", (), {"out_dim": 768, "layer_dims": {3: 384, 7: 384}})()
    same = DistillationAdapter.from_dims(student, 768, {3: 5})
    assert isinstance(same.final, torch.nn.Identity)
    cross = DistillationAdapter.from_dims(student, 1152, {3: 5, 7: 26})
    s = StudentVisionOutput(torch.randn(10, 768), [10], {3: torch.randn(10, 384), 7: torch.randn(10, 384)})
    out = cross(s)
    assert out.tokens.shape == (10, 1152) and set(out.intermediate_tokens) == {5, 26}
    assert out.intermediate_tokens[26].shape == (10, 1152)
    with pytest.raises(ValueError):
        DistillationAdapter.from_dims(student, 768, {3: 5, 7: 5})


def test_composite_and_intermediate():
    s, t = _pair([6, 2], layers=[1, 4])
    crit = CompositeLoss([
        {"name": "token_feature", "weight": 1.0},
        {"name": "relational_token", "weight": 0.5, "label": "rel"},
        {"name": "intermediate_layer", "weight": 0.5, "mode": "relational"},
    ])
    total, logs = crit(s, t)
    assert set(logs) == {"token_feature_0", "rel", "intermediate_layer_2"}  # the task adds "total"
    manual = logs["token_feature_0"] + 0.5 * logs["rel"] + 0.5 * logs["intermediate_layer_2"]
    assert torch.allclose(total, manual)
    assert IntermediateLayerLoss(layers=[4])(s, t) > 0


def test_logit_kd_zero_for_identical_and_positive_otherwise():
    kd = GraftedLogitKDLoss(temperature=2.0, ce_weight=0.5)
    logits = torch.randn(2, 5, 11)
    mask = torch.tensor([[0, 1, 1, 1, 0], [0, 0, 0, 1, 0]], dtype=torch.bool)
    tgt = torch.randint(0, 11, (2, 5))
    loss, logs = kd(logits, logits, mask, tgt)
    assert logs["kd_kl"].abs() < 1e-6 and "kd_ce" in logs
    loss2, logs2 = kd(torch.randn(2, 5, 11), logits, mask, tgt)
    assert logs2["kd_kl"] > 0


def test_composite_prefix_and_description():
    s, t = _pair([3], layers=[1])
    crit = CompositeLoss([{"name": "token_feature", "label": "mse"}, {"name": "token_feature", "metric": "cosine", "weight": 0.5, "label": "cos"}])
    _, logs = crit(s, t, prefix="lm_")
    assert set(logs) == {"lm_mse", "lm_cos"}
    assert crit.describe() == "1*mse[TokenFeatureLoss] + 0.5*cos[TokenFeatureLoss]"


def test_manifold_distillation_terms():
    """Hao et al. 2022: zero for identical relations (any widths, any per-token scale), positive otherwise;
    inter-image term only for equal token counts; relations use the student's unprojected features."""
    import dataclasses

    import torch

    from onboard_vlm.distill.losses import ManifoldLoss
    from onboard_vlm.interfaces import StudentVisionOutput, TeacherVisionOutput

    torch.manual_seed(0)
    t = torch.randn(3 * 16, 32)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    rotated = (t @ q) * torch.rand(t.shape[0], 1).add(0.5)  # same relations: rotation + per-token scale
    loss = ManifoldLoss(intra=1, inter=1, random=1, num_random=20)
    teacher = TeacherVisionOutput(tokens=t, image_token_lengths=[16, 16, 16])
    same = StudentVisionOutput(tokens=rotated, image_token_lengths=[16, 16, 16])
    assert set(loss.terms(rotated, t, [16, 16, 16])) == {"intra", "inter", "random"}
    assert float(loss(same, teacher)) < 1e-10
    other = StudentVisionOutput(tokens=torch.randn(48, 8), image_token_lengths=[16, 16, 16])  # other width: no head needed
    assert float(loss(other, teacher)) > 1e-3
    assert "inter" not in loss.terms(t[:30], t[:30], [16, 14])  # unequal token counts: no cross-image term
    projected = dataclasses.replace(other, tokens=torch.zeros(48, 32), raw_tokens=rotated)  # raw features are used
    assert float(loss(projected, teacher)) < 1e-10
    # sum = squared Frobenius norm per relation matrix: grows with the token count, mean does not
    s16 = ManifoldLoss(intra=1, inter=0, random=0, reduction="sum")(other, teacher)
    m16 = ManifoldLoss(intra=1, inter=0, random=0)(other, teacher)
    assert torch.allclose(s16, m16 * 16 * 16, rtol=1e-4)


def test_undefined_loss_terms_are_skipped_in_evaluation_and_zero_in_training():
    """The inter-image term needs equal token counts: on a batch without them it neither contributes nor enters
    evaluation means, but training still logs the key (every DDP rank must log the same keys)."""
    import torch

    from onboard_vlm.distill.losses import CompositeLoss
    from onboard_vlm.interfaces import StudentVisionOutput, TeacherVisionOutput

    torch.manual_seed(0)
    crit = CompositeLoss([{"name": "token_feature", "label": "cos", "metric": "cosine"},
                          {"name": "manifold", "label": "inter", "intra": 0, "inter": 1, "random": 0},
                          {"name": "intermediate_layer", "label": "mid_inter", "mode": "manifold", "intra": 0, "inter": 1, "random": 0}])
    t = TeacherVisionOutput(torch.randn(30, 8), [16, 14], {3: torch.randn(30, 8)})
    s = StudentVisionOutput(torch.randn(30, 8), [16, 14], {3: torch.randn(30, 8)})
    crit.eval()
    total, logs = crit(s, t)
    assert set(logs) == {"cos"} and torch.allclose(total, logs["cos"])
    crit.train()
    total, logs = crit(s, t)
    assert set(logs) == {"cos", "inter", "mid_inter"} and float(logs["inter"]) == 0.0 and torch.allclose(total, logs["cos"])
