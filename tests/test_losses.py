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
    assert set(logs) == {"token_feature_0", "rel", "intermediate_layer_2", "total"}
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
