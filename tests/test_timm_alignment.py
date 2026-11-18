"""(2) Exact 48x spatial alignment of TimmSpatialStudent (no spatial drift)."""

import pytest
import timm
import torch
import torch.nn.functional as F
from torch import nn

from onboard_vlm.geometry import pad_dense
from onboard_vlm.students.timm_spatial import TimmSpatialStudent

FAMILIES = ["mobilenetv4_conv_small", "mobilenetv4_hybrid_medium", "fastvit_t8", "efficientvit_b0", "tiny_vit_5m_224"]
SIZES = [(48, 48), (96, 528), (336, 144), (144, 480), (672, 912)]


@pytest.mark.parametrize("alignment", ["stem_stride3", "tap16_conv3"])
@pytest.mark.parametrize("name", FAMILIES)
def test_output_grid_is_exactly_h48_w48(name, alignment):
    s = TimmSpatialStudent(name, out_dim=16, pretrained=False, alignment=alignment).eval()
    assert all(48 % r == 0 for r in s.stage_strides)
    for h, w in SIZES:
        with torch.no_grad():
            y, _ = s._final_map(torch.zeros(1, 3, h, w))
        assert tuple(y.shape[-2:]) == (h // 48, w // 48), (name, alignment, h, w)
        x, m, lengths = pad_dense([torch.rand(3, h, w)])
        with torch.no_grad():
            out = s(x, m, return_layers=sorted(s.layer_dims))
        assert out.tokens.shape == (lengths[0], 16)
        assert all(v.shape[0] == lengths[0] for v in out.intermediate_tokens.values())


def _circular(model: nn.Module) -> nn.Module:
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d) and any(isinstance(p, int) and p > 0 for p in mod.padding):
            mod.padding_mode = "circular"
    return model


@pytest.mark.parametrize("alignment", ["stem_stride3", "tap16_conv3"])
@pytest.mark.parametrize("name", ["mobilenetv4_conv_small", "fastvit_t8"])
def test_one_token_shift_equivariance(name, alignment):
    """With circular padding a pure-conv student is exactly shift-equivariant: moving the image by
    48 px must move the token map by exactly one token. Any resampling / misaligned stride breaks it."""
    s = _circular(TimmSpatialStudent(name, out_dim=16, pretrained=False, alignment=alignment)).eval()
    f = lambda z: s.head(s._final_map(s.input_norm(z))[0])  # noqa: E731
    x = torch.rand(1, 3, 144, 480)
    with torch.no_grad():
        a, b = f(x), f(torch.roll(x, 48, dims=-1))
        c = f(torch.roll(x, 48, dims=-2))
    assert torch.allclose(torch.roll(a, 1, dims=-1), b, atol=1e-5)
    assert torch.allclose(torch.roll(a, 1, dims=-2), c, atol=1e-5)


def test_stride32_plus_interpolation_drifts():
    """Negative control: the forbidden stride-32 + bilinear approach is NOT one-token equivariant."""
    m = _circular(timm.create_model("mobilenetv4_conv_small", pretrained=False, features_only=True)).eval()
    f = lambda z: F.interpolate(m(z)[-1], size=(z.shape[-2] // 48, z.shape[-1] // 48), mode="bilinear", align_corners=False)  # noqa: E731
    x = torch.rand(1, 3, 144, 480)
    with torch.no_grad():
        a, b = f(x), f(torch.roll(x, 48, dims=-1))
    assert not torch.allclose(torch.roll(a, 1, dims=-1), b, atol=1e-2)


def test_stem_surgery_only_touches_first_downsampling_block():
    s = TimmSpatialStudent("fastvit_t8", out_dim=16, pretrained=False)
    assert sorted(s.modified_modules) == ["stem_0.conv_kxk.0.conv", "stem_0.conv_scale.conv"]
    strides3 = [n for n, m in s.backbone.named_modules() if isinstance(m, nn.Conv2d) and m.stride == (3, 3)]
    assert sorted(strides3) == sorted(s.modified_modules)
