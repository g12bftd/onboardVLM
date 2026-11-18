"""Gemma 4 geometry: bit-exact parity with Gemma4ImageProcessor and lossless (un)patchify."""

import random

import pytest
import torch
from transformers import Gemma4ImageProcessor

from onboard_vlm import geometry as G
from onboard_vlm.interfaces import TransformParams

SIZES = [(480, 640), (1080, 1920), (300, 1200), (2000, 500), (97, 131), (48, 48)]


@pytest.mark.parametrize("budget", [70, 140, 280, 560, 1120])
def test_patches_match_hf_processor_bit_exactly(budget):
    proc = Gemma4ImageProcessor(max_soft_tokens=budget)
    for hw in SIZES:
        img = torch.randint(0, 256, (3, *hw), dtype=torch.uint8)
        ref = proc([img], return_tensors="pt")
        p = G.sample_transform(hw, budget, train=False)
        out = G.apply_transform(img, p)
        pv, pos = G.build_teacher_patches([G.to_unit_float(out)], max_patches=budget * 9)
        assert torch.equal(pv, ref["pixel_values"]), (budget, hw)
        assert torch.equal(pos, ref["image_position_ids"]), (budget, hw)
        assert p.num_tokens == int(ref["num_soft_tokens_per_image"][0])
        assert p.target_hw[0] % 48 == 0 and p.target_hw[1] % 48 == 0


def test_unpatchify_roundtrip_and_mask():
    imgs = [torch.rand(3, 96, 144), torch.rand(3, 240, 48)]
    pv, pos = G.build_teacher_patches(imgs, max_patches=70 * 9)
    dense, mask, sizes = G.unpatchify_batch(pv, pos)
    assert sizes == [(96, 144), (240, 48)]
    assert torch.equal(dense[0, :, :96, :144], imgs[0]) and torch.equal(dense[1, :, :240, :48], imgs[1])
    assert dense[0, :, 96:].abs().sum() == 0
    assert mask.shape == (2, 5, 3) and mask.sum(dim=(1, 2)).tolist() == [6, 5]


def test_train_transform_is_deterministic_given_params():
    rng = random.Random(3)
    img = torch.randint(0, 256, (3, 400, 600), dtype=torch.uint8)
    p = G.sample_transform((400, 600), 280, train=True, rng=rng, hflip_prob=1.0)
    assert p.crop_box is not None and p.hflip
    p2 = TransformParams.from_dict(p.to_dict())
    assert torch.equal(G.apply_transform(img, p), G.apply_transform(img, p2))
    assert tuple(G.apply_transform(img, p).shape[-2:]) == p.target_hw


def test_pad_dense_and_flatten_order():
    imgs = [torch.rand(3, 96, 144), torch.rand(3, 48, 96)]
    x, m, lengths = G.pad_dense(imgs)
    assert x.shape == (2, 3, 96, 144) and lengths == [6, 2]
    grid = torch.arange(2 * 2 * 3).view(2, 2, 3, 1).float()
    flat = G.flatten_valid(grid, m)
    # image 0: all 6 cells row-major; image 1: its top-left 1x2 rectangle
    assert flat.squeeze(-1).tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
