"""(3) True parameter sharing in MultiplexedViT (numel verification)."""

import pytest
import torch

from onboard_vlm.registry import build_student
from onboard_vlm.students.common import num_params
from onboard_vlm.students.multiplexed_vit import MultiplexedViT, SharedBlock, StepTransform

CFG = dict(out_dim=64, embed_dim=48, num_heads=4, mlp_ratio=4.0, num_shared_blocks=2, steps_per_block=3)


@pytest.mark.parametrize("transform", ["minivit", "lowrank", "diag", "none", ["head_mix", "dwconv", "lowrank", "diag"]])
def test_parameter_count_reflects_sharing(transform):
    m = MultiplexedViT(step_transform=transform, **CFG)
    trunk = m.blocks
    one_shared = num_params(SharedBlock(48, 4, 4.0, True))
    assert num_params(trunk.shared) == CFG["num_shared_blocks"] * one_shared  # stored once per group, not per step
    assert len(trunk.steps) == CFG["num_shared_blocks"] * CFG["steps_per_block"]
    # every parameter is registered exactly once (no aliasing / cloned copies)
    named_all = list(m.named_parameters(remove_duplicate=False))
    assert len(named_all) == len(list(m.parameters()))
    ptrs = [p.data_ptr() for _, p in named_all]
    assert len(ptrs) == len(set(ptrs))
    others = num_params(m) - num_params(trunk)
    assert num_params(m) == others + CFG["num_shared_blocks"] * one_shared + num_params(trunk.steps)
    # per-step parameters are lightweight relative to a shared block
    assert num_params(trunk.steps[0]) < 0.2 * one_shared


def test_much_smaller_than_unshared_vit_of_same_depth():
    mux = MultiplexedViT(**CFG)
    std = build_student({"name": "compact_vit", "out_dim": 64, "embed_dim": 48, "num_heads": 4, "depth": 6})
    assert num_params(mux) < 0.6 * num_params(std)


def test_same_module_object_runs_every_step():
    m = MultiplexedViT(step_transform="none", **CFG).eval()
    calls = {id(b): 0 for b in m.blocks.shared}

    def counter(key):
        return lambda mod, inp, out: calls.__setitem__(key, calls[key] + 1)

    hooks = [b.mlp.fc1.register_forward_hook(counter(id(b))) for b in m.blocks.shared]
    x = torch.rand(1, 3, 96, 96)
    with torch.no_grad():
        m(x, torch.ones(1, 2, 2, dtype=torch.bool))
    for h in hooks:
        h.remove()
    assert list(calls.values()) == [CFG["steps_per_block"]] * CFG["num_shared_blocks"]


def test_identity_initialised_transforms_equal_plain_sharing():
    """Zero/identity-initialised step transforms start exactly at plain weight sharing."""
    torch.manual_seed(0)
    a = MultiplexedViT(step_transform="minivit", **CFG).eval()
    b = MultiplexedViT(step_transform="none", **CFG).eval()
    b.load_state_dict({k: v for k, v in a.state_dict().items() if k in b.state_dict()})
    x = torch.rand(2, 3, 96, 144)
    m = torch.ones(2, 2, 3, dtype=torch.bool)
    with torch.no_grad():
        assert torch.allclose(a(x, m).tokens, b(x, m).tokens, atol=1e-5)
    assert isinstance(a.blocks.steps[0], StepTransform)
