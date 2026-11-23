"""Exact Gemma 4 E2B/E4B teacher: manifest, range-read extraction, construction dtype, verification.

The real-checkpoint check is opt-in (network + ~320 MB): ONBOARD_VLM_HUB_TESTS=1 pytest tests/test_hub.py
"""

import hashlib
import io
import os

import pytest
import torch
from safetensors.torch import save

from onboard_vlm.teachers import hub


def test_manifest_matches_pinned_teachers():
    man = hub.load_manifest()
    vt = man["vision_tower"]
    assert vt["num_tensors"] == len(vt["tensors"]) == 658
    assert vt["identical_across_checkpoints"] is True
    assert vt["digest"] == hub.set_digest({k: v["sha256"] for k, v in vt["tensors"].items()})
    assert set(man["checkpoints"]) == set(hub.SUPPORTED_TEACHERS)
    for mid, c in man["checkpoints"].items():
        spec = hub.SUPPORTED_TEACHERS[mid]
        assert c["revision"] == spec.revision and c["vision_tower_digest"] == vt["digest"]
        assert c["embed_vision"]["shape"] == [spec.text_hidden_size, 768]
    # E2B / E4B (and base / -it) differ only in embed_vision
    assert len({c["embed_vision"]["sha256"] for c in man["checkpoints"].values()}) == 4
    n_params = sum(torch.Size(v["shape"]).numel() for k, v in vt["tensors"].items() if not k.endswith(("_min", "_max")))
    assert n_params == 167_364_608


def test_range_read_subset_is_byte_exact():
    tensors = {
        "model.vision_tower.a": torch.randn(3, 5).to(torch.bfloat16),
        "model.language_model.big": torch.randn(64, 64),
        "model.vision_tower.b": torch.arange(7, dtype=torch.float32),
        "model.embed_vision.embedding_projection.weight": torch.randn(4, 3).to(torch.bfloat16),
    }
    blob = save(tensors)
    got, hashes, header = hub.read_safetensors_subset(io.BytesIO(blob), hub._keep)
    assert set(got) == {"model.vision_tower.a", "model.vision_tower.b", "model.embed_vision.embedding_projection.weight"}
    for k, v in got.items():
        assert torch.equal(v, tensors[k]) and v.dtype == tensors[k].dtype
        assert hashes[k] == hub.tensor_sha256(tensors[k])
        assert hashes[k] == hashlib.sha256(tensors[k].contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def test_build_vision_tower_keeps_rope_fp32_and_matches_hf_construction(tiny_cfg):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

    torch.manual_seed(0)
    ref = Gemma4VisionModel._from_config(tiny_cfg.vision_config, dtype=torch.bfloat16).eval()
    sd = ref.state_dict()
    ours = hub.build_vision_tower(tiny_cfg.vision_config, sd)
    assert ours.encoder.rotary_emb.inv_freq.dtype == torch.float32
    # the bug this guards against: fp32 construction + .to(bf16) rounds the RoPE frequencies
    naive = Gemma4VisionModel(tiny_cfg.vision_config).to(torch.bfloat16).eval()
    naive.load_state_dict(sd)
    assert naive.encoder.rotary_emb.inv_freq.dtype == torch.bfloat16
    restored = hub.cast_keep_rope_fp32(naive, torch.bfloat16)
    assert restored.encoder.rotary_emb.inv_freq.dtype == torch.float32
    pv = torch.rand(1, 630, 768).to(torch.bfloat16)
    from onboard_vlm.geometry import patch_positions

    pos = torch.full((1, 630, 2), -1, dtype=torch.long)
    pos[0, :63 * 9] = patch_positions(21, 27)
    with torch.no_grad():
        assert torch.equal(ref(pixel_values=pv, pixel_position_ids=pos).last_hidden_state, ours(pixel_values=pv, pixel_position_ids=pos).last_hidden_state)


def test_verify_rejects_foreign_weights(tiny_cfg):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

    with pytest.raises(RuntimeError, match="not byte-identical"):
        hub.verify_module_weights(Gemma4VisionModel(tiny_cfg.vision_config))


def test_teacher_cast_keeps_rope_fp32(tiny_teacher):
    tiny_teacher.to(torch.bfloat16)
    assert tiny_teacher.vision_tower.encoder.rotary_emb.inv_freq.dtype == torch.float32
    tiny_teacher.to(torch.float32)


@pytest.mark.skipif(not os.environ.get("ONBOARD_VLM_HUB_TESTS"), reason="set ONBOARD_VLM_HUB_TESTS=1 (downloads ~320 MB)")
@pytest.mark.parametrize("model_id", ["google/gemma-4-E2B-it", "google/gemma-4-E4B-it"])
def test_real_checkpoint_is_exact(model_id):
    from onboard_vlm.teachers.gemma4 import Gemma4Teacher

    t = Gemma4Teacher(model_id, verify=True)
    assert t.hidden_size == 768 and t.num_layers == 16
    assert sum(p.numel() for p in t.vision_tower.parameters()) == 167_364_608
    assert t.lm_hidden_size == hub.SUPPORTED_TEACHERS[model_id].text_hidden_size
