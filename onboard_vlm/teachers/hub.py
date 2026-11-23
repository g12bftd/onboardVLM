"""Exact Gemma 4 E2B / E4B vision-encoder weights, pinned and verified.

Facts established from the official checkpoints (see ``manifests/gemma4_vision.json``):

* ``google/gemma-4-E2B``, ``-E2B-it``, ``-E4B`` and ``-E4B-it`` each ship one ``model.safetensors``
  (10.2 GB / 16.0 GB) whose 658 ``model.vision_tower.*`` tensors (167,364,608 parameters, bf16,
  319 MB) are **byte-identical across all four checkpoints**. There is one Gemma 4 small vision
  encoder.
* Only ``model.embed_vision.embedding_projection.weight`` (the projection into the language model,
  ``[1536, 768]`` for E2B, ``[2560, 768]`` for E4B) differs per checkpoint, including base vs ``-it``.

This module reads *only* those tensors with HTTP range requests (no 10-16 GB download), hashes the
raw checkpoint bytes, and verifies them against the manifest bundled with the repository, so a
teacher can never silently load different weights. The tower is then built exactly like
``from_pretrained`` builds it (``_from_config(dtype=bf16)``: non-persistent RoPE buffers stay fp32),
which makes its outputs bitwise identical to ``Gemma4ForConditionalGeneration.get_image_features``
(checked by ``scripts/verify_gemma4_teacher.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

VISION_PREFIX = "model.vision_tower."
EMBED_VISION_KEY = "model.embed_vision.embedding_projection.weight"
MANIFEST_PATH = Path(__file__).with_name("manifests") / "gemma4_vision.json"
_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


@dataclass(frozen=True)
class TeacherSpec:
    model_id: str
    revision: str
    size: str  # "E2B" | "E4B"
    instruction_tuned: bool
    text_hidden_size: int


# Pinned commits (Hugging Face, Apache-2.0, not gated).
SUPPORTED_TEACHERS: dict[str, TeacherSpec] = {
    "google/gemma-4-E2B-it": TeacherSpec("google/gemma-4-E2B-it", "4adf15748de9e1a0097b6274cee9c15836c41c37", "E2B", True, 1536),
    "google/gemma-4-E4B-it": TeacherSpec("google/gemma-4-E4B-it", "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c", "E4B", True, 2560),
    "google/gemma-4-E2B": TeacherSpec("google/gemma-4-E2B", "433813a669639fd7eaa83155828e9000ed357d53", "E2B", False, 1536),
    "google/gemma-4-E4B": TeacherSpec("google/gemma-4-E4B", "11b8250ef59eb3ebc5dae7b21144d7b84f2c06dd", "E4B", False, 2560),
}


class UnsupportedTeacherError(ValueError):
    """Only the Gemma 4 E2B / E4B checkpoints are supported teachers."""


def resolve_teacher(model_id: str, revision: str | None = None) -> TeacherSpec:
    if model_id not in SUPPORTED_TEACHERS:
        raise UnsupportedTeacherError(
            f"{model_id!r} is not a supported teacher. This project distils exactly the Gemma 4 E2B/E4B vision "
            f"encoder; supported ids: {sorted(SUPPORTED_TEACHERS)}"
        )
    spec = SUPPORTED_TEACHERS[model_id]
    if revision is not None and revision != spec.revision:
        raise UnsupportedTeacherError(
            f"{model_id}@{revision} differs from the pinned revision {spec.revision}. Re-run "
            "scripts/verify_gemma4_teacher.py --write_manifest and update SUPPORTED_TEACHERS deliberately."
        )
    return spec


def pinned_revision(model_id: str) -> str:
    """Pinned commit of a supported checkpoint (processors / VLMs are always loaded at this revision)."""
    return resolve_teacher(model_id).revision


def tensor_sha256(t: torch.Tensor) -> str:
    """SHA-256 of a tensor's raw little-endian bytes (identical to hashing the checkpoint bytes)."""
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def set_digest(hashes: dict[str, str]) -> str:
    """Order-independent digest of a ``{name: sha256}`` set."""
    return hashlib.sha256("".join(f"{k}:{hashes[k]}\n" for k in sorted(hashes)).encode()).hexdigest()


def load_manifest(path: str | os.PathLike = MANIFEST_PATH) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# Range-read extraction from the official checkpoint
# --------------------------------------------------------------------------------------
def read_safetensors_subset(fileobj, keep) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, Any]]:
    """Read only tensors with ``keep(name)`` from an open safetensors file (local or remote).

    Byte-contiguous tensors are fetched in one read. Returns tensors, sha256 of their raw bytes,
    and the parsed header entries.
    """
    n = struct.unpack("<Q", fileobj.read(8))[0]
    header = json.loads(fileobj.read(n))
    base = 8 + n
    names = sorted((k for k in header if k != "__metadata__" and keep(k)), key=lambda k: header[k]["data_offsets"][0])
    runs: list[list[Any]] = []
    for k in names:
        s, e = header[k]["data_offsets"]
        if runs and runs[-1][1] == s:
            runs[-1][1] = e
            runs[-1][2].append(k)
        else:
            runs.append([s, e, [k]])
    tensors: dict[str, torch.Tensor] = {}
    hashes: dict[str, str] = {}
    for s0, e0, ks in runs:
        fileobj.seek(base + s0)
        blob = fileobj.read(e0 - s0)
        if len(blob) != e0 - s0:
            raise IOError(f"short read: {len(blob)} of {e0 - s0} bytes")
        for k in ks:
            s, e = header[k]["data_offsets"]
            raw = blob[s - s0 : e - s0]
            hashes[k] = hashlib.sha256(raw).hexdigest()
            h = header[k]
            tensors[k] = torch.frombuffer(bytearray(raw), dtype=_DTYPES[h["dtype"]]).reshape(h["shape"]).clone()
        del blob
    return tensors, hashes, {k: header[k] for k in names}


def _keep(k: str) -> bool:
    return k.startswith(VISION_PREFIX) or k == EMBED_VISION_KEY


def default_cache_dir() -> Path:
    return Path(os.environ.get("ONBOARD_VLM_CACHE", Path.home() / ".cache" / "onboard_vlm")) / "gemma4"


def fetch_vision_weights(model_id: str, revision: str | None = None, cache_dir: str | os.PathLike | None = None, verify: bool = True) -> Path:
    """Extract the vision tower + ``embed_vision`` of a pinned checkpoint into a local cache.

    Layout: ``<cache>/<repo>@<rev>/{config.json, vision_tower.safetensors, embed_vision.safetensors,
    extraction.json}``. Cached files are re-verified on every load when ``verify`` is set.
    """
    from huggingface_hub import HfFileSystem, hf_hub_download
    from safetensors.torch import save_file

    spec = resolve_teacher(model_id, revision)
    out = Path(cache_dir or default_cache_dir()) / f"{spec.model_id.replace('/', '--')}@{spec.revision}"
    done = out / "extraction.json"
    if done.is_file():
        return out
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = hf_hub_download(spec.model_id, "config.json", revision=spec.revision)
    (out / "config.json").write_bytes(Path(cfg_path).read_bytes())
    with HfFileSystem().open(f"{spec.model_id}/model.safetensors", "rb", revision=spec.revision, block_size=64 << 20) as f:
        tensors, hashes, header = read_safetensors_subset(f, _keep)
    vision = {k[len(VISION_PREFIX) :]: v for k, v in tensors.items() if k.startswith(VISION_PREFIX)}
    v_hashes = {k[len(VISION_PREFIX) :]: h for k, h in hashes.items() if k.startswith(VISION_PREFIX)}
    if verify:
        _check_against_manifest(spec, v_hashes, hashes[EMBED_VISION_KEY])
    save_file(vision, str(out / "vision_tower.safetensors"))
    save_file({"embedding_projection.weight": tensors[EMBED_VISION_KEY]}, str(out / "embed_vision.safetensors"))
    done.write_text(json.dumps({
        "model_id": spec.model_id, "revision": spec.revision, "vision_tower_digest": set_digest(v_hashes),
        "embed_vision_sha256": hashes[EMBED_VISION_KEY], "tensors": {k: header[k]["dtype"] for k in header},
    }, indent=1))
    return out


def _check_against_manifest(spec: TeacherSpec, vision_hashes: dict[str, str], embed_sha: str) -> None:
    man = load_manifest()
    ref = {k: v["sha256"] for k, v in man["vision_tower"]["tensors"].items()}
    if vision_hashes != ref:
        bad = sorted(k for k in set(ref) | set(vision_hashes) if ref.get(k) != vision_hashes.get(k))
        raise RuntimeError(f"{spec.model_id}@{spec.revision}: vision tower differs from the manifest in {len(bad)} tensors, e.g. {bad[:3]}")
    exp = man["checkpoints"][spec.model_id]["embed_vision"]["sha256"]
    if embed_sha != exp:
        raise RuntimeError(f"{spec.model_id}: embed_vision sha256 {embed_sha} != manifest {exp}")


def verify_module_weights(vision_tower: torch.nn.Module, embed_projection: torch.Tensor | None = None, model_id: str | None = None) -> None:
    """Hash every persistent tensor of a loaded tower (and ``embed_vision``) against the manifest."""
    man = load_manifest()
    ref = {k: v["sha256"] for k, v in man["vision_tower"]["tensors"].items()}
    got = {k: tensor_sha256(v) for k, v in vision_tower.state_dict().items()}
    if got != ref:
        bad = sorted(k for k in set(ref) | set(got) if ref.get(k) != got.get(k))
        raise RuntimeError(f"loaded vision tower is not byte-identical to Gemma 4 E2B/E4B ({len(bad)} tensors differ, e.g. {bad[:3]})")
    if embed_projection is not None and model_id is not None:
        exp = man["checkpoints"][model_id]["embed_vision"]["sha256"]
        if tensor_sha256(embed_projection) != exp:
            raise RuntimeError(f"embed_vision of {model_id} is not byte-identical to the checkpoint")


def build_vision_tower(config: Any, state_dict: dict[str, torch.Tensor], dtype: torch.dtype = torch.bfloat16, attn_implementation: str = "sdpa") -> torch.nn.Module:
    """``Gemma4VisionModel`` constructed as ``from_pretrained`` does, then strictly loaded.

    ``_from_config(dtype=...)`` keeps non-persistent RoPE frequencies in fp32. Building in fp32 and
    calling ``.to(bfloat16)`` would round ``inv_freq`` to bf16 and change outputs (max abs error 512 on
    real tokens of magnitude ~3.7e4).
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

    vc = config.__class__.from_dict(config.to_dict())
    vc._attn_implementation = attn_implementation
    model = Gemma4VisionModel._from_config(vc, dtype=dtype)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def load_exact_vision_tower(model_id: str, revision: str | None = None, cache_dir: str | os.PathLike | None = None, compute_dtype: str | torch.dtype = "bfloat16", verify: bool = True, attn_implementation: str = "sdpa"):
    """Returns ``(vision_tower, embed_projection_weight, vision_config, spec)`` for a pinned E2B/E4B checkpoint.

    ``compute_dtype="bfloat16"`` reproduces the deployed VLM bitwise; ``"float32"`` upcasts the bf16
    weights losslessly (the stored values are unchanged, only arithmetic precision increases).
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig

    spec = resolve_teacher(model_id, revision)
    path = fetch_vision_weights(model_id, spec.revision, cache_dir, verify=verify)
    cfg = AutoConfig.from_pretrained(str(path))
    vision_sd = load_file(str(path / "vision_tower.safetensors"))
    embed_w = load_file(str(path / "embed_vision.safetensors"))["embedding_projection.weight"]
    tower = build_vision_tower(cfg.vision_config, vision_sd, dtype=torch.bfloat16, attn_implementation=attn_implementation)
    if verify:
        verify_module_weights(tower, embed_w, spec.model_id)
    dt = compute_dtype if isinstance(compute_dtype, torch.dtype) else getattr(torch, compute_dtype)
    if dt != torch.bfloat16:
        tower = cast_keep_rope_fp32(tower, dt)
    return tower, embed_w, cfg.vision_config, spec


def cast_keep_rope_fp32(model: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    """``model.to(dtype)`` but restore fp32 RoPE ``inv_freq`` buffers afterwards."""
    model = model.to(dtype)
    restore_rope_fp32(model)
    return model


def restore_rope_fp32(model: torch.nn.Module) -> None:
    for mod in model.modules():
        if hasattr(mod, "compute_axial_rope_parameters") and hasattr(mod, "inv_freq"):
            inv, _ = mod.compute_axial_rope_parameters(mod.config)
            mod.inv_freq = inv.to(device=mod.inv_freq.device, dtype=torch.float32)
            mod.original_inv_freq = mod.inv_freq.clone()
