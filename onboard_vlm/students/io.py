"""Student checkpoint export / import: ``<dir>/student.safetensors`` + ``<dir>/student_config.json``."""

from __future__ import annotations

import json
import os
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from onboard_vlm.registry import build_student


def save_student(directory: str, student: nn.Module, config: dict[str, Any]) -> None:
    """``config`` is the registry config used to build the student (``{"name": ..., **kwargs}``)."""
    os.makedirs(directory, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in student.state_dict().items()}
    save_file(state, os.path.join(directory, "student.safetensors"))
    with open(os.path.join(directory, "student_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_student(directory: str, map_location: str | torch.device = "cpu", **overrides: Any) -> tuple[nn.Module, dict[str, Any]]:
    with open(os.path.join(directory, "student_config.json")) as f:
        config = json.load(f)
    config = {**config, **overrides}
    if config.get("name") == "timm_spatial":
        config["pretrained"] = False  # weights come from the checkpoint
    if config.get("name") == "gemma4_truncated":
        config["load_weights"] = False  # weights come from the checkpoint (no teacher download)
    student = build_student(config)
    student.load_state_dict(load_file(os.path.join(directory, "student.safetensors"), device=str(map_location)))
    return student, config
