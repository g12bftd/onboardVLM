"""Profile vision encoders: params, MACs/FLOPs and peak activation memory at Gemma token budgets.

    python run_profile.py                                       # defaults in config/profile.yaml
    python run_profile.py device=cuda dtype=bfloat16 latency_iters=50
    python run_profile.py 'students=[compact_vit_s,timm_mobilenetv4_conv_m]' 'token_budgets=[140,280,560,1120]'
"""

from __future__ import annotations

import json
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from onboard_vlm.registry import build_student
from onboard_vlm.utils.profiler import format_table, profile_encoder, rows_to_dicts

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")


def student_cfg(name: str, deploy_dim: int) -> dict:
    """Resolve config/student/<name>.yaml through Hydra (presets inherit from family base configs)."""
    from hydra import compose

    node = compose(config_name="defaults", overrides=[f"student={name}", f"deploy_vision_dim={deploy_dim}"])
    cfg = OmegaConf.to_container(node.student, resolve=True)
    if cfg.get("name") == "timm_spatial":
        cfg["pretrained"] = False  # architecture only
    return cfg


@hydra.main(config_path="config", config_name="profile", version_base=None)
def main(cfg: DictConfig) -> None:
    dtype = getattr(torch, cfg.dtype)
    kw = dict(token_budgets=list(cfg.token_budgets), source_hw=tuple(cfg.source_hw), device=cfg.device, dtype=dtype, latency_iters=cfg.latency_iters)
    rows = []
    if cfg.include_teacher:
        from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        vc = Gemma4VisionConfig(**(OmegaConf.to_container(cfg.teacher_vision_config) if cfg.teacher_vision_config else {}))
        rows += profile_encoder(Gemma4VisionModel(vc), f"gemma4_vision_{vc.hidden_size}d_{vc.num_hidden_layers}L", kind="gemma", **kw)
    for name in cfg.students:
        rows += profile_encoder(build_student(student_cfg(name, cfg.deploy_vision_dim)), name, **kw)
    print(format_table(rows))
    os.makedirs(os.path.dirname(os.path.abspath(cfg.output)), exist_ok=True)
    with open(cfg.output, "w") as f:
        json.dump(rows_to_dicts(rows), f, indent=2)


if __name__ == "__main__":
    main()
