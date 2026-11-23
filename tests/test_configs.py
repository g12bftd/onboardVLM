"""Every YAML config composes, and every student preset builds (architecture only, no downloads)."""

import glob
import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from conftest import ROOT
from onboard_vlm.distill.adapter import DistillationAdapter
from onboard_vlm.distill.losses import CompositeLoss
from onboard_vlm.geometry import pad_dense
from onboard_vlm.registry import build_student

CFG = os.path.join(ROOT, "config")
STUDENTS = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(CFG, "student", "*.yaml")))
HEAVY = {"timm_mobilenetv5_300m_gemma3n", "gemma4_vit_teacher_copy"}  # >150M params: compose-only
EXPERIMENTS = sorted(os.path.relpath(p, os.path.join(CFG, "experiment"))[:-5] for p in glob.glob(os.path.join(CFG, "experiment", "**", "*.yaml"), recursive=True))

OmegaConf.register_new_resolver("join", lambda xs, sep="-": sep.join(str(x) for x in xs), replace=True)


def _compose(overrides):
    with initialize_config_dir(config_dir=CFG, version_base=None):
        return compose("defaults", overrides=overrides)


@pytest.mark.parametrize("name", STUDENTS)
def test_student_preset_builds_and_emits_contract(name):
    cfg = OmegaConf.to_container(_compose([f"student={name}"]).student, resolve=True)
    assert cfg["out_dim"] == 768
    if name in HEAVY:
        return
    if cfg["name"] == "timm_spatial":
        cfg["pretrained"] = False
    if cfg["name"] == "gemma4_vit":
        cfg["init"] = "random"
    s = build_student(cfg).eval()
    x, m, lengths = pad_dense([torch.rand(3, 144, 240)])
    with torch.no_grad():
        out = s(x, m)
    assert out.tokens.shape == (lengths[0], 768)


@pytest.mark.parametrize("exp", [e for e in EXPERIMENTS if not e.startswith("sweeps/")])
def test_experiment_composes_and_layer_map_is_valid(exp):
    over = [f"+experiment={exp}"] + (["stage1_experiment=x"] if exp.startswith("stage2/") else [])
    cfg = _compose(over)
    assert cfg.task._target_.endswith(("Stage1FeatureDistillTask", "Stage2GraftedKDTask"))
    if exp.startswith("stage1/"):
        st = OmegaConf.to_container(cfg.student, resolve=True)
        if st["name"] == "timm_spatial":
            st["pretrained"] = False
        elif st["name"] == "gemma4_vit":
            st["init"] = "random"
        s = build_student(st)
        lm = {int(k): int(v) for k, v in OmegaConf.to_container(cfg.layer_map).items()}
        DistillationAdapter.from_dims(s, 768, lm)
        assert all(0 <= t < 16 for t in lm.values())
        CompositeLoss(OmegaConf.to_container(cfg.criterion, resolve=True))


@pytest.mark.parametrize("exp", [e for e in EXPERIMENTS if e.startswith("sweeps/")])
def test_sweep_configs_compose(exp):
    cfg = _compose([f"+experiment={exp}"])
    assert "experiment_name" in cfg


@pytest.mark.parametrize("teacher", ["gemma4_e2b", "gemma4_e4b", "gemma4_e2b_base", "gemma4_e4b_base"])
def test_teacher_configs_are_pinned(teacher):
    from onboard_vlm.teachers.hub import resolve_teacher

    t = _compose([f"teacher={teacher}"]).teacher
    assert resolve_teacher(t.model_id, t.revision).revision == t.revision and t.verify is True
