"""Every YAML config composes, and every student preset builds (architecture only, no downloads)."""

import glob
import os

import pytest
import torch
from conftest import ROOT
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from onboard_vlm.distill.adapter import DistillationAdapter
from onboard_vlm.distill.losses import CompositeLoss
from onboard_vlm.geometry import pad_dense
from onboard_vlm.registry import build_student

CFG = os.path.join(ROOT, "config")
STUDENTS = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(CFG, "student", "*.yaml")))
HEAVY = {"timm_mobilenetv5_300m_gemma3n"}  # >150M params: compose-only
DATA_MODULES = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(CFG, "data_module", "*.yaml")))
TASK_CRITERIA = {"FeatureDistillationTask": "feature", "EmbeddingDistillationTask": "embedding", "LogitDistillationTask": "logit"}
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
    s = build_student(cfg).eval()
    x, m, lengths = pad_dense([torch.rand(3, 144, 240)])
    with torch.no_grad():
        out = s(x, m)
    assert out.tokens.shape == (lengths[0], 768)


@pytest.mark.parametrize("exp", [e for e in EXPERIMENTS if not e.startswith("sweeps/")])
def test_experiment_composes_and_task_matches_criterion(exp):
    cfg = _compose([f"+experiment={exp}", "student_dir=x"])
    task = cfg.task._target_.rsplit(".", 1)[-1]
    assert task in TASK_CRITERIA
    assert exp.split("/")[0] in ("debug", TASK_CRITERIA[task])  # experiments are grouped by distillation type
    crit = OmegaConf.to_container(cfg.criterion, resolve=True)
    OmegaConf.to_container(cfg.data_module, resolve=True)  # every interpolation resolves
    if task == "LogitDistillationTask":
        assert crit["name"] == "grafted_logit_kd"
        return
    assert isinstance(crit, list) and crit
    if task == "EmbeddingDistillationTask":
        assert not any(t["name"] == "intermediate_layer" for t in crit)
    if exp.startswith("debug/"):
        return
    st = OmegaConf.to_container(cfg.student, resolve=True)
    if st["name"] == "timm_spatial":
        st["pretrained"] = False
    if st["name"] in HEAVY:
        return
    s = build_student(st)
    lm = {int(k): int(v) for k, v in OmegaConf.to_container(cfg.layer_map).items()}
    DistillationAdapter.from_dims(s, 768, lm)
    assert all(0 <= t < 16 for t in lm.values())
    CompositeLoss(crit)


@pytest.mark.parametrize("exp", [e for e in EXPERIMENTS if e.startswith("sweeps/")])
def test_sweep_configs_compose(exp):
    cfg = _compose([f"+experiment={exp}"])
    assert "experiment_name" in cfg


@pytest.mark.parametrize("dm", DATA_MODULES)
def test_data_module_configs_resolve(dm):
    d = OmegaConf.to_container(_compose([f"data_module={dm}"]).data_module, resolve=True)
    assert d["_target_"].endswith("DistillDataModule") and d["train"]


def test_general_vision_matches_corpus_definition():
    import yaml

    spec = yaml.safe_load(open(os.path.join(ROOT, "make_datasets", "corpus", "sources.yaml")))
    for dm in ("general_vision", "general_vision_lite"):
        d = OmegaConf.to_container(_compose([f"data_module={dm}", "corpus_root=/C"]).data_module, resolve=True)
        roots = {t["root"][len("/C/"):] for t in d["train"] if t["name"] == "image_corpus"}
        assert roots == {s["dir"] for s in spec["sources"]}, dm  # every built source is trained on, nothing else
        assert abs(sum(d["mixture"].values()) - 1.0) < 1e-9 and abs(sum(t["weight"] for t in d["train"]) - 1.0) < 1e-9
        assert d["dataset_defaults"]["image_corpus"]["require_finalized"] is True  # one split definition for every entry
        assert {t["group"] for t in d["val"]} == {t["group"] for t in d["train"]}  # every train group is monitored


@pytest.mark.parametrize("teacher", ["gemma4_e2b", "gemma4_e4b", "gemma4_e2b_base", "gemma4_e4b_base"])
def test_teacher_configs_are_pinned(teacher):
    from onboard_vlm.teachers.gemma4 import SUPPORTED_TEACHERS

    t = _compose([f"teacher={teacher}"]).teacher
    assert t.name == "gemma4" and t.revision == SUPPORTED_TEACHERS[t.model_id] and t.load == "vision"
