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
HEAVY = {"timm_mobilenetv5_300m_gemma3n", "gemma4_truncated", "gemma4_truncated_l8", "gemma4_shrink"}  # >150M params or needs the checkpoint: compose-only
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
    assert exp.split("/")[0] in ("debug", "baselines", "shrink", "lit", TASK_CRITERIA[task])  # grouped by distillation type (or study)
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
    if st["name"] in HEAVY or st.get("model_name", "").startswith("mobilenetv5"):
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
        assert roots == {s["dir"] for s in spec["sources"] if not s.get("optional")}, dm  # every required source is trained on
        assert "imagenet1k" in roots  # default objects source (ImageNet-21k is the optional alternative)
        assert abs(sum(d["mixture"].values()) - 1.0) < 1e-9 and abs(sum(t["weight"] for t in d["train"]) - 1.0) < 1e-9
        assert d["dataset_defaults"]["image_corpus"]["require_finalized"] is True  # one split definition for every entry
        assert {t["group"] for t in d["val"]} == {t["group"] for t in d["train"]}  # every train group is monitored


@pytest.mark.parametrize("teacher", ["gemma4_e2b", "gemma4_e4b", "gemma4_e2b_base", "gemma4_e4b_base"])
def test_teacher_configs_are_pinned(teacher):
    from onboard_vlm.teachers.gemma4 import SUPPORTED_TEACHERS

    t = _compose([f"teacher={teacher}"]).teacher
    assert t.name == "gemma4" and t.revision == SUPPORTED_TEACHERS[t.model_id] and t.load == "vision"


def test_corpus_split_is_defined_once():
    """generate_targets.yaml and fidelity.yaml must use the training split definition (no train/test leakage)."""
    with initialize_config_dir(config_dir=CFG, version_base=None):
        ref = OmegaConf.to_container(compose("defaults").corpus_split)
        gv = OmegaConf.to_container(compose("defaults", overrides=["data_module=general_vision"]).data_module.dataset_defaults.image_corpus, resolve=True)
        gen = OmegaConf.to_container(compose("generate_targets").corpus_split)
        fid = OmegaConf.to_container(compose("fidelity").corpus_split)
        objects = {compose(n).objects_source for n in ("defaults", "generate_targets", "fidelity")}
        swapped = OmegaConf.to_container(compose("defaults", overrides=["data_module=general_vision", "objects_source=imagenet21k", "corpus_root=/C"]).data_module, resolve=True)
    assert gen == fid == ref and {k: gv[k] for k in ref} == ref
    assert objects == {"imagenet1k"}
    assert any(t.get("root") == "/C/imagenet21k" and t.get("group") == "objects" for t in swapped["train"])


def test_main_protocol_defaults():
    cfg = _compose([])
    assert cfg.task._target_.endswith("EmbeddingDistillationTask")  # main method
    assert list(cfg.input_domain.budgets) == [70, 140, 280, 560, 1120] and list(cfg.train_budgets) == list(cfg.input_domain.budgets)
    dm = OmegaConf.to_container(_compose(["data_module=general_vision"]).data_module, resolve=True)
    assert dm["sampling"]["mode"] == "crop" and dm["evaluation"]["test_budgets"] == [70, 140, 280, 560, 1120]
    assert _compose(["trainer=gpu_ddp"]).trainer.use_distributed_sampler is False
    assert "_s0" in cfg.run_id and cfg.run_id in cfg.export_dir


@pytest.mark.parametrize("name", ["fidelity", "probe", "generate_targets", "eval"])
def test_evaluation_configs_compose(name):
    with initialize_config_dir(config_dir=CFG, version_base=None):
        cfg = compose(name)
    OmegaConf.to_container(cfg, resolve=True)


STUDIES = ["identity", "lr", "target", "input", "objective", "md_scale", "mixup", "multiplex", "depth", "width", "init", "grid"]


@pytest.mark.parametrize("study", STUDIES)
def test_study_launcher_names_are_the_run_ids(study):
    """The launcher decides "done" from the run name it predicts: it must be exactly the run_id the overrides produce."""
    import subprocess

    import onboard_vlm  # noqa: F401  (registers the seed_tag resolver)

    env = {**os.environ, "ONBOARD_VLM_ROOT": "/nonexistent", "ONBOARD_VLM_OUT": "/nonexistent/out"}
    out = subprocess.run(["bash", os.path.join(ROOT, "scripts", "slurm", "submit_shrink_study.sh"), "--study", study, "--list"],
                         capture_output=True, text=True, env=env, check=True).stdout.split("\n")
    lines = [ln for ln in out if ln.strip()]
    assert lines
    for ln in lines:
        name, overrides = ln.split("|")
        cfg = _compose(overrides.split() + ["seed=0"])
        assert cfg.run_id == f"{name}_s0", (cfg.run_id, name)
    assert all(f"_{_compose(ln.split('|')[1].split()).teacher.key}_" in ln.split("|")[0] for ln in lines)  # teacher in the name
    if study == "target":  # the target space x VLM design: vision tokens, E2B's and E4B's LM-input space
        got = {(c.teacher.key, c.objective_name) for c in (_compose(ln.split("|")[1].split()) for ln in lines)}
        assert got == {("e2b_it", "mseV"), ("e2b_it", "mse"), ("e4b_it", "mse")}
    if study == "objective":  # every objective of the literature study, one at a time
        assert {_compose(ln.split("|")[1].split()).objective_name.split("_w")[0] for ln in lines} == {"mse", "fm", "md", "fm_md", "fm_fmL", "fm_mdL"}
    if study not in ("input", "mixup"):  # the ladder starts from the simplest recipe: whole images, one budget, no mixup
        for ln in lines:
            cfg = _compose(ln.split("|")[1].split())
            assert cfg.input_name == "full280" and cfg.data_module.sampling.mode == "full" and list(cfg.train_budgets) == [280]
            assert float(cfg.data_module.collator.get("mixup_alpha", 0.0)) == 0.0


@pytest.mark.parametrize("objective", sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(CFG, "objective", "*.yaml"))))
def test_objectives_build_their_losses(objective):
    """Each objective builds; relation-only intermediate terms come without projection heads (no unused DDP parameters)."""
    import onboard_vlm  # noqa: F401

    cfg = OmegaConf.to_container(_compose(["+experiment=lit/base", f"objective={objective}"]), resolve=True)
    assert cfg["task"]["_target_"].endswith("EmbeddingDistillationTask")
    CompositeLoss(cfg["criterion"])
    vc = cfg["task"].get("vision_criterion") or []
    if vc:
        CompositeLoss(vc)
    uses_layers = any(t["name"] == "intermediate_layer" for t in vc)
    assert (cfg["layer_map"] == "auto") == uses_layers
    relation_only = uses_layers and all(t.get("mode") == "manifold" for t in vc if t["name"] == "intermediate_layer")
    assert cfg["task"]["adapter"].get("layer_heads", True) is (not relation_only)
    assert cfg["data_module"]["collator"]["mixup_alpha"] == 0.0 and cfg["input_name"] == "full280"  # the simplest recipe


def test_study_launcher_keeps_e2b_and_e4b_runs_apart():
    """--teacher changes every run name (the launcher would otherwise take E2B outputs for finished E4B runs), and
    --only selects configurations by name."""
    import subprocess

    env = {**os.environ, "ONBOARD_VLM_ROOT": "/nonexistent", "ONBOARD_VLM_OUT": "/nonexistent/out"}
    sh = os.path.join(ROOT, "scripts", "slurm", "submit_shrink_study.sh")
    names = {t: {ln.split("|")[0] for ln in subprocess.run(["bash", sh, "--study", "objective", "--teacher", t, "--list"], capture_output=True,
                                                           text=True, env=env, check=True).stdout.split("\n") if ln} for t in ("e2b", "gemma4_e4b")}
    assert len(names["e2b"]) == len(names["gemma4_e4b"]) == 6 and not names["e2b"] & names["gemma4_e4b"]
    only = subprocess.run(["bash", sh, "--study", "objective", "--only", "*_fm_mx*", "--list"], capture_output=True, text=True, env=env,
                          check=True).stdout.split()
    assert [ln.split("|")[0] for ln in only if "|" in ln] == ["lit_e2b_it_full280_fm_mx0.0_lr0.001_L8_D768_x1-none"]
