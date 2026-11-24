"""E2: frozen-feature probes with one protocol for students, the teacher and external encoders.

    python -u run_probe.py encoder.kind=gemma 'encoder.student_dir=outputs/students/embedding_gemma4_vit_l8_d384_e2b' run_name=g4vit
    python -u run_probe.py encoder.kind=gemma encoder.teacher=true run_name=teacher
    python -u run_probe.py encoder.kind=timm encoder.model_name=mobilenetv4_conv_medium.e500_r256_in1k run_name=mnv4m
    python -u run_probe.py encoder.kind=hf_vision encoder.model_id=google/siglip2-base-patch16-224 run_name=siglip2_b

Tasks (``tasks=[...]``): ``imagenet`` (k-NN + linear probe on mean-pooled tokens, top-1),
``ade20k`` (linear segmentation probe, mIoU), ``nyuv2`` (linear log-depth probe, RMSE / AbsRel / d1).
Every result carries a 95% bootstrap CI over evaluation items (``results.json``).
"""

from __future__ import annotations

import json
import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from onboard_vlm.eval import probe_data, probes
from onboard_vlm.eval.stats import bootstrap_ci
from onboard_vlm.utils.config import to_plain
from onboard_vlm.utils.logging import get_logger, setup_logging, stage

log = get_logger("onboard_vlm.probe")


def build_encoder(cfg: DictConfig):
    e = cfg.encoder
    if e.kind == "timm":
        return probes.TimmGridEncoder(e.model_name, cfg.device, e.get("input_size"))
    if e.kind == "hf_vision":
        return probes.HFVisionGridEncoder(e.model_id, cfg.device)
    if e.kind != "gemma":
        raise ValueError(f"encoder.kind must be gemma | timm | hf_vision, got {e.kind}")
    dtype = getattr(torch, cfg.dtype)
    if e.get("teacher"):
        from onboard_vlm.registry import build_teacher

        model = build_teacher({**to_plain(cfg.teacher), "load": "vision"}).to(cfg.device)
        return probes.GemmaGridEncoder(model, e.budget, True, cfg.device)
    if e.get("student_dir"):
        from onboard_vlm.students.io import load_student

        model, _ = load_student(e.student_dir)
    else:
        from onboard_vlm.registry import build_student

        scfg = to_plain(e.student)
        if scfg.get("name") == "gemma4_truncated":
            scfg = {**{k: v for k, v in to_plain(cfg.teacher).items() if k in ("model_id", "revision", "tiny")}, **scfg}
        model = build_student(scfg)
    return probes.GemmaGridEncoder(model.to(device=cfg.device, dtype=dtype), e.budget, False, cfg.device)


def run_imagenet(enc, cfg, out: dict) -> None:
    c = cfg.imagenet
    make = (lambda split, pc: probe_data.WdsClassification(c[f"{split}_pattern"], pc)) if c.format == "wds" else \
        (lambda split, pc: probe_data.ImageFolderClassification(c[f"{split}_root"], pc))
    train, test = make("train", c.train_per_class), make("val", None)
    log.info("     ImageNet-1k probe: %s train (%s per class), %s val", f"{len(train):,}", c.train_per_class, f"{len(test):,}")
    tx = torch.stack(probes.extract(enc, train, cfg.batch_size, pooled=True))
    ty = torch.tensor([train.blobs[i][1] if hasattr(train, "blobs") else train.items[i][1] for i in range(len(train))])
    ex = torch.stack(probes.extract(enc, test, cfg.batch_size, pooled=True, limit=c.get("val_limit")))
    ey = torch.tensor([test.blobs[i][1] if hasattr(test, "blobs") else test.items[i][1] for i in range(len(ex))])
    knn = (probes.knn_classify(tx, ty, ex, k=c.knn_k, num_classes=1000) == ey).float().numpy()
    lin_pred, wd = probes.linear_classify(tx, ty, ex, 1000, device=cfg.device, max_iter=c.linear_max_iter, seed=cfg.seed)
    lin = (lin_pred == ey).float().numpy()
    out["imagenet"] = {"knn_top1": bootstrap_ci(knn, **cfg.bootstrap), "linear_top1": bootstrap_ci(lin, **cfg.bootstrap),
                       "linear_weight_decay": wd, "train_images": len(train), "val_images": int(len(ey))}


def run_ade20k(enc, cfg, out: dict) -> None:
    c = cfg.ade20k
    train = probe_data.ADE20KSegmentation(c.root, "training", c.train_limit)
    val = probe_data.ADE20KSegmentation(c.root, "validation", c.get("val_limit"))
    tg = probes.extract(enc, train, cfg.batch_size)
    probe = probes.train_dense_probe(tg, [train[i]["seg"] for i in range(len(train))], 150, "seg", cfg.device, c.epochs, seed=cfg.seed)
    vg = probes.extract(enc, val, cfg.batch_size)
    res = probes.eval_segmentation(probe, vg, [val[i]["seg"] for i in range(len(val))], 150, cfg.device)
    out["ade20k"] = {"miou": res["miou"], "pixel_acc": bootstrap_ci(res["pixel_acc_per_image"], **cfg.bootstrap),
                     "train_images": len(train), "val_images": len(val)}


def run_nyuv2(enc, cfg, out: dict) -> None:
    c = cfg.nyuv2
    train, test = probe_data.NYUv2Depth(c.root, "train"), probe_data.NYUv2Depth(c.root, "test")
    probe = probes.train_dense_probe(probes.extract(enc, train, cfg.batch_size), [train[i]["depth"] for i in range(len(train))], 1,
                                     "depth", cfg.device, c.epochs, seed=cfg.seed)
    res = probes.eval_depth(probe, probes.extract(enc, test, cfg.batch_size), [test[i]["depth"] for i in range(len(test))], cfg.device)
    out["nyuv2"] = {k: bootstrap_ci(v, **cfg.bootstrap) for k, v in res.items()} | {"train_images": len(train), "test_images": len(test)}


@hydra.main(config_path="config", config_name="probe", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging(HydraConfig.get().runtime.output_dir)
    log.info("===> E2 probes '%s'\n%s", cfg.run_name, OmegaConf.to_yaml(cfg, resolve=True))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    with stage(f"Loading encoder ({cfg.encoder.kind})", log):
        enc = build_encoder(cfg)
    out: dict = {"run_name": cfg.run_name, "encoder": to_plain(cfg.encoder)}
    for task in cfg.tasks:
        with stage(f"Probe {task}", log):
            {"imagenet": run_imagenet, "ade20k": run_ade20k, "nyuv2": run_nyuv2}[task](enc, cfg, out)
            log.info("     %s", json.dumps(out[task], default=str)[:400])
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    log.info("<=== E2 written to %s", cfg.output_dir)


if __name__ == "__main__":
    main()
