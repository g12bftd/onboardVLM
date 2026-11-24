"""Distillation entry point (Hydra + PyTorch Lightning, TimeFM `cerebro_v2` conventions).

    python -u run_distill.py +experiment=feature/gemma4_vit_l8_d384_e2b
    python -u run_distill.py +experiment=embedding/gemma4_vit_l8_d384_e2b
    python -u run_distill.py +experiment=logit/gemma4_vit_l8_d384_e2b student_dir=outputs/students/feature_gemma4_vit_l8_d384_e2b
    python -u run_distill.py +experiment=debug/dryrun_feature          # offline sanity check

The distillation type is the task (config/task/*.yaml): feature_distillation | embedding_distillation |
logit_distillation. Every stage is logged in real time to stdout and <hydra run dir>/run_rank*.log.
"""

from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from onboard_vlm.utils.logging import get_logger, setup_logging, stage

OmegaConf.register_new_resolver("join", lambda xs, sep="-": sep.join(str(x) for x in xs), replace=True)


def _tb_logger(cfg: DictConfig):
    try:
        from pytorch_lightning.loggers import TensorBoardLogger

        return TensorBoardLogger(save_dir=cfg.io.base_output_path, name=cfg.tag, version=cfg.io.version)
    except ModuleNotFoundError:  # tensorboard not installed
        from pytorch_lightning.loggers import CSVLogger

        return CSVLogger(save_dir=cfg.io.base_output_path, name=cfg.tag, version=cfg.io.version)


def train(cfg: DictConfig) -> None:
    log = get_logger("onboard_vlm.run")
    log.info("===> Run '%s' | task=%s | teacher=%s | student=%s | data=%s",
             cfg.experiment_name, cfg.task._target_.rsplit(".", 1)[-1], cfg.teacher.get("model_id", "tiny"),
             cfg.student.name, cfg.data_module.get("_target_", "?").rsplit(".", 1)[-1])
    log.info("     PyTorch %s | Lightning %s | CUDA available: %s (%d devices)", torch.__version__, pl.__version__,
             torch.cuda.is_available(), torch.cuda.device_count())

    with stage(f"Seeding everything with {cfg.seed}", log):
        pl.seed_everything(cfg.seed, workers=True)
        torch.set_float32_matmul_precision("high")

    with stage(f"Building task {cfg.task._target_}", log):
        task = hydra.utils.instantiate(cfg.task, cfg, _recursive_=False)
        task.describe()

    with stage("Building data module", log):
        data_module = hydra.utils.instantiate(cfg.data_module, _recursive_=False)

    with stage("Building callbacks, logger and trainer", log):
        callbacks = [hydra.utils.instantiate(c) for c in cfg.callbacks.values()]
        log.info("     callbacks: %s", ", ".join(type(c).__name__ for c in callbacks))
        tb = _tb_logger(cfg)
        trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True), logger=tb, callbacks=callbacks)
        log.info("     metrics -> %s", getattr(tb, "log_dir", cfg.io.base_output_path))

    if cfg.training:
        log.info("===> Starting fit (datasets are set up and the teacher is loaded at the start of fit)")
        trainer.fit(task, data_module, ckpt_path=cfg.resume_ckpt)

    if cfg.final_validate:
        with stage("Final validation", log):
            trainer.validate(task, data_module, ckpt_path="best" if cfg.training and _has_best(trainer) else None)

    if trainer.is_global_zero:
        with stage(f"Exporting deployable student to {cfg.export_dir}", log):
            task.export(cfg.export_dir)


def _has_best(trainer: pl.Trainer) -> bool:
    cb = getattr(trainer, "checkpoint_callback", None)
    return bool(cb is not None and getattr(cb, "best_model_path", ""))


@hydra.main(config_path="config", config_name="defaults", version_base=None)
def main(cfg: DictConfig) -> None:
    run_dir = HydraConfig.get().runtime.output_dir
    log = setup_logging(run_dir)
    log.info("===> Resolved configuration (also saved under %s/.hydra/)\n%s", run_dir, OmegaConf.to_yaml(cfg, resolve=True))
    train(cfg)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = os.environ.get("HYDRA_FULL_ERROR", "1")
    main()
