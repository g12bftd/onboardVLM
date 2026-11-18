"""Stage-1 / Stage-2 distillation entry point (Hydra + PyTorch Lightning, as in TimeFM).

    python -u run_distill.py +experiment=stage1/compact_vit_s_e2b
    python -u run_distill.py +experiment=stage2/compact_vit_s_e2b
    python -u run_distill.py +experiment=debug/dryrun_stage1          # offline sanity check
    python -u run_distill.py +experiment=stage1/compact_vit_s_e2b ++batch_size=64 ++trainer.max_epochs=50
"""

from __future__ import annotations

import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _logger(cfg: DictConfig):
    try:
        from pytorch_lightning.loggers import TensorBoardLogger

        return TensorBoardLogger(save_dir=cfg.io.base_output_path, name=cfg.tag, version=cfg.io.version)
    except ModuleNotFoundError:  # tensorboard not installed
        from pytorch_lightning.loggers import CSVLogger

        return CSVLogger(save_dir=cfg.io.base_output_path, name=cfg.tag, version=cfg.io.version)


@hydra.main(config_path="config", config_name="defaults", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    log.info("config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    task = hydra.utils.instantiate(cfg.task, _recursive_=False)
    data_module = hydra.utils.instantiate(cfg.data_module, _recursive_=False)
    callbacks = [hydra.utils.instantiate(c) for c in cfg.callbacks.values()]
    trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True), logger=_logger(cfg), callbacks=callbacks)
    trainer.fit(task, data_module, ckpt_path=cfg.resume_ckpt)

    if trainer.is_global_zero:
        if hasattr(task, "export_student"):
            task.export_student(cfg.export_dir)
        elif hasattr(task, "export"):
            task.export(cfg.export_dir)
        log.info("exported deployable student to %s", cfg.export_dir)


if __name__ == "__main__":
    main()
