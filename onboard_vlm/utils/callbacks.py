"""Lightning callback that reports training progress in real time through the pipeline logger."""

from __future__ import annotations

import time
from typing import Any

import pytorch_lightning as pl

from onboard_vlm.utils.logging import describe_module, fmt_metrics, get_logger, memory_summary

log = get_logger("onboard_vlm.train")


class PipelineLoggingCallback(pl.Callback):
    """Logs fit setup, every ``every_n_steps`` training steps (losses, lr, throughput, ETA, memory),
    epoch boundaries, validation results, checkpoint saves and the end of training.

    Args:
        every_n_steps: training-step logging interval.
    """

    def __init__(self, every_n_steps: int = 50):
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self._t_fit = self._t_window = 0.0
        self._images = 0

    # -- fit ------------------------------------------------------------------------------
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._t_fit = self._t_window = time.perf_counter()
        log.info("===> Training: %s", type(pl_module).__name__)
        student = getattr(pl_module, "student", None)
        if student is not None:
            log.info("     student  %s", describe_module(student))
        log.info(
            "     devices=%s x %s nodes | strategy=%s | precision=%s | max_epochs=%s | max_steps=%s | est. optimizer steps=%s",
            trainer.num_devices, trainer.num_nodes, type(trainer.strategy).__name__, trainer.precision,
            trainer.max_epochs, trainer.max_steps, getattr(trainer, "estimated_stepping_batches", "?"),
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        n = trainer.num_training_batches
        log.info("---- epoch %d started (%s batches/epoch)", trainer.current_epoch, n)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        try:
            self._images += len(batch)
        except TypeError:
            pass
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps:
            return
        dt = time.perf_counter() - self._t_window
        rate = self._images / dt if dt > 0 else 0.0
        total = trainer.estimated_stepping_batches
        elapsed = time.perf_counter() - self._t_fit
        eta = elapsed / step * (total - step) if step and isinstance(total, (int, float)) and total != float("inf") else float("nan")
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else float("nan")
        log.info(
            "step %d/%s | epoch %d | %s | lr %.3g | %.1f img/s | elapsed %s | ETA %s | %s",
            step, total, trainer.current_epoch, fmt_metrics(trainer.callback_metrics, ["train/"]), lr, rate,
            _hms(elapsed), _hms(eta), memory_summary(),
        )
        self._t_window, self._images = time.perf_counter(), 0

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        log.info("---- epoch %d finished | %s", trainer.current_epoch, fmt_metrics(trainer.callback_metrics, ["train/"]))

    # -- validation -------------------------------------------------------------------------
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.sanity_checking:
            log.info("===> Validation (epoch %d, step %d)", trainer.current_epoch, trainer.global_step)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        tag = "sanity check" if trainer.sanity_checking else "validation"
        log.info("<=== %s: %s", tag, fmt_metrics(trainer.callback_metrics, ["val"]))

    # -- checkpoints / end --------------------------------------------------------------------
    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict[str, Any]) -> None:
        if getattr(self, "_last_ckpt_step", None) == trainer.global_step:  # top-k and last.ckpt share one state
            return
        self._last_ckpt_step = trainer.global_step
        cb = getattr(trainer, "checkpoint_callback", None)
        where = getattr(cb, "dirpath", None) or trainer.default_root_dir
        log.info("     saving checkpoint at step %d -> %s", trainer.global_step, where)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        log.info("<=== Training finished in %s (%d steps) | %s", _hms(time.perf_counter() - self._t_fit), trainer.global_step, memory_summary())

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        log.error("!!!! training stopped by %s: %s", type(exception).__name__, exception)


def _hms(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):  # NaN / inf
        return "?"
    s = int(seconds)
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"

