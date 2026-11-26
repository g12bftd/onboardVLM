"""Lightning callback that reports training progress in real time through the pipeline logger, and
records the training compute of every run (``train_summary.json``: images and soft tokens seen per
budget, optimizer steps, wall-clock and device-hours), so methods can be compared at equal compute."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
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
        self.totals: dict[str, Any] = {"images": 0, "soft_tokens": 0, "images_by_budget": Counter(), "tokens_by_budget": Counter()}

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
            w = trainer.world_size
            n_tok = sum(getattr(batch, "image_token_lengths", []) or [])
            budget = batch.transforms[0].token_budget if getattr(batch, "transforms", None) else None
            self.totals["images"] += len(batch) * w
            self.totals["soft_tokens"] += n_tok * w
            if budget is not None:
                self.totals["images_by_budget"][budget] += len(batch) * w
                self.totals["tokens_by_budget"][budget] += n_tok * w
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

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        log.info("===> Test (step %d)", trainer.global_step)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        log.info("<=== test: %s", fmt_metrics(trainer.callback_metrics, ["test"]))

    # -- checkpoints / end --------------------------------------------------------------------
    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict[str, Any]) -> None:
        if getattr(self, "_last_ckpt_step", None) == trainer.global_step:  # top-k and last.ckpt share one state
            return
        self._last_ckpt_step = trainer.global_step
        cb = getattr(trainer, "checkpoint_callback", None)
        where = getattr(cb, "dirpath", None) or trainer.default_root_dir
        log.info("     saving checkpoint at step %d -> %s", trainer.global_step, where)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        wall = time.perf_counter() - self._t_fit
        log.info("<=== Training finished in %s (%d steps) | %s", _hms(wall), trainer.global_step, memory_summary())
        if trainer.is_global_zero:
            summary = {
                "optimizer_steps": trainer.global_step, "wall_clock_s": round(wall, 1),
                "accelerator": type(trainer.accelerator).__name__, "devices": trainer.num_devices * trainer.num_nodes,
                "device_hours": round(wall * trainer.num_devices * trainer.num_nodes / 3600, 3),
                "images_seen": self.totals["images"], "soft_tokens_seen": self.totals["soft_tokens"],
                "images_by_budget": dict(self.totals["images_by_budget"]), "tokens_by_budget": dict(self.totals["tokens_by_budget"]),
            }
            cb = getattr(trainer, "checkpoint_callback", None)
            if cb is not None and getattr(cb, "best_model_path", ""):  # the checkpoint model selection chose, and on what
                score = cb.best_model_score
                summary["selection"] = {"metric": cb.monitor, "mode": cb.mode, "best_score": float(score) if score is not None else None,
                                        "best_checkpoint": cb.best_model_path}
            path = os.path.join(trainer.log_dir or trainer.default_root_dir, "train_summary.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(summary, f, indent=1)
            log.info("     compute: %s images, %s soft tokens, %.2f device-hours (%s) -> %s", f"{summary['images_seen']:,}",
                     f"{summary['soft_tokens_seen']:,}", summary["device_hours"], summary["accelerator"], path)

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        log.error("!!!! training stopped by %s: %s", type(exception).__name__, exception)


def _hms(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):  # NaN / inf
        return "?"
    s = int(seconds)
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"



class TrajectoryCallback(pl.Callback):
    """Measurements over training time, for learning curves.

    * Logs ``progress/images_seen`` and ``progress/soft_tokens_seen`` with every logged training step and
      after every validation, so all curves can share a compute axis.
    * Exports the deployable student at fixed fractions of training (``fractions``; 0 = before the first
      step) to ``<dirpath>/step_<step>/`` with ``trajectory.json`` (step, fraction, images and soft tokens
      seen). ``scripts/trajectory_eval.py`` evaluates every export (E1 / E2 / E3) after training.

    Args:
        dirpath: where the exports go (``${output_root}/trajectories/${run_id}``).
        fractions: fractions of the total optimizer steps; log-spaced by default (curves change fastest early).
    """

    def __init__(self, dirpath: str, fractions: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)):
        super().__init__()
        self.dirpath, self.fractions = dirpath, tuple(sorted(set(float(f) for f in fractions)))
        self.images = self.tokens = 0
        self.targets: dict[int, float] = {}

    def _progress(self) -> dict[str, float]:
        return {"progress/images_seen": float(self.images), "progress/soft_tokens_seen": float(self.tokens)}

    def _log(self, trainer: pl.Trainer) -> None:
        for lg in trainer.loggers:
            lg.log_metrics(self._progress(), step=trainer.global_step)

    def _export(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        step = trainer.global_step
        if trainer.is_global_zero:
            d = os.path.join(self.dirpath, f"step_{step:08d}")
            pl_module.export(d)
            with open(os.path.join(d, "trajectory.json"), "w") as f:
                json.dump({"step": step, "fraction": self.targets.get(step, step / max(1, trainer.estimated_stepping_batches)),
                           "images_seen": self.images, "soft_tokens_seen": self.tokens, "epoch": trainer.current_epoch}, f, indent=1)
            log.info("     trajectory: exported the student at step %d (%s images seen) -> %s", step, f"{self.images:,}", d)
        trainer.strategy.barrier()

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        total = int(trainer.estimated_stepping_batches)
        self.targets = {}
        for f in self.fractions:
            self.targets.setdefault(int(round(f * total)), f)
        log.info("     trajectory exports at steps %s of %d -> %s", sorted(self.targets), total, self.dirpath)
        if 0 in self.targets and trainer.global_step == 0:
            self._export(trainer, pl_module)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        w = trainer.world_size
        self.images += len(batch) * w
        self.tokens += sum(getattr(batch, "image_token_lengths", []) or []) * w
        if trainer.global_step % max(1, trainer.log_every_n_steps) == 0:
            self._log(trainer)
        if trainer.global_step in self.targets and trainer.global_step > 0:
            self._export(trainer, pl_module)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.sanity_checking:
            self._log(trainer)


class DiagnosticsCallback(pl.Callback):
    """Optimisation diagnostics on the loggers (``diag/*``), every ``every_n_steps`` optimizer steps:

    * ``grad_norm`` (L2, before clipping) of the student and of the training-only heads, and ``grad_clipped``
      (1 when the norm exceeds ``trainer.gradient_clip_val``): how often clipping changes the update;
    * ``param_norm`` of the student, and ``update_ratio`` = ||theta_t - theta_{t-n}|| / ||theta_{t-n}||: the
      effective step size, comparable across model sizes trained with one learning rate;
    * ``step_time_s``, ``images_per_s``, ``soft_tokens_per_s`` and ``max_mem_gb`` (peak device memory).
    """

    def __init__(self, every_n_steps: int = 50, track_updates: bool = True):
        super().__init__()
        self.every_n_steps, self.track_updates = max(1, int(every_n_steps)), track_updates
        self._snapshot: list | None = None
        self._t = 0.0
        self._images = self._tokens = 0
        self._pending: dict[str, float] = {}

    @staticmethod
    def _norm(params) -> float:
        import torch

        sq = [p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
        return float(torch.stack(sq).sum().sqrt()) if sq else 0.0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._t = time.perf_counter()

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: Any) -> None:
        step = trainer.global_step + 1  # the step this update produces
        if step % self.every_n_steps:
            return
        student = getattr(pl_module, "student", None)
        s_params = [p for p in (student.parameters() if student is not None else []) if p.requires_grad]
        ids = {id(p) for p in s_params}
        other = [p for p in pl_module.parameters() if p.requires_grad and id(p) not in ids]
        g = self._norm(s_params + other)
        clip = trainer.gradient_clip_val
        self._pending = {"diag/grad_norm": g, "diag/grad_norm_student": self._norm(s_params), "diag/grad_norm_heads": self._norm(other)}
        if clip:
            self._pending["diag/grad_clipped"] = float(g > float(clip))

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        import torch

        w = trainer.world_size
        self._images += len(batch) * w
        self._tokens += sum(getattr(batch, "image_token_lengths", []) or []) * w
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps:
            return
        now = time.perf_counter()
        dt = max(now - self._t, 1e-9)
        out = {**self._pending, "diag/step_time_s": dt / self.every_n_steps, "diag/images_per_s": self._images / dt,
               "diag/soft_tokens_per_s": self._tokens / dt}
        if torch.cuda.is_available():
            out["diag/max_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        student = getattr(pl_module, "student", None)
        if student is not None:
            with torch.no_grad():
                params = [p.detach() for p in student.parameters() if p.requires_grad]
                pn = torch.stack([p.float().pow(2).sum() for p in params]).sum().sqrt() if params else torch.zeros(())
                out["diag/param_norm"] = float(pn)
                if self.track_updates:
                    if self._snapshot is not None and len(self._snapshot) == len(params):
                        du = torch.stack([(p.float() - q).pow(2).sum() for p, q in zip(params, self._snapshot)]).sum().sqrt()
                        dn = torch.stack([q.pow(2).sum() for q in self._snapshot]).sum().sqrt()
                        out["diag/update_ratio"] = float(du / dn.clamp_min(1e-12))
                    self._snapshot = [p.float().clone() for p in params]
        if trainer.is_global_zero:
            for lg in trainer.loggers:
                lg.log_metrics(out, step=step)
        self._t, self._images, self._tokens, self._pending = now, 0, 0, {}
