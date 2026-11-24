"""Pipeline logging: every stage of a run is announced in real time (stdout + ``<run_dir>/run.log``).

TimeFM-style ``===>`` stage headers, extended with timing, memory and rank information:

    [2026-11-24 10:02:11][rank0][onboard_vlm.teachers] ===> Loading teacher google/gemma-4-E2B-it@4adf157
    [2026-11-24 10:02:51][rank0][onboard_vlm.teachers]      vision tower: 167.4M params (bf16) | embed_vision 768->1536
    [2026-11-24 10:02:51][rank0][onboard_vlm.teachers] <=== Loading teacher ... done in 40.1s | RSS 11.8 GB | GPU 0.0 GB

Only rank 0 logs at INFO under DDP (other ranks log warnings and errors), so multi-GPU output stays readable.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import Any, Iterator

import torch
from torch import nn

ROOT_LOGGER = "onboard_vlm"
_FMT = "[%(asctime)s][rank%(rank)s][%(name)s] %(message)s"


def _rank() -> int:
    for k in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        if k in os.environ:
            return int(os.environ[k])
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class _RankFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = _rank()
        return True


def setup_logging(log_dir: str | None = None, level: int = logging.INFO, all_ranks: bool = False) -> logging.Logger:
    """Configure the ``onboard_vlm`` logger (idempotent). Call once at the start of an entry point."""
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level if (_rank() == 0 or all_ranks) else logging.WARNING)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    stream = logging.StreamHandler(sys.stdout)
    handlers: list[logging.Handler] = [stream]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_dir, f"run_rank{_rank()}.log")))
    for h in handlers:
        h.setFormatter(logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
        h.addFilter(_RankFilter())
        logger.addHandler(h)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Child of the ``onboard_vlm`` logger (use ``__name__`` of an ``onboard_vlm.*`` module)."""
    name = name if name.startswith(ROOT_LOGGER) else f"{ROOT_LOGGER}.{name}"
    logger = logging.getLogger(name)
    if not logging.getLogger(ROOT_LOGGER).handlers:  # library use without setup: still print
        setup_logging()
    return logger


def memory_summary() -> str:
    parts = []
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # Linux: KiB -> GiB
        parts.append(f"peak RSS {rss:.1f} GB")
    except Exception:  # pragma: no cover - non-Linux
        pass
    if torch.cuda.is_available():
        parts.append(f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GB (peak {torch.cuda.max_memory_allocated() / 2**30:.1f})")
    return " | ".join(parts)


@contextlib.contextmanager
def stage(name: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Announce a pipeline stage, then report its duration and memory (or the failure)."""
    logger = logger or get_logger("pipeline")
    logger.info("===> %s", name)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("<=== %s FAILED after %.1fs", name, time.perf_counter() - t0)
        raise
    logger.info("<=== %s done in %.1fs | %s", name, time.perf_counter() - t0, memory_summary())


def count_params(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def describe_module(module: nn.Module) -> str:
    total, trainable = count_params(module)
    p = next(module.parameters(), None)
    where = f"{p.dtype}, {p.device}".replace("torch.", "") if p is not None else "no params"
    return f"{type(module).__name__}: {total / 1e6:.2f}M params ({trainable / 1e6:.2f}M trainable; {where})"


def fmt_metrics(metrics: dict[str, Any], keys: list[str] | None = None) -> str:
    out = []
    for k, v in sorted(metrics.items()):
        if keys is not None and not any(s in k for s in keys):
            continue
        try:
            out.append(f"{k}={float(v):.4g}")
        except (TypeError, ValueError):
            continue
    return " ".join(out)
