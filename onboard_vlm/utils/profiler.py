"""Vision-encoder profiler: parameters, MACs/FLOPs and peak activation memory at Gemma token budgets.

* FLOPs come from ``torch.utils.flop_counter.FlopCounterMode`` (matmul / conv / SDPA); MACs = FLOPs / 2.
* Peak activation memory is measured with ``torch.cuda.max_memory_allocated`` on GPU, and on CPU
  with a ``TorchDispatchMode`` that tracks the bytes of live tensor storages created during the
  forward pass (parameters and inputs excluded). Inference mode, batch size 1.
"""

from __future__ import annotations

import functools
import time
import weakref
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.utils.flop_counter import FlopCounterMode

from onboard_vlm.geometry import build_teacher_patches, pad_dense, target_size
from onboard_vlm.interfaces import POOLING_KERNEL


@dataclass
class ProfileRow:
    model: str
    token_budget: int
    image_hw: tuple[int, int]
    num_tokens: int
    params_m: float
    gmacs: float
    gflops: float
    peak_activation_mb: float
    latency_ms: float | None = None


class _LiveTensorMemory(TorchDispatchMode):
    """Tracks peak bytes of storages allocated by ops while active (approximate, device-agnostic)."""

    def __init__(self) -> None:
        super().__init__()
        self.refs: dict[int, int] = {}
        self.sizes: dict[int, int] = {}
        self.current = 0
        self.peak = 0

    def _release(self, key: int) -> None:
        self.refs[key] -= 1
        if self.refs[key] == 0:
            self.current -= self.sizes.pop(key)
            del self.refs[key]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_leaves(out):
            if isinstance(t, torch.Tensor) and not isinstance(t, nn.Parameter):
                try:
                    st = t.untyped_storage()
                except (RuntimeError, NotImplementedError):
                    continue
                key = st.data_ptr()
                if key == 0:
                    continue
                if key not in self.refs:
                    self.refs[key] = 0
                    self.sizes[key] = st.nbytes()
                    self.current += st.nbytes()
                    self.peak = max(self.peak, self.current)
                self.refs[key] += 1
                weakref.finalize(t, self._release, key)
        return out


def _student_inputs(hw: tuple[int, int], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    x, m, _ = pad_dense([torch.rand(3, *hw, device=device, dtype=dtype)])
    return x, m


def _gemma_inputs(hw: tuple[int, int], budget: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    pv, pos = build_teacher_patches([torch.rand(3, *hw, device=device, dtype=dtype)], max_patches=budget * POOLING_KERNEL**2)
    return pv, pos


@torch.inference_mode()
def profile_encoder(
    model: nn.Module,
    name: str,
    kind: str = "student",
    token_budgets: Sequence[int] = (140, 280, 560),
    source_hw: tuple[int, int] = (1080, 1440),
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    latency_iters: int = 0,
) -> list[ProfileRow]:
    """Profile a student (``kind="student"``, dense input) or a ``Gemma4VisionModel`` (``kind="gemma"``).

    ``source_hw`` is the camera frame; each budget resizes it exactly like ``Gemma4ImageProcessor``.
    """
    device = torch.device(device)
    model = model.to(device, dtype).eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    rows = []
    for budget in token_budgets:
        hw = target_size(*source_hw, budget)
        if kind == "student":
            run = functools.partial(model, *_student_inputs(hw, device, dtype))
        elif kind == "gemma":
            pv, pos = _gemma_inputs(hw, budget, device, dtype)
            run = functools.partial(model, pixel_values=pv, pixel_position_ids=pos)
        else:
            raise ValueError(kind)
        run()  # warm-up (lazy init, cudnn autotune)
        fc = FlopCounterMode(display=False)
        with fc:
            run()
        flops = fc.get_total_flops()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            base = torch.cuda.memory_allocated(device)
            run()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device) - base
        else:
            mem = _LiveTensorMemory()
            with mem:
                run()
            peak = mem.peak
        latency = None
        if latency_iters:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(latency_iters):
                run()
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency = (time.perf_counter() - t0) / latency_iters * 1e3
        rows.append(ProfileRow(name, budget, hw, (hw[0] // 48) * (hw[1] // 48), round(params_m, 3), round(flops / 2e9, 3), round(flops / 1e9, 3), round(peak / 2**20, 2), latency))
    return rows


def format_table(rows: Sequence[ProfileRow]) -> str:
    cols = ["model", "token_budget", "image_hw", "num_tokens", "params_m", "gmacs", "gflops", "peak_activation_mb", "latency_ms"]
    data = [[str(asdict(r)[c]) for c in cols] for r in rows]
    widths = [max(len(c), *(len(d[i]) for d in data)) for i, c in enumerate(cols)]
    line = lambda vals: "| " + " | ".join(v.ljust(w) for v, w in zip(vals, widths)) + " |"  # noqa: E731
    return "\n".join([line(cols), "|" + "|".join("-" * (w + 2) for w in widths) + "|", *map(line, data)])


def rows_to_dicts(rows: Sequence[ProfileRow]) -> list[dict[str, Any]]:
    return [asdict(r) for r in rows]
