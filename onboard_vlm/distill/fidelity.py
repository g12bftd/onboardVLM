"""Student-teacher fidelity metrics: one definition shared by every task (val / test logging), the
E1 fidelity evaluation (``run_fidelity.py``) and the paper tables.

Gemma 4's ``embed_vision`` is ``W @ RMSNorm(z)`` with a *scale-free* RMSNorm, so the language model
never sees the per-token scale of the vision output ``z``. Every metric here is therefore invariant
to a positive rescaling of any token, and the primary ones are measured in the language model's
input space ``E(z) = embed_vision(z)``:

* ``fid_lm_cos``     mean per-token cosine of ``E(z_s)`` and ``E(z_t)`` (primary);
* ``fid_lm_rel_err`` mean per-token ``||E(z_s) - E(z_t)|| / ||E(z_t)||`` (primary);
* ``fid_vis_cos``    mean per-token cosine in the 768-d vision space;
* ``fid_vis_cos_p05`` 5th percentile of the per-token vision cosine (worst tokens);
* ``fid_cka``        per-image linear CKA of the RMS-normalised token sets (spatial structure),
                     averaged over images.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn.functional as F

FIDELITY_KEYS = ("fid_lm_cos", "fid_lm_rel_err", "fid_vis_cos", "fid_vis_cos_p05", "fid_cka")


def _rms_normalise(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Linear CKA between two token sets of one image (``[n, d1]``, ``[n, d2]``), Gram form."""
    if x.shape[0] < 2:
        return x.new_tensor(1.0)
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    kx, ky = x @ x.T, y @ y.T
    return (kx * ky).sum() / (kx.norm() * ky.norm()).clamp_min(1e-12)


@torch.no_grad()
def per_token_fidelity(
    student: torch.Tensor, teacher: torch.Tensor, project: Callable[[torch.Tensor], torch.Tensor] | None = None
) -> dict[str, torch.Tensor]:
    """Per-token cosines (and LM-space relative errors) for packed ``[N, D]`` tokens."""
    s, t = student.float(), teacher.float()
    out = {"vis_cos": F.cosine_similarity(s, t, dim=-1)}
    if project is not None:
        es, et = project(student).float(), project(teacher).float()
        out["lm_cos"] = F.cosine_similarity(es, et, dim=-1)
        out["lm_rel_err"] = (es - et).norm(dim=-1) / et.norm(dim=-1).clamp_min(1e-12)
    return out


@torch.no_grad()
def per_image_fidelity(
    student: torch.Tensor,
    teacher: torch.Tensor,
    lengths: Sequence[int],
    project: Callable[[torch.Tensor], torch.Tensor] | None = None,
    cka: bool = True,
) -> list[dict[str, float]]:
    """One record per image: mean / min token cosines, LM-space error and CKA (E1 per-image outputs)."""
    tok = per_token_fidelity(student, teacher, project)
    recs = []
    for s_i, t_i, sl in zip(student.split(list(lengths)), teacher.split(list(lengths)), _slices(lengths)):
        r = {"num_tokens": int(sl.stop - sl.start), "vis_cos": float(tok["vis_cos"][sl].mean()), "vis_cos_min": float(tok["vis_cos"][sl].min())}
        if "lm_cos" in tok:
            r["lm_cos"] = float(tok["lm_cos"][sl].mean())
            r["lm_rel_err"] = float(tok["lm_rel_err"][sl].mean())
        if cka:
            r["cka"] = float(linear_cka(_rms_normalise(s_i.float()), _rms_normalise(t_i.float())))
        recs.append(r)
    return recs


@torch.no_grad()
def fidelity_metrics(
    student: torch.Tensor,
    teacher: torch.Tensor,
    lengths: Sequence[int],
    project: Callable[[torch.Tensor], torch.Tensor] | None = None,
    cka: bool = True,
) -> dict[str, torch.Tensor]:
    """Batch-level ``fid_*`` metrics (token-weighted means; CKA averaged over images)."""
    tok = per_token_fidelity(student, teacher, project)
    out = {"fid_vis_cos": tok["vis_cos"].mean(), "fid_vis_cos_p05": torch.quantile(tok["vis_cos"], 0.05)}
    if "lm_cos" in tok:
        out["fid_lm_cos"] = tok["lm_cos"].mean()
        out["fid_lm_rel_err"] = tok["lm_rel_err"].mean()
    if cka:
        vals = [linear_cka(_rms_normalise(s.float()), _rms_normalise(t.float())) for s, t in zip(student.split(list(lengths)), teacher.split(list(lengths)))]
        out["fid_cka"] = torch.stack(vals).mean()
    return out


def _slices(lengths: Sequence[int]) -> list[slice]:
    out, start = [], 0
    for n in lengths:
        out.append(slice(start, start + int(n)))
        start += int(n)
    return out


def student_tokens(student: torch.nn.Module, batch) -> torch.Tensor:
    """The student's packed output tokens for a ``DistillBatch`` (Gemma-architecture students consume the
    teacher's patch sequence; every other student the dense, shape-exact image batch)."""
    if getattr(student, "accepts_teacher_patches", False):
        ti = batch.teacher_inputs
        return student(pixel_values=ti.pixel_values, pixel_position_ids=ti.pixel_position_ids).tokens
    return student(batch.student_images, batch.valid_token_mask).tokens
