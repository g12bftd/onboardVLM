"""Distillation objectives (composable as a weighted sum from YAML).

All vision losses take ``(student, teacher, which=None)`` where both are packed
``[N, D]`` outputs in identical token order (student already passed through the
:class:`DistillationAdapter`, so its intermediate tokens are keyed by *teacher* layer).

Why ``normalize="rms"`` is the default for feature losses: ``Gemma4MultimodalEmbedder`` applies
a scale-free RMSNorm to every vision token before projecting it into the language model, so
the LM never sees per-token magnitude. Matching RMS-normalised tokens therefore targets exactly
what the VLM consumes, and avoids the large ``sqrt(D)``-scaled norms of raw Gemma tokens
dominating an MSE.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from onboard_vlm.interfaces import StudentVisionOutput, TeacherVisionOutput
from onboard_vlm.registry import build_loss, register_loss


def normalize_tokens(x: torch.Tensor, mode: str, eps: float = 1e-6) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "rms":
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if mode == "layernorm":
        return F.layer_norm(x, x.shape[-1:], eps=eps)
    if mode == "l2":
        return F.normalize(x, dim=-1, eps=eps)
    raise ValueError(f"unknown normalize mode {mode!r}")


def _select(out: StudentVisionOutput | TeacherVisionOutput, which: int | None) -> torch.Tensor:
    if which is None:
        return out.tokens
    if which not in out.intermediate_tokens:
        raise KeyError(f"layer {which} missing; available {sorted(out.intermediate_tokens)}")
    return out.intermediate_tokens[which]


def _segment_mean(values: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    """Mean of per-token values within each image -> ``[B]``."""
    idx = torch.repeat_interleave(torch.arange(len(lengths), device=values.device), torch.as_tensor(lengths, device=values.device))
    sums = values.new_zeros(len(lengths)).index_add_(0, idx, values)
    return sums / torch.as_tensor(lengths, device=values.device, dtype=values.dtype)


def _check_aligned(s: StudentVisionOutput, t: TeacherVisionOutput) -> None:
    if list(s.image_token_lengths) != list(t.image_token_lengths):
        raise ValueError(f"student/teacher token counts differ: {s.image_token_lengths} vs {t.image_token_lengths}")


@register_loss("token_feature")
class TokenFeatureLoss(nn.Module):
    """Per-token feature matching (Beyer et al., CVPR 2022 "patient & consistent" KD on tokens).

    Args:
        metric: ``"mse"``, ``"smooth_l1"`` or ``"cosine"`` (``1 - cos``).
        normalize: applied to both sides first: ``"rms"`` (default), ``"layernorm"``, ``"l2"``, ``"none"``.
        reduction: ``"token"`` (every token weighs the same) or ``"image"`` (every image weighs the same).
    """

    def __init__(self, metric: str = "mse", normalize: str = "rms", beta: float = 1.0, reduction: str = "token"):
        super().__init__()
        if metric not in ("mse", "smooth_l1", "cosine"):
            raise ValueError(f"unknown metric {metric!r}")
        self.metric, self.normalize, self.beta, self.reduction = metric, normalize, beta, reduction

    def per_token(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s, t = s.float(), t.detach().float()
        if self.metric == "cosine":
            return 1.0 - F.cosine_similarity(s, t, dim=-1)
        s, t = normalize_tokens(s, self.normalize), normalize_tokens(t, self.normalize)
        if self.metric == "mse":
            return (s - t).pow(2).mean(-1)
        return F.smooth_l1_loss(s, t, beta=self.beta, reduction="none").mean(-1)

    def forward(self, student: StudentVisionOutput, teacher: TeacherVisionOutput, which: int | None = None) -> torch.Tensor:
        _check_aligned(student, teacher)
        v = self.per_token(_select(student, which), _select(teacher, which))
        return v.mean() if self.reduction == "token" else _segment_mean(v, student.image_token_lengths).mean()


@register_loss("relational_token")
class RelationalTokenLoss(nn.Module):
    """Per-image token-relation matching (Hao et al., NeurIPS 2022, intra-image manifold term).

    For every image separately, builds ``S = sim(X, X)`` over its own ``n_i`` tokens and matches
    student to teacher. Tokens of different images are never compared (no global ``X X^T`` over the
    packed batch), so the loss is invariant to batch composition and order.

    Args:
        similarity: ``"cosine"`` or ``"dot"`` (scaled by ``1/sqrt(D)``).
        match: ``"mse"`` on similarity matrices or ``"kl"`` between row-wise softmax(S / temperature).
        temperature: for ``"kl"``.
    """

    def __init__(self, similarity: str = "cosine", match: str = "mse", temperature: float = 0.1):
        super().__init__()
        if similarity not in ("cosine", "dot") or match not in ("mse", "kl"):
            raise ValueError(f"bad similarity/match: {similarity}/{match}")
        self.similarity, self.match, self.temperature = similarity, match, temperature

    def _sim(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.similarity == "cosine":
            x = F.normalize(x, dim=-1)
            return x @ x.T
        return (x @ x.T) / x.shape[-1] ** 0.5

    def per_image(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ss, st = self._sim(s), self._sim(t.detach())
        if self.match == "mse":
            return (ss - st).pow(2).mean()
        log_p_s = F.log_softmax(ss / self.temperature, dim=-1)
        log_p_t = F.log_softmax(st / self.temperature, dim=-1)
        return F.kl_div(log_p_s, log_p_t, log_target=True, reduction="none").sum(-1).mean()

    def forward(self, student: StudentVisionOutput, teacher: TeacherVisionOutput, which: int | None = None) -> torch.Tensor:
        _check_aligned(student, teacher)
        lens = student.image_token_lengths
        s_parts = torch.split(_select(student, which), lens)
        t_parts = torch.split(_select(teacher, which), lens)
        return torch.stack([self.per_image(s, t) for s, t in zip(s_parts, t_parts)]).mean()


@register_loss("intermediate_layer")
class IntermediateLayerLoss(nn.Module):
    """Layer-mapped hidden-token matching (``layer_map: {student_idx: teacher_idx}`` lives in the adapter).

    Args:
        layers: teacher layer indices to match (default: every layer present in the adapted student output).
        mode: ``"feature"`` (:class:`TokenFeatureLoss`) or ``"relational"`` (:class:`RelationalTokenLoss`).
        **inner: kwargs of the inner loss.
    """

    def __init__(self, layers: Sequence[int] | None = None, mode: str = "feature", **inner: Any):
        super().__init__()
        self.layers = list(layers) if layers is not None else None
        if mode == "feature":
            self.inner: nn.Module = TokenFeatureLoss(**inner)
        elif mode == "relational":
            self.inner = RelationalTokenLoss(**inner)
        else:
            raise ValueError(f"unknown mode {mode!r}")

    def forward(self, student: StudentVisionOutput, teacher: TeacherVisionOutput, which: int | None = None) -> torch.Tensor:
        layers = self.layers if self.layers is not None else sorted(student.intermediate_tokens)
        if not layers:
            raise ValueError("IntermediateLayerLoss has no layers to match (empty layer_map?)")
        return torch.stack([self.inner(student, teacher, which=i) for i in layers]).mean()


@register_loss("grafted_logit_kd")
class GraftedLogitKDLoss(nn.Module):
    """Stage-2 objective: temperature-scaled KL(teacher || grafted student) on next-token logits.

    The student encoder is grafted into the frozen VLM (``vlm.grafted.GraftedVLM``); only
    ``embed_vision`` (or a LoRA on it, optionally the student too) is trainable. Logits are compared
    only on positions selected by ``mask`` (the answer span), restricted with ``logits_to_keep`` so
    the ``[B, L, 262144]`` vocabulary tensor is never materialised for the whole sequence.

    Args:
        temperature: softmax temperature ``T``; the loss is multiplied by ``T**2`` (Hinton et al.).
        ce_weight: optional cross-entropy on the reference answer tokens (``targets``).
    """

    def __init__(self, temperature: float = 1.0, ce_weight: float = 0.0):
        super().__init__()
        self.temperature, self.ce_weight = temperature, ce_weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``student_logits``/``teacher_logits`` ``[B, K, V]``; ``mask`` ``[B, K]`` bool; ``targets`` ``[B, K]``."""
        t = self.temperature
        s = student_logits[mask].float() / t
        q = teacher_logits[mask].float().detach() / t
        kl = F.kl_div(F.log_softmax(s, -1), F.log_softmax(q, -1), log_target=True, reduction="batchmean") * t * t
        logs = {"kd_kl": kl.detach()}
        loss = kl
        if self.ce_weight > 0 and targets is not None:
            ce = F.cross_entropy(student_logits[mask].float(), targets[mask])
            logs["kd_ce"] = ce.detach()
            loss = loss + self.ce_weight * ce
        return loss, logs


SPACES = ("vision", "lm")


class CompositeLoss(nn.Module):
    """Weighted sum of vision losses built from ``[{name, weight, label?, space?, **kwargs}, ...]``.

    ``space: vision`` (default) compares the 768-d vision-tower tokens. ``space: lm`` first maps
    student and teacher final tokens through the teacher checkpoint's *frozen* ``embed_vision``
    (scale-free RMSNorm + projection to 1536 / 2560): the exact space the language model reads.
    Only the student is trained either way.
    """

    def __init__(self, terms: Sequence[Mapping[str, Any]]):
        super().__init__()
        self.labels: list[str] = []
        self.weights: list[float] = []
        self.spaces: list[str] = []
        losses = []
        for i, term in enumerate(terms):
            term = dict(term)
            weight = float(term.pop("weight", 1.0))
            label = term.pop("label", None) or f"{term['name']}_{i}"
            space = term.pop("space", "vision")
            if space not in SPACES:
                raise ValueError(f"space must be one of {SPACES}, got {space!r}")
            if space == "lm" and term["name"] == "intermediate_layer":
                raise ValueError("space: lm applies to final tokens only (embed_vision sees the last layer)")
            losses.append(build_loss(term))
            self.labels.append(label)
            self.weights.append(weight)
            self.spaces.append(space)
        self.losses = nn.ModuleList(losses)

    @property
    def uses_lm_space(self) -> bool:
        return "lm" in self.spaces

    def forward(
        self,
        student: StudentVisionOutput,
        teacher: TeacherVisionOutput,
        lm_projector: Any = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``lm_projector``: ``[N, 768] -> [N, D_lm]`` (frozen ``embed_vision``), needed for ``space: lm`` terms."""
        total = student.tokens.new_zeros((), dtype=torch.float32)
        logs: dict[str, torch.Tensor] = {}
        lm_pair: tuple[StudentVisionOutput, TeacherVisionOutput] | None = None
        for label, w, loss, space in zip(self.labels, self.weights, self.losses, self.spaces):
            s, t = student, teacher
            if space == "lm":
                if lm_projector is None:
                    raise RuntimeError("space: lm loss terms need the teacher's embed_vision projector")
                if lm_pair is None:
                    with torch.no_grad():
                        t_lm = lm_projector(teacher.tokens)
                    lm_pair = (
                        StudentVisionOutput(lm_projector(student.tokens), student.image_token_lengths),
                        TeacherVisionOutput(t_lm, teacher.image_token_lengths),
                    )
                s, t = lm_pair
            v = loss(s, t)
            logs[label] = v.detach()
            total = total + w * v
        logs["total"] = total.detach()
        return total, logs
