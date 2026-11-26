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


def _select(out: StudentVisionOutput | TeacherVisionOutput, which: int | None, raw: bool = False) -> torch.Tensor:
    """Final (``which=None``) or layer tokens; ``raw``: the student's features before any projection head."""
    if raw and isinstance(out, StudentVisionOutput):
        if which is None and out.raw_tokens is not None:
            return out.raw_tokens
        if which is not None and which in out.raw_intermediate_tokens:
            return out.raw_intermediate_tokens[which]
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


@register_loss("manifold")
class ManifoldLoss(nn.Module):
    """Fine-grained manifold distillation (Hao et al., NeurIPS 2022): match the *relations between tokens*.

    Tokens are L2-normalised per token (so student and teacher widths may differ: no projection head), and
    relation matrices ``M = F F^T`` of student and teacher are matched with a squared error (mean over entries):

    * ``intra``: per image, the ``N x N`` relations of its own tokens (spatial structure of one image);
    * ``inter``: per token position, the ``B x B`` relations across the images of the batch (needs every image
      to have the same token count, which shape-exact batches guarantee; skipped otherwise);
    * ``random``: ``num_random`` tokens sampled from the whole batch, ``K x K`` relations (mixes both).

    The paper's loss is ``alpha * intra + beta * inter + gamma * random``. Give the weights here, or configure
    one ``manifold`` entry per term (``intra: 1, inter: 0, random: 0`` ...) so each term has its own curve.
    ``reduction``: ``mean`` over the matrix entries (size-free) or ``sum`` = the squared Frobenius norm of each
    relation-matrix difference (averaged over images / positions), which grows with the token count.

    A term that is undefined for a batch (``inter`` with images of different token counts) returns ``None``:
    :class:`CompositeLoss` leaves it out of evaluation means (and logs 0 in training, where every rank logs every key).
    """

    def __init__(self, intra: float = 1.0, inter: float = 1.0, random: float = 1.0, num_random: int = 256, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be mean | sum, got {reduction!r}")
        self.w = {"intra": float(intra), "inter": float(inter), "random": float(random)}
        self.num_random = int(num_random)
        self.reduction = reduction

    def _rel_mse(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d = (a @ a.transpose(-1, -2) - b @ b.transpose(-1, -2)).pow(2)
        return d.mean() if self.reduction == "mean" else d.sum(dim=(-2, -1)).mean()

    def terms(self, s: torch.Tensor, t: torch.Tensor, lengths: Sequence[int]) -> dict[str, torch.Tensor]:
        s, t = F.normalize(s.float(), dim=-1), F.normalize(t.detach().float(), dim=-1)
        out: dict[str, torch.Tensor] = {}
        if self.w["intra"]:
            out["intra"] = torch.stack([self._rel_mse(a, b) for a, b in zip(torch.split(s, list(lengths)), torch.split(t, list(lengths)))]).mean()
        if self.w["inter"] and len(lengths) > 1 and len(set(lengths)) == 1:
            n = int(lengths[0])
            out["inter"] = self._rel_mse(s.view(len(lengths), n, -1).transpose(0, 1), t.view(len(lengths), n, -1).transpose(0, 1))
        if self.w["random"] and s.shape[0] > 1:
            idx = torch.randperm(s.shape[0], device=s.device)[: self.num_random]
            out["random"] = self._rel_mse(s[idx], t[idx])
        return out

    def forward(self, student: StudentVisionOutput, teacher: TeacherVisionOutput, which: int | None = None) -> torch.Tensor | None:
        _check_aligned(student, teacher)
        s = _select(student, which, raw=True)  # relations need no projection: the student's own features
        terms = self.terms(s, _select(teacher, which), student.image_token_lengths)
        if not terms:
            return None  # undefined for this batch (e.g. inter-image relations of images with different token counts)
        return sum(self.w[k] * v for k, v in terms.items())


@register_loss("intermediate_layer")
class IntermediateLayerLoss(nn.Module):
    """Layer-mapped hidden-token matching (``layer_map: {student_idx: teacher_idx}`` lives in the adapter).

    Args:
        layers: teacher layer indices to match (default: every layer present in the adapted student output).
        mode: ``"feature"`` (:class:`TokenFeatureLoss`, FitNets-style), ``"relational"`` (:class:`RelationalTokenLoss`)
            or ``"manifold"`` (:class:`ManifoldLoss`, Hao et al. 2022 applied to hidden layers as in their paper).
        **inner: kwargs of the inner loss.
    """

    def __init__(self, layers: Sequence[int] | None = None, mode: str = "feature", **inner: Any):
        super().__init__()
        self.layers = list(layers) if layers is not None else None
        if mode == "feature":
            self.inner: nn.Module = TokenFeatureLoss(**inner)
        elif mode == "relational":
            self.inner = RelationalTokenLoss(**inner)
        elif mode == "manifold":
            self.inner = ManifoldLoss(**inner)
        else:
            raise ValueError(f"unknown mode {mode!r}")

    def forward(self, student: StudentVisionOutput, teacher: TeacherVisionOutput, which: int | None = None) -> torch.Tensor | None:
        layers = self.layers if self.layers is not None else sorted(student.intermediate_tokens)
        if not layers:
            raise ValueError("IntermediateLayerLoss has no layers to match (empty layer_map?)")
        vals = [v for i in layers if (v := self.inner(student, teacher, which=i)) is not None]
        return torch.stack(vals).mean() if vals else None


@register_loss("grafted_logit_kd")
class GraftedLogitKDLoss(nn.Module):
    """Logit-distillation objective: temperature-scaled KL(teacher || grafted student) on next-token logits.

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


class CompositeLoss(nn.Module):
    """Weighted sum of token losses built from ``[{name, weight, label?, **kwargs}, ...]``.

    Which token space the terms compare in (vision-tower tokens or LM-input embeddings) is decided
    by the distillation task, not by the terms.
    """

    def __init__(self, terms: Sequence[Mapping[str, Any]]):
        super().__init__()
        self.labels: list[str] = []
        self.weights: list[float] = []
        losses = []
        for i, term in enumerate(terms):
            term = dict(term)
            weight = float(term.pop("weight", 1.0))
            label = term.pop("label", None) or f"{term['name']}_{i}"
            losses.append(build_loss(term))
            self.labels.append(label)
            self.weights.append(weight)
        self.losses = nn.ModuleList(losses)

    def describe(self) -> str:
        return " + ".join(f"{w:g}*{label}[{type(loss).__name__}]" for label, w, loss in zip(self.labels, self.weights, self.losses))

    def forward(self, student: StudentVisionOutput, teacher: TeacherVisionOutput, prefix: str = "") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = student.tokens.new_zeros((), dtype=torch.float32)
        logs: dict[str, torch.Tensor] = {}
        for label, w, loss in zip(self.labels, self.weights, self.losses):
            v = loss(student, teacher)
            if v is None:  # term undefined for this batch: no contribution; logged as 0 only in training (DDP key sync)
                if self.training:
                    logs[prefix + label] = total.new_zeros(())
                continue
            logs[prefix + label] = v.detach()
            total = total + w * v
        return total, logs
