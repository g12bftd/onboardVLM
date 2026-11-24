"""E2: frozen-feature probes, one protocol for our students, the teacher and external encoders.

Encoders return a token grid ``[gh, gw, D]`` per image:

* ``gemma``: the teacher or any exported / configured student, fed the Gemma 4 processor output at a
  token budget (the deployment input), shape-exact batches;
* ``timm``: a pretrained timm model at its native input resolution and normalisation (final feature map);
* ``hf_vision``: a Hugging Face vision tower (e.g. SigLIP2) at its native resolution (patch tokens).

Probes (features are extracted once and cached in memory; the probe itself is trained on them):

* classification: weighted k-NN (cosine, k=20, T=0.07; DINO protocol) and a linear probe (AdamW,
  weight decay chosen on a held-out 10% of the train features) on mean-pooled tokens; top-1 accuracy;
* segmentation: a per-token linear classifier, logits bilinearly upsampled to the label map,
  cross-entropy with ignore index; mIoU;
* depth: a per-token linear regressor of log-depth, bilinearly upsampled; RMSE, AbsRel, delta<1.25.

Per-item scores are returned so ``onboard_vlm.eval.stats`` can attach bootstrap CIs.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from onboard_vlm.utils.logging import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------------------------------------------
# Encoders: list[UInt8[3, H, W]] -> list[Float[gh, gw, D]]
# ----------------------------------------------------------------------------------------------
class GemmaGridEncoder:
    """Teacher (``Gemma4Teacher``) or student (``VisionStudent``) on the processor output at ``budget``."""

    def __init__(self, model: nn.Module, budget: int, is_teacher: bool, device: str):
        self.model, self.budget, self.is_teacher, self.device = model.eval(), budget, is_teacher, device

    @torch.no_grad()
    def __call__(self, images: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        from onboard_vlm.data.collator import DualViewCollator
        from onboard_vlm.geometry import target_size

        out: list[torch.Tensor | None] = [None] * len(images)
        groups: dict[tuple[int, int], list[int]] = {}
        for i, im in enumerate(images):
            groups.setdefault(target_size(*im.shape[-2:], self.budget), []).append(i)
        collate = DualViewCollator(self.budget, train=False)
        for (h, w), idx in groups.items():
            batch = collate([{"sample_id": str(i), "image": images[i]} for i in idx]).to(self.device)
            if self.is_teacher:
                tok = self.model.encode_vision(batch.teacher_inputs).tokens
            else:
                from onboard_vlm.distill.fidelity import student_tokens

                tok = student_tokens(self.model, batch)
            gh, gw = h // 48, w // 48
            for j, t in zip(idx, tok.float().split([gh * gw] * len(idx))):
                out[j] = t.reshape(gh, gw, -1).cpu()
        return out  # type: ignore[return-value]


class TimmGridEncoder:
    """A pretrained timm model at its native input size and normalisation; the final feature map."""

    def __init__(self, model_name: str, device: str, input_size: int | None = None):
        import timm

        self.model = timm.create_model(model_name, pretrained=True, features_only=True).to(device).eval()
        cfg = self.model.pretrained_cfg
        self.size = input_size or cfg.get("input_size", (3, 224, 224))[-1]
        self.mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)), device=device).view(1, 3, 1, 1)
        self.device = device

    @torch.no_grad()
    def __call__(self, images: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        x = torch.stack([F.interpolate(im[None].float().to(self.device) / 255, size=(self.size, self.size), mode="bicubic",
                                       antialias=True, align_corners=False)[0] for im in images])
        f = self.model((x - self.mean) / self.std)[-1]
        if f.shape[1] != self.model.feature_info.channels()[-1]:  # NHWC models
            f = f.permute(0, 3, 1, 2)
        return [g.permute(1, 2, 0).float().cpu() for g in f]


class HFVisionGridEncoder:
    """A Hugging Face vision tower (``AutoModel`` with a ``vision_model``) at its native resolution."""

    def __init__(self, model_id: str, device: str):
        from transformers import AutoImageProcessor, AutoModel

        model = AutoModel.from_pretrained(model_id)
        self.model = getattr(model, "vision_model", model).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.device = device

    @torch.no_grad()
    def __call__(self, images: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        inputs = self.processor(images=[im.permute(1, 2, 0).numpy() for im in images], return_tensors="pt").to(self.device)
        h = self.model(**inputs).last_hidden_state
        n = h.shape[1]
        side = int(math.isqrt(n))
        if side * side != n:  # drop a class token
            h, side = h[:, 1:], int(math.isqrt(n - 1))
        return [t.reshape(side, side, -1).float().cpu() for t in h]


# ----------------------------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------------------------
def extract(encoder: Callable, dataset, batch_size: int = 32, pooled: bool = False, limit: int | None = None) -> list[Any]:
    """Run ``encoder`` over ``dataset`` items (dicts with ``image``); return grids (or mean-pooled vectors)."""
    n = len(dataset) if limit is None else min(limit, len(dataset))
    feats: list[Any] = []
    for s in range(0, n, batch_size):
        items = [dataset[i] for i in range(s, min(n, s + batch_size))]
        grids = encoder([it["image"] for it in items])
        feats.extend(g.mean((0, 1)) if pooled else g for g in grids)
    return feats


# ----------------------------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------------------------
def knn_classify(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, k: int = 20, temperature: float = 0.07,
                 num_classes: int | None = None, chunk: int = 1024) -> torch.Tensor:
    """Temperature-weighted cosine k-NN (DINO). Returns predicted labels."""
    num_classes = num_classes or int(train_y.max()) + 1
    tr = F.normalize(train_x.float(), dim=1)
    preds = []
    for s in range(0, len(test_x), chunk):
        sim = F.normalize(test_x[s : s + chunk].float(), dim=1) @ tr.T
        w, idx = sim.topk(min(k, len(tr)), dim=1)
        votes = torch.zeros(len(sim), num_classes).scatter_add_(1, train_y[idx], (w / temperature).exp())
        preds.append(votes.argmax(1))
    return torch.cat(preds)


def _fit_logreg(x: torch.Tensor, y: torch.Tensor, num_classes: int, wd: float, device: str, max_iter: int = 500) -> nn.Linear:
    """Multinomial logistic regression, full-batch L-BFGS, L2 penalty ``wd/2 ||W||^2`` (deterministic)."""
    head = nn.Linear(x.shape[1], num_classes).to(device)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    x, y = x.to(device), y.to(device)
    opt = torch.optim.LBFGS(head.parameters(), lr=1.0, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(head(x), y) + 0.5 * wd * head.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return head.eval()


def linear_classify(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, num_classes: int, device: str = "cpu",
                    weight_decays: Sequence[float] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2), max_iter: int = 500, seed: int = 0) -> tuple[torch.Tensor, float]:
    """Linear probe (logistic regression) on standardised features; the L2 strength is selected on a
    held-out 10% of the training features, then the probe is refit on all of them."""
    mu, sd = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
    tx, ex = ((train_x - mu) / sd).float(), ((test_x - mu) / sd).float()
    perm = torch.randperm(len(tx), generator=torch.Generator().manual_seed(seed))
    n_val = max(1, len(tx) // 10)
    va, tr = perm[:n_val], perm[n_val:]

    def val_acc(wd: float) -> float:
        head = _fit_logreg(tx[tr], train_y[tr], num_classes, wd, device, max_iter)
        with torch.no_grad():
            return (head(tx[va].to(device)).argmax(1).cpu() == train_y[va]).float().mean().item()

    best = max(weight_decays, key=val_acc)
    head = _fit_logreg(tx, train_y, num_classes, best, device, max_iter)
    with torch.no_grad():
        return head(ex.to(device)).argmax(1).cpu(), best


# ----------------------------------------------------------------------------------------------
# Dense probes
# ----------------------------------------------------------------------------------------------
class DenseLinearProbe(nn.Module):
    def __init__(self, dim: int, out: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim, affine=False)  # standardise features (frozen after training)
        self.proj = nn.Conv2d(dim, out, 1)

    def forward(self, grid: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(self.proj(self.norm(grid)), size=size, mode="bilinear", align_corners=False)


def train_dense_probe(grids: Sequence[torch.Tensor], targets: Sequence[torch.Tensor], out: int, task: str, device: str = "cpu",
                      epochs: int = 20, lr: float = 1e-3, wd: float = 1e-4, seed: int = 0, ignore_index: int = 255) -> DenseLinearProbe:
    """``grids[i]``: ``[gh, gw, D]``; ``targets[i]``: label map ``[H, W]`` (long) or log-depth ``[H, W]`` (float, NaN = invalid)."""
    torch.manual_seed(seed)
    probe = DenseLinearProbe(grids[0].shape[-1], out).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    order = np.random.default_rng(seed)
    for _ in range(epochs):
        probe.train()
        for i in order.permutation(len(grids)):
            g = grids[i].permute(2, 0, 1)[None].to(device)
            y = targets[i].to(device)
            pred = probe(g, tuple(y.shape))
            if task == "seg":
                loss = F.cross_entropy(pred, y[None], ignore_index=ignore_index)
            else:
                m = ~torch.isnan(y)
                loss = (pred[0, 0][m] - y[m]).abs().mean() if m.any() else pred.sum() * 0
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe.eval()


@torch.no_grad()
def eval_segmentation(probe: DenseLinearProbe, grids, labels, num_classes: int, device: str = "cpu", ignore_index: int = 255) -> dict[str, Any]:
    inter = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    per_image = []
    for g, y in zip(grids, labels):
        pred = probe(g.permute(2, 0, 1)[None].to(device), tuple(y.shape))[0].argmax(0).cpu()
        valid = y != ignore_index
        p, t = pred[valid].numpy(), y[valid].numpy()
        ii = np.bincount(t[p == t], minlength=num_classes)[:num_classes]
        uu = (np.bincount(p, minlength=num_classes) + np.bincount(t, minlength=num_classes))[:num_classes] - ii
        inter += ii
        union += uu
        per_image.append(float((p == t).mean()) if len(t) else float("nan"))
    iou = inter / np.maximum(union, 1)
    return {"miou": float(iou[union > 0].mean()), "pixel_acc_per_image": per_image}


@torch.no_grad()
def eval_depth(probe: DenseLinearProbe, grids, log_depths, device: str = "cpu") -> dict[str, Any]:
    rmse, absrel, d1 = [], [], []
    for g, y in zip(grids, log_depths):
        pred = probe(g.permute(2, 0, 1)[None].to(device), tuple(y.shape))[0, 0].cpu()
        m = ~torch.isnan(y)
        p, t = pred[m].exp(), y[m].exp()
        rmse.append(float(((p - t) ** 2).mean().sqrt()))
        absrel.append(float(((p - t).abs() / t).mean()))
        d1.append(float((torch.maximum(p / t, t / p) < 1.25).float().mean()))
    return {"rmse": rmse, "absrel": absrel, "delta1": d1}
