"""Collators. Every image is geometrically transformed exactly once, here, and the result is
branched into the teacher view (Gemma patch sequence) and the student view (dense tensor)."""

from __future__ import annotations

import random
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import get_worker_info

from onboard_vlm.geometry import (
    apply_transform,
    build_teacher_patches,
    pad_dense,
    sample_bucket_transform,
    sample_transform,
    to_unit_float,
)
from onboard_vlm.interfaces import (
    POOLING_KERNEL,
    SUPPORTED_TOKEN_BUDGETS,
    DistillBatch,
    TeacherInputs,
    TeacherVisionOutput,
    TransformParams,
)

_PASSTHROUGH = ("question", "answers", "choices", "answer_index", "answerable", "answer_type", "category", "meta", "dataset", "frame_timestamps")


class _RngMixin:
    """Per-process / per-worker RNG so multi-worker loaders do not replay identical budgets/crops."""

    seed: int

    def _rng(self) -> random.Random:
        wi = get_worker_info()
        wid = wi.id if wi is not None else -1
        if getattr(self, "_rng_owner", None) != wid:
            self._rng_obj = random.Random(hash((self.seed, wid, torch.initial_seed())))
            self._rng_owner = wid
        return self._rng_obj


class DualViewCollator(_RngMixin):
    """Collator for all distillation tasks, producing :class:`DistillBatch`.

    Args:
        token_budgets: one budget or a list to sample from per batch (multi-resolution training).
        train: random-resized crop + flip (else the full image, exactly like Gemma's processor).
        crop_scale, crop_ratio, hflip_prob: augmentation parameters (see ``geometry.sample_transform``).
        degradations: ``{op: probability}`` capture degradations (``data.degrade``), train only.
        build_teacher_inputs: also emit Gemma patch inputs (disable when teacher features are cached).
        build_student_view: emit the dense student tensor (logit distillation feeds students through the grafted
            VLM's patch path instead, so it can be disabled there; ``student_images`` is then empty).
        text_builder: callable ``(samples, images_u8, budget) -> dict`` (logit distillation), e.g.
            :class:`onboard_vlm.data.prompts.Gemma4ChatInputs` with ``with_answer=True``.
        seed: base seed of the per-worker RNG.
    """

    def __init__(
        self,
        token_budgets: int | Sequence[int] = 280,
        train: bool = False,
        crop_scale: Sequence[float] = (0.5, 1.0),
        crop_ratio: Sequence[float] = (3 / 4, 4 / 3),
        hflip_prob: float = 0.0,
        degradations: dict[str, Any] | None = None,
        build_teacher_inputs: bool = True,
        build_student_view: bool = True,
        text_builder: Callable[..., dict[str, torch.Tensor]] | None = None,
        seed: int = 0,
    ):
        self.token_budgets = [token_budgets] if isinstance(token_budgets, int) else list(token_budgets)
        bad = set(self.token_budgets) - set(SUPPORTED_TOKEN_BUDGETS)
        if bad:
            raise ValueError(f"unsupported token budgets {sorted(bad)}; use {SUPPORTED_TOKEN_BUDGETS}")
        self.train = train
        self.crop_scale = tuple(crop_scale)
        self.crop_ratio = tuple(crop_ratio)
        self.hflip_prob = hflip_prob
        self.degradations = dict(degradations or {})
        self.build_teacher_inputs = build_teacher_inputs
        self.build_student_view = build_student_view
        self.text_builder = text_builder
        self.seed = seed

    def _transform(self, sample: dict[str, Any], budget: int, rng: random.Random) -> tuple[torch.Tensor, TransformParams]:
        img = sample["image"]
        params: TransformParams | None = sample.get("transform")  # cached / pre-specified transform
        spec = sample.get("batch_spec")
        if spec is not None:  # shape-exact batching (BucketedBatchSampler / ExactShapeBatchSampler)
            if params is not None:
                raise ValueError("cached transforms cannot be combined with shape-bucketed batches")
            mode, b, h, w = spec
            if mode == "crop":
                params = sample_bucket_transform(
                    tuple(img.shape[-2:]), (h, w), b, rng=rng, crop_scale=self.crop_scale,
                    hflip_prob=self.hflip_prob, degradations=self.degradations if self.train else None,
                )
            else:  # "full": the processor's own geometry, which must give the batch's shape
                params = sample_transform(tuple(img.shape[-2:]), b, train=False)
                if tuple(params.target_hw) != (h, w):
                    raise RuntimeError(f"{sample.get('sample_id')}: processor shape {params.target_hw} != batch shape {(h, w)}")
        if params is None:
            params = sample_transform(
                tuple(img.shape[-2:]), budget, train=self.train, rng=rng,
                crop_scale=self.crop_scale, crop_ratio=self.crop_ratio, hflip_prob=self.hflip_prob,
                degradations=self.degradations,
            )
        return apply_transform(img, params), params

    def __call__(self, samples: list[dict[str, Any]]) -> DistillBatch:
        if any("image" not in s for s in samples):
            raise ValueError("DualViewCollator expects image samples (use LMDBVQADataset(mode='image') for videos)")
        rng = self._rng()
        specs = {s["batch_spec"] for s in samples if "batch_spec" in s}
        if len(specs) > 1:
            raise RuntimeError(f"one batch spans several shapes {specs}")
        budget = next(iter(specs))[1] if specs else rng.choice(self.token_budgets)
        out = [self._transform(s, budget, rng) for s in samples]
        images_u8 = [o[0] for o in out]
        transforms = [o[1] for o in out]
        images = [to_unit_float(im) for im in images_u8]

        student_images, valid_mask, lengths = pad_dense(images)
        if not self.build_student_view:
            student_images = student_images.new_zeros((len(images), 3, 0, 0))
        teacher_inputs = None
        if self.build_teacher_inputs:
            max_patches = max(t.token_budget for t in transforms) * POOLING_KERNEL**2
            pv, pos = build_teacher_patches(images, max_patches=max_patches)
            teacher_inputs = TeacherInputs(pv, pos, list(lengths))

        cached = None
        if all("teacher_tokens" in s for s in samples):
            layers = samples[0]["teacher_layers"].keys()
            cached = TeacherVisionOutput(
                tokens=torch.cat([s["teacher_tokens"] for s in samples]),
                intermediate_tokens={i: torch.cat([s["teacher_layers"][i] for s in samples]) for i in layers},
                image_token_lengths=[s["teacher_tokens"].shape[0] for s in samples],
            )
            if cached.image_token_lengths != lengths:
                raise RuntimeError(f"cached teacher lengths {cached.image_token_lengths} != student lengths {lengths}")

        text_inputs = None
        if self.text_builder is not None:
            text_inputs = self.text_builder(samples, images_u8, budget)
            n_soft = text_inputs.pop("num_soft_tokens_per_image", None)
            if n_soft is not None and [int(x) for x in torch.as_tensor(n_soft).flatten()] != lengths:
                raise RuntimeError(f"processor soft-token counts {n_soft} != geometry {lengths}")
            for k in ("pixel_values", "image_position_ids"):  # we use our own (bit-identical) patches
                text_inputs.pop(k, None)

        return DistillBatch(
            teacher_inputs=teacher_inputs,
            student_images=student_images,
            valid_token_mask=valid_mask,
            image_token_lengths=lengths,
            transforms=transforms,
            sample_ids=[s["sample_id"] for s in samples],
            text_inputs=text_inputs,
            cached_teacher=cached,
            extras=[{k: s[k] for k in _PASSTHROUGH if k in s} for s in samples],
        )


class VLMEvalCollator:
    """Evaluation collator: deterministic geometry once per image/frame -> processor inputs.

    Images use ``image_token_budget``; videos use ``video_token_budget`` per frame (Gemma 4's video
    processor defaults to 70 soft tokens per frame). Returns ``(model_inputs, extras)``.
    """

    def __init__(self, text_builder: Callable[..., dict[str, torch.Tensor]], image_token_budget: int = 280, video_token_budget: int = 70):
        self.text_builder = text_builder
        self.image_token_budget = image_token_budget
        self.video_token_budget = video_token_budget

    def __call__(self, samples: list[dict[str, Any]]) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
        kinds = {"frames" in s for s in samples}
        if len(kinds) != 1:
            raise ValueError("cannot mix image and video samples in one eval batch")
        is_video = kinds.pop()
        budget = self.video_token_budget if is_video else self.image_token_budget
        visuals = []
        for s in samples:
            x = s["frames"] if is_video else s["image"]
            visuals.append(apply_transform(x, sample_transform(tuple(x.shape[-2:]), budget, train=False)))
        ts = [s.get("frame_timestamps") for s in samples] if is_video else None
        inputs = self.text_builder(samples, visuals, budget, ts) if is_video else self.text_builder(samples, visuals, budget)
        extras = [{"sample_id": s["sample_id"], **{k: s[k] for k in _PASSTHROUGH if k in s}} for s in samples]
        return dict(inputs), extras
