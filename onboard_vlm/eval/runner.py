"""Benchmark runner: identical generation path for the Gemma teacher and every grafted student."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from onboard_vlm.data.collator import VLMEvalCollator
from onboard_vlm.data.prompts import Gemma4ChatInputs
from onboard_vlm.eval.benchmarks import BENCHMARKS, BenchmarkSpec
from onboard_vlm.eval.metrics.llm_judge import CachedJudge, OpenAICompatibleJudge

log = logging.getLogger(__name__)


def _to(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


@torch.no_grad()
def continuation_logprob(model: torch.nn.Module, inputs: dict[str, Any], cont_ids: torch.Tensor) -> torch.Tensor:
    """``log p(cont | prompt)`` per row for left-padded prompts. cont_ids ``[C]`` -> ``[B]``."""
    b = inputs["input_ids"].shape[0]
    c = cont_ids.numel()
    cont = cont_ids.to(inputs["input_ids"].device)[None].expand(b, c)
    x = dict(inputs)
    x["input_ids"] = torch.cat([inputs["input_ids"], cont], 1)
    x["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(cont)], 1)
    if "mm_token_type_ids" in inputs:
        x["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], torch.zeros_like(cont)], 1)
    logits = model(**x, logits_to_keep=c + 1).logits[:, :c].float()  # positions predicting cont tokens
    return torch.log_softmax(logits, -1).gather(-1, cont[..., None]).squeeze(-1).sum(-1)


def build_judge(cfg: dict[str, Any] | None, cache_path: str) -> CachedJudge | None:
    if not cfg:
        return None
    cfg = dict(cfg)
    backend = cfg.pop("backend", "openai")
    if backend != "openai":
        raise ValueError(f"unknown judge backend {backend!r} (use 'openai' for any OpenAI-compatible endpoint)")
    return CachedJudge(OpenAICompatibleJudge(**cfg), cache_path)


def run_benchmark(
    model: torch.nn.Module,
    processor: Any,
    benchmark: str | BenchmarkSpec,
    dataset: Dataset,
    out_dir: str,
    *,
    batch_size: int = 8,
    num_workers: int = 4,
    image_token_budget: int = 280,
    video_token_budget: int = 70,
    max_new_tokens: int | None = None,
    judge: dict[str, Any] | None = None,
    answerability_scoring: bool = True,
    model_name: str = "onboard_vlm",
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Generate answers for ``dataset``, score them with the benchmark's official metric and export
    submission files. ``model`` is a ``GraftedVLM`` (student) or a plain ``Gemma4ForConditionalGeneration``.
    """
    spec = BENCHMARKS[benchmark] if isinstance(benchmark, str) else benchmark
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device) if device is not None else next(model.parameters()).device
    builder = Gemma4ChatInputs(processor, prompt=spec.prompt)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=VLMEvalCollator(builder, image_token_budget, video_token_budget),
    )
    tok = processor.tokenizer
    unans_ids = torch.tensor(tok("Unanswerable", add_special_tokens=False)["input_ids"]) if spec.answerability and answerability_scoring else None

    records: list[dict[str, Any]] = []
    pred_path = os.path.join(out_dir, "predictions.jsonl")
    model.eval()
    with open(pred_path, "w") as fout:
        for inputs, extras in loader:
            inputs = _to(inputs, device)
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens or spec.max_new_tokens, do_sample=False)
            texts = tok.batch_decode(gen[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            ans_scores = None
            if unans_ids is not None:
                ans_scores = (1.0 - continuation_logprob(model, inputs, unans_ids).exp()).tolist()
            for i, (raw, ex) in enumerate(zip(texts, extras)):
                rec = {**ex, "raw_prediction": raw, "prediction": spec.postprocess(raw)}
                if ans_scores is not None:
                    rec["answerability_score"] = ans_scores[i]
                records.append(rec)
                fout.write(json.dumps(rec, default=str) + "\n")

    cached_judge = build_judge(judge, os.path.join(out_dir, "judge_cache.jsonl")) if spec.uses_judge else None
    metrics = spec.scorer(records, cached_judge) if spec.uses_judge else spec.scorer(records)
    metrics.update(benchmark=spec.name, num_predictions=len(records), image_token_budget=image_token_budget, video_token_budget=video_token_budget)
    if spec.export is not None:
        if spec.name == "egotextvqa":
            spec.export(records, out_dir, model_name)  # type: ignore[call-arg]
        else:
            spec.export(records, out_dir)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info("%s: %s", spec.name, json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, default=str))
    return metrics
