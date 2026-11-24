"""Generate the teacher's own responses for logit distillation (sequence-level KD targets).

    python -u run_generate_targets.py                                   # config/generate_targets.yaml
    python -u run_generate_targets.py split_name=val 'sources=${val_sources}'
    python -u run_generate_targets.py shard_id=0 num_shards=8 device=cuda:0      # ... shard_id=7

The original Gemma 4 VLM (``teacher`` config, pinned revision) answers every prompt greedily at
``budget`` soft tokens with the processor's own geometry. Each record stores the image, the exact
prompt (``question``) and the response (``answers[0]``) plus provenance; train with
``data_module=teacher_responses`` (``prompt: plain``, ``with_answer: true``).
"""

from __future__ import annotations

import json
import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset

from onboard_vlm.data.collator import VLMEvalCollator
from onboard_vlm.data.lmdb_io import LMDBWriter, VQARecord, encode_image
from onboard_vlm.data.prompts import Gemma4ChatInputs
from onboard_vlm.data.teacher_targets import PromptedImages
from onboard_vlm.utils.config import to_plain
from onboard_vlm.utils.logging import get_logger, setup_logging, stage

log = get_logger("onboard_vlm.generate")


def _load(cfg: DictConfig):
    dtype = getattr(torch, cfg.dtype)
    if cfg.teacher.get("tiny") is not None:
        from transformers import Gemma4ForConditionalGeneration

        from onboard_vlm.utils.tiny import tiny_gemma4_config, tiny_gemma4_processor

        return Gemma4ForConditionalGeneration(tiny_gemma4_config(**cfg.teacher.tiny)).to(dtype).eval(), tiny_gemma4_processor(), "tiny"
    from transformers import AutoProcessor

    from onboard_vlm.teachers.gemma4 import load_gemma4_vlm, resolve_revision

    rev = resolve_revision(cfg.teacher.model_id, cfg.teacher.get("revision"))
    return load_gemma4_vlm(cfg.teacher.model_id, rev, dtype), AutoProcessor.from_pretrained(cfg.teacher.model_id, revision=rev), rev


@hydra.main(config_path="config", config_name="generate_targets", version_base=None)
def main(cfg: DictConfig) -> None:
    log_ = setup_logging(HydraConfig.get().runtime.output_dir)
    log_.info("===> Generating teacher responses\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    with stage("Loading the original Gemma 4 VLM", log):
        vlm, processor, rev = _load(cfg)
        vlm.to(cfg.device)
    with stage("Setting up prompted images", log):
        parts = []
        for src in to_plain(cfg.sources):
            src = dict(src)
            base = {**to_plain(cfg.dataset_defaults).get(src["dataset"]["name"], {}), **src.pop("dataset")}
            ds = PromptedImages(base, templates=to_plain(cfg.templates), seed=cfg.seed, **src)
            log.info("     %s -> %s prompts", base.get("root") or base.get("path"), f"{len(ds):,}")
            parts.append(ds)
        data = ConcatDataset(parts)
        if cfg.num_shards > 1:
            data = Subset(data, list(range(cfg.shard_id, len(data), cfg.num_shards)))
        log.info("     shard %d/%d: %s prompts at %d soft tokens", cfg.shard_id, cfg.num_shards, f"{len(data):,}", cfg.budget)
    builder = Gemma4ChatInputs(processor, prompt="plain")
    loader = DataLoader(data, batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=_Collate(builder, cfg.budget))
    tok = processor.tokenizer
    os.makedirs(cfg.output_dir, exist_ok=True)
    gen_cfg = {"do_sample": False, "max_new_tokens": int(cfg.max_new_tokens)}
    provenance = {"teacher": cfg.teacher.get("model_id", "tiny"), "revision": rev, "budget": int(cfg.budget), "generation": gen_cfg}
    n = skipped = 0
    with stage(f"Generating -> {cfg.output_dir}/part_{cfg.shard_id:03d}.lmdb", log), LMDBWriter(cfg.output_dir, f"part_{cfg.shard_id:03d}", "teacher_responses") as w:
        for inputs, samples in loader:
            inputs = {k: v.to(cfg.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            with torch.no_grad():
                out = vlm.generate(**inputs, **gen_cfg)
            new = out[:, inputs["input_ids"].shape[1] :]
            texts = tok.batch_decode(new, skip_special_tokens=True)
            for s, text, ids in zip(samples, texts, new):
                if cfg.skip_empty and not text.strip():
                    skipped += 1
                    continue
                enc, hw = encode_image(s["image_pil"], max_side=None)
                finished = bool((ids == tok.eos_token_id).any()) if tok.eos_token_id is not None else True
                w.add(VQARecord(
                    sample_id=s["sample_id"], dataset="teacher_responses", split="train", image=enc, question=s["question"],
                    answers=[text.strip()], category=s["template"],
                    meta={"source_id": s["source_id"], "source_dataset": s.get("dataset"), "finished": finished, **provenance},
                ), hw)
                n += 1
            log.info("     %s responses (%d empty skipped)", f"{n:,}", skipped)
    with open(os.path.join(cfg.output_dir, f"part_{cfg.shard_id:03d}.provenance.json"), "w") as f:
        json.dump({**provenance, "num_records": n, "num_empty_skipped": skipped, "templates": to_plain(cfg.templates), "sources": to_plain(cfg.sources)}, f, indent=1)


class _Collate:
    """Processor inputs for generation + the samples (with a PIL copy of the source image to store)."""

    def __init__(self, builder: Gemma4ChatInputs, budget: int):
        self.inner = VLMEvalCollator(builder, image_token_budget=budget)

    def __call__(self, samples):
        from PIL import Image

        inputs, _ = self.inner(samples)
        for s in samples:
            s["image_pil"] = Image.fromarray(s["image"].permute(1, 2, 0).numpy())
        return inputs, samples


if __name__ == "__main__":
    main()
