"""Benchmark evaluation of the Gemma 4 teacher or any grafted student.

    # teacher baseline (original vision tower)
    python run_eval.py benchmark=vizwiz_vqa
    # grafted student (Stage-1 or Stage-2 export)
    python run_eval.py benchmark=textvqa student_dir=outputs/students/stage2_compact_vit_s_e2b run_name=cvit_s
    # all benchmarks (Hydra multirun)
    python run_eval.py -m benchmark=vizwiz_vqa,textvqa,wearvqa,egoblind,egotextvqa_outdoor_high,egotextvqa_outdoor_low,egotextvqa_indoor,egoschema_subset,cvbench
    # offline dry run
    python run_eval.py benchmark=synthetic_debug tiny=true device=cpu
"""

from __future__ import annotations

import json
import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from onboard_vlm.eval.runner import run_benchmark
from onboard_vlm.registry import build_dataset, build_student
from onboard_vlm.utils.config import to_plain

log = logging.getLogger(__name__)


def _load_model(cfg: DictConfig):
    dtype = getattr(torch, cfg.dtype)
    if cfg.tiny:  # random tiny VLM + offline processor
        from transformers import Gemma4ForConditionalGeneration

        from onboard_vlm.utils.tiny import tiny_gemma4_config, tiny_gemma4_processor
        from onboard_vlm.vlm.grafted import GraftedVLM

        from onboard_vlm.students.io import load_student

        vlm = Gemma4ForConditionalGeneration(tiny_gemma4_config()).to(dtype)
        model = vlm
        if cfg.student_dir is not None:
            model = GraftedVLM(vlm, load_student(cfg.student_dir)[0].to(dtype))
        elif cfg.get("tiny_student"):
            model = GraftedVLM(vlm, build_student(to_plain(cfg.tiny_student)).to(dtype))
        return model.to(cfg.device), tiny_gemma4_processor()

    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    from onboard_vlm.vlm.grafted import GraftedVLM

    from onboard_vlm.teachers.hub import pinned_revision

    rev = pinned_revision(cfg.vlm_id)
    processor = AutoProcessor.from_pretrained(cfg.vlm_id, revision=rev)
    if cfg.student_dir is None:  # teacher baseline: the stock VLM at the verified commit
        model = Gemma4ForConditionalGeneration.from_pretrained(cfg.vlm_id, revision=rev, dtype=dtype, attn_implementation="sdpa")
    else:
        model = GraftedVLM.from_pretrained(cfg.vlm_id, cfg.student_dir, dtype=dtype)
    return model.to(cfg.device), processor


@hydra.main(config_path="config", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    model, processor = _load_model(cfg)
    dataset = build_dataset(to_plain(cfg.benchmark.dataset))
    metrics = run_benchmark(
        model, processor, cfg.benchmark.spec, dataset, cfg.output_dir,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        image_token_budget=cfg.image_token_budget, video_token_budget=cfg.video_token_budget,
        max_new_tokens=cfg.get("max_new_tokens"), judge=to_plain(cfg.judge) if cfg.use_judge else None,
        answerability_scoring=cfg.answerability_scoring, model_name=cfg.run_name, device=cfg.device,
    )
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
