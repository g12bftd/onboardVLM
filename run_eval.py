"""Benchmark evaluation of the Gemma 4 teacher or any grafted student.

    # teacher baseline (original vision tower)
    python run_eval.py benchmark=vizwiz_vqa
    # grafted student (export of any distillation task)
    python run_eval.py benchmark=textvqa student_dir=outputs/students/feature_compact_vit_s_e2b run_name=cvit_s
    # all benchmarks (Hydra multirun)
    python run_eval.py -m benchmark=vizwiz_vqa,textvqa,wearvqa,egoblind,egotextvqa_outdoor_high,egotextvqa_outdoor_low,egotextvqa_indoor,egoschema_subset,cvbench
    # offline dry run
    python run_eval.py benchmark=synthetic_debug tiny=true device=cpu
"""

from __future__ import annotations

import json

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from onboard_vlm.eval.runner import run_benchmark
from onboard_vlm.registry import build_dataset, build_student
from onboard_vlm.utils.config import to_plain
from onboard_vlm.utils.logging import get_logger, setup_logging, stage

log = get_logger("onboard_vlm.eval")


def _load_model(cfg: DictConfig):
    dtype = getattr(torch, cfg.dtype)
    if cfg.tiny:  # random tiny VLM + offline processor
        from transformers import Gemma4ForConditionalGeneration

        from onboard_vlm.students.io import load_student
        from onboard_vlm.utils.tiny import tiny_gemma4_config, tiny_gemma4_processor
        from onboard_vlm.vlm.grafted import GraftedVLM

        vlm = Gemma4ForConditionalGeneration(tiny_gemma4_config()).to(dtype)
        model = vlm
        if cfg.student_dir is not None:
            model = GraftedVLM(vlm, load_student(cfg.student_dir)[0].to(dtype))
        elif cfg.get("tiny_student"):
            model = GraftedVLM(vlm, build_student(to_plain(cfg.tiny_student)).to(dtype))
        return model.to(cfg.device), tiny_gemma4_processor()

    from transformers import AutoProcessor

    from onboard_vlm.teachers.gemma4 import load_gemma4_vlm, resolve_revision
    from onboard_vlm.vlm.grafted import GraftedVLM

    rev = resolve_revision(cfg.vlm_id, cfg.get("revision"))
    processor = AutoProcessor.from_pretrained(cfg.vlm_id, revision=rev)
    if cfg.student_dir is None:  # teacher baseline: the original VLM
        model = load_gemma4_vlm(cfg.vlm_id, rev, dtype)
    else:
        model = GraftedVLM.from_pretrained(cfg.vlm_id, cfg.student_dir, dtype=dtype, revision=rev)
    return model.to(cfg.device), processor


@hydra.main(config_path="config", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging(HydraConfig.get().runtime.output_dir)
    log.info("===> Evaluation '%s' on %s\n%s", cfg.run_name, cfg.benchmark.name, OmegaConf.to_yaml(cfg, resolve=True))
    with stage("Loading " + ("original Gemma 4 VLM (teacher baseline)" if cfg.student_dir is None else f"VLM with grafted student {cfg.student_dir}"), log):
        model, processor = _load_model(cfg)
    with stage(f"Setting up benchmark {cfg.benchmark.name}", log):
        dataset = build_dataset(to_plain(cfg.benchmark.dataset))
        log.info("     %s samples", f"{len(dataset):,}")
    with stage(f"Running {cfg.benchmark.name} -> {cfg.output_dir}", log):
        metrics = run_benchmark(
            model, processor, cfg.benchmark.spec, dataset, cfg.output_dir,
            batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            image_token_budget=cfg.image_token_budget, video_token_budget=cfg.video_token_budget,
            max_new_tokens=cfg.get("max_new_tokens"), judge=to_plain(cfg.judge) if cfg.use_judge else None,
            answerability_scoring=cfg.answerability_scoring, model_name=cfg.run_name, device=cfg.device,
        )
    log.info("<=== metrics\n%s", json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
