"""Precompute teacher soft tokens (+ intermediate layers) for fast feature-distillation sweeps.

    python run_cache_teacher.py teacher=gemma4_e2b dataset.path=$ONBOARD_VLM_DATA/vizwiz layers=[3,9,15]
    # shard across GPUs / jobs:
    python run_cache_teacher.py shard_id=0 num_shards=4 device=cuda:0  (... shard_id=3 device=cuda:3)

Then train with ``data_module=cached_benchmark_train cache_root=<cache_root>`` (any train entry can be
wrapped as ``{name: cached_teacher, base: {...}, cache_root: ...}``). Each view stores its exact
TransformParams, so student inputs are reproduced bit-exactly from the source image.
"""

from __future__ import annotations

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from onboard_vlm.data.collator import DualViewCollator
from onboard_vlm.registry import build_dataset, build_teacher
from onboard_vlm.teachers.feature_cache import TeacherFeatureCacheWriter
from onboard_vlm.utils.config import to_plain
from onboard_vlm.utils.logging import setup_logging, stage


@hydra.main(config_path="config", config_name="cache_teacher", version_base=None)
def main(cfg: DictConfig) -> None:
    log = setup_logging(HydraConfig.get().runtime.output_dir)
    log.info("===> Teacher-feature caching\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    with stage(f"Loading teacher {cfg.teacher.get('model_id', 'tiny')} on {cfg.device}", log):
        teacher = build_teacher(to_plain(cfg.teacher)).to(cfg.device)
    with stage("Setting up dataset", log):
        dataset = build_dataset(to_plain(cfg.dataset))
        if cfg.num_shards > 1:
            dataset = Subset(dataset, list(range(cfg.shard_id, len(dataset), cfg.num_shards)))
        log.info("     %s images (shard %d/%d)", f"{len(dataset):,}", cfg.shard_id, cfg.num_shards)
    layers = list(cfg.layers)
    meta = {
        "teacher": to_plain(cfg.teacher), "hidden_size": teacher.hidden_size, "layers": layers,
        "token_budgets": list(cfg.token_budgets), "views_per_image": cfg.views_per_image,
        "train_augment": cfg.train_augment, "dtype": cfg.cache_dtype,
    }
    writer = TeacherFeatureCacheWriter(cfg.cache_root, meta, cfg.shard_size, getattr(torch, cfg.cache_dtype), rank=cfg.shard_id)
    for view in range(cfg.views_per_image):
        with stage(f"Caching view {view + 1}/{cfg.views_per_image} (layers {layers}, budgets {list(cfg.token_budgets)})", log):
            collate = DualViewCollator(list(cfg.token_budgets), train=cfg.train_augment, hflip_prob=cfg.hflip_prob, seed=1000 * view + cfg.shard_id)
            loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate)
            for i, batch in enumerate(loader):
                with torch.no_grad():
                    out = teacher.encode_vision(batch.teacher_inputs.to(cfg.device), layers)
                writer.add_batch(batch.sample_ids, batch.transforms, out)
                if i % 50 == 0:
                    log.info("     view %d batch %d/%d", view, i, len(loader))
    with stage(f"Writing cache index to {cfg.cache_root}", log):
        writer.close()


if __name__ == "__main__":
    main()
