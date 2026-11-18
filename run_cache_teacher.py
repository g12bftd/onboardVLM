"""Precompute teacher soft tokens (+ intermediate layers) for fast Stage-1 sweeps.

    python run_cache_teacher.py teacher=gemma4_e2b dataset.path=$ONBOARD_VLM_DATA/vizwiz layers=[3,9,15]
    # shard across GPUs / jobs:
    python run_cache_teacher.py shard_id=0 num_shards=4 device=cuda:0  (... shard_id=3 device=cuda:3)

Then train with ``data_module=stage1_cached cache_root=<cache_root>``. Each view stores its exact
TransformParams, so student inputs are reproduced bit-exactly from the source image.
"""

from __future__ import annotations

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from onboard_vlm.data.collator import DualViewCollator
from onboard_vlm.registry import build_dataset, build_teacher
from onboard_vlm.teachers.feature_cache import TeacherFeatureCacheWriter
from onboard_vlm.utils.config import to_plain

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="cache_teacher", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    teacher = build_teacher(to_plain(cfg.teacher)).to(cfg.device)
    dataset = build_dataset(to_plain(cfg.dataset))
    if cfg.num_shards > 1:
        dataset = Subset(dataset, list(range(cfg.shard_id, len(dataset), cfg.num_shards)))
    layers = list(cfg.layers)
    meta = {
        "teacher": to_plain(cfg.teacher), "hidden_size": teacher.hidden_size, "layers": layers,
        "token_budgets": list(cfg.token_budgets), "views_per_image": cfg.views_per_image,
        "train_augment": cfg.train_augment, "dtype": cfg.cache_dtype,
    }
    writer = TeacherFeatureCacheWriter(cfg.cache_root, meta, cfg.shard_size, getattr(torch, cfg.cache_dtype), rank=cfg.shard_id)
    for view in range(cfg.views_per_image):
        collate = DualViewCollator(list(cfg.token_budgets), train=cfg.train_augment, hflip_prob=cfg.hflip_prob, seed=1000 * view + cfg.shard_id)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate)
        for i, batch in enumerate(loader):
            out = teacher.encode_vision(batch.teacher_inputs.to(cfg.device), layers)
            writer.add_batch(batch.sample_ids, batch.transforms, out)
            if i % 50 == 0:
                log.info("view %d batch %d/%d", view, i, len(loader))
    writer.close()
    log.info("cache written to %s", cfg.cache_root)


if __name__ == "__main__":
    main()
