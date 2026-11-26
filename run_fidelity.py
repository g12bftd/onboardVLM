"""E1: student-teacher fidelity on held-out images, per evaluation set x token budget x aspect ratio.

    python -u run_fidelity.py student_dir=outputs/students/embedding_gemma4_vit_l8_d384_e2b run_name=g4vit_l8_d384
    python -u run_fidelity.py 'student={name: gemma4_truncated, num_layers: 8}' run_name=teacher_first8   # baseline

For every image the teacher (original Gemma 4 tower, bf16) and the student see the same processor
output (bit-exact geometry, shape-exact batches, no padding); images outside the declared input
domain are excluded and counted. Writes:

* ``per_image.jsonl``: one record per (set, budget, image) with size, aspect, #tokens and the
  fidelity metrics of :mod:`onboard_vlm.distill.fidelity` (for correlation with downstream accuracy);
* ``summary.json`` / ``summary.md``: means with 95% bootstrap CIs per (set, budget), per aspect bin,
  token-level 5th-percentile cosine, and the out-of-domain counts.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from onboard_vlm.data.availability import available
from onboard_vlm.data.collator import DualViewCollator
from onboard_vlm.data.data_module import ShapedDataset, dataset_sizes
from onboard_vlm.data.samplers import ExactShapeBatchSampler
from onboard_vlm.distill.fidelity import per_image_fidelity, per_token_fidelity, student_tokens
from onboard_vlm.eval.stats import bootstrap_ci
from onboard_vlm.geometry import InputDomain, check_student_covers
from onboard_vlm.registry import build_dataset, build_student, build_teacher
from onboard_vlm.utils.config import to_plain
from onboard_vlm.utils.logging import describe_module, get_logger, setup_logging, stage
from onboard_vlm.utils.provenance import write_provenance

log = get_logger("onboard_vlm.fidelity")
ASPECT_BINS = (("portrait", 0.0, 0.9), ("square", 0.9, 1.1), ("landscape", 1.1, 2.0), ("wide", 2.0, float("inf")))
METRICS = ("lm_cos", "lm_rel_err", "lm_mse", "vis_cos", "vis_cos_min", "vis_mse", "vis_rel_err", "cka")


def aspect_bin(w_over_h: float) -> str:
    return next(name for name, lo, hi in ASPECT_BINS if lo <= w_over_h < hi)


def _load_student(cfg: DictConfig, dtype: torch.dtype, device: str):
    if cfg.get("student_dir"):
        from onboard_vlm.students.io import load_student

        student, scfg = load_student(cfg.student_dir)
    else:
        scfg = to_plain(cfg.student)
        if scfg.get("name") == "gemma4_truncated":  # cut from the same checkpoint as the teacher
            scfg = {**{k: v for k, v in to_plain(cfg.teacher).items() if k in ("model_id", "revision", "tiny")}, **scfg}
        student = build_student(scfg)
    return student.to(device=device, dtype=dtype).eval(), scfg


@hydra.main(config_path="config", config_name="fidelity", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging(HydraConfig.get().runtime.output_dir)
    log.info("===> E1 fidelity '%s'\n%s", cfg.run_name, OmegaConf.to_yaml(cfg, resolve=True))
    dtype, device = getattr(torch, cfg.dtype), cfg.device
    domain = InputDomain.from_cfg(to_plain(cfg.input_domain))
    with stage(f"Loading teacher {cfg.teacher.get('model_id', 'tiny')}", log):
        teacher = build_teacher({**to_plain(cfg.teacher), "load": "vision"}).to(device).eval()
    with stage("Loading student", log):
        student, scfg = _load_student(cfg, dtype, device)
        check_student_covers(student, domain)
        log.info("     %s | input domain: %s", describe_module(student), domain.describe())
    os.makedirs(cfg.output_dir, exist_ok=True)
    write_provenance(cfg.output_dir, cfg)
    rows_by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    tokens_by_key: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    excluded: dict[str, dict[int, int]] = {}
    collate = DualViewCollator(train=False)
    skipped: list[dict] = []
    with open(os.path.join(cfg.output_dir, "per_image.jsonl"), "w") as fout:
        for es in to_plain(cfg.sets):
            with stage(f"Set {es['name']} ({es.get('group', '-')})", log):
                dcfg = {**to_plain(cfg.get("dataset_defaults") or {}).get(es["dataset"]["name"], {}), **es["dataset"]}
                cap = cfg.get("max_images_per_set")
                if cap:
                    if dcfg["name"] == "image_corpus":
                        dcfg["limit"] = min(int(dcfg.get("limit") or cap), int(cap))
                    elif dcfg["name"] == "lmdb_vqa":
                        dcfg["sample"] = int(cap)
                if cfg.get("allow_missing_sources"):
                    dcfg, note = available(dcfg)
                    if note:
                        log.warning("     PRELIMINARY: %s: %s%s", es["name"], note, " -> skipped" if dcfg is None else "")
                        skipped.append({"set": es["name"], "note": note, "skipped": dcfg is None})
                    if dcfg is None:
                        continue
                ds = build_dataset(dcfg)
                sizes = dataset_sizes(ds)
                log.info("     %s images", f"{len(ds):,}")
                for b in cfg.budgets:
                    sampler = ExactShapeBatchSampler(sizes, [b], cfg.tokens_per_batch, domain=domain, num_replicas=1, rank=0)
                    excluded.setdefault(es["name"], {})[b] = sampler.excluded.get(b, 0)
                    loader = DataLoader(ShapedDataset(ds), batch_sampler=sampler, collate_fn=collate, num_workers=cfg.num_workers)
                    for batch in loader:
                        batch = batch.to(device)
                        with torch.no_grad():
                            t = teacher.encode_vision(batch.teacher_inputs).tokens
                            s = student_tokens(student, batch)
                        recs = per_image_fidelity(s, t, batch.image_token_lengths, teacher.project_to_lm, cka=cfg.cka)
                        tokens_by_key[(es["name"], b)].append(per_token_fidelity(s, t)["vis_cos"].cpu().numpy().astype(np.float32))
                        for sid, tr, r in zip(batch.sample_ids, batch.transforms, recs):
                            h, w = tr.target_hw
                            row = {"set": es["name"], "group": es.get("group"), "budget": b, "sample_id": sid, "height": h, "width": w,
                                   "aspect": round(w / h, 4), "aspect_bin": aspect_bin(w / h), **r}
                            rows_by_key[(es["name"], b)].append(row)
                            fout.write(json.dumps(row) + "\n")
                    n = len(rows_by_key[(es["name"], b)])
                    m = np.mean([r.get("lm_cos", np.nan) for r in rows_by_key[(es["name"], b)]]) if n else float("nan")
                    log.info("     budget %4d: %s images (%d out of domain) | lm_cos %.4f", b, f"{n:,}", excluded[es["name"]][b], m)
    with open(os.path.join(cfg.output_dir, "skipped_sets.json"), "w") as f:
        json.dump(skipped, f, indent=1)
    with stage("Bootstrap summaries", log):
        summary = summarise(rows_by_key, tokens_by_key, excluded, cfg)
        with open(os.path.join(cfg.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        write_markdown(os.path.join(cfg.output_dir, "summary.md"), summary)
    log.info("<=== E1 written to %s", cfg.output_dir)


def summarise(rows_by_key, tokens_by_key, excluded, cfg) -> dict:
    boot = dict(n_boot=cfg.bootstrap.n, seed=cfg.bootstrap.seed)
    out = {"run_name": cfg.run_name, "student": cfg.get("student_dir") or to_plain(cfg.get("student")),
           "teacher": cfg.teacher.get("model_id", "tiny"), "input_domain": to_plain(cfg.input_domain), "sets": []}
    for (name, b), rows in rows_by_key.items():
        entry = {"set": name, "group": rows[0]["group"] if rows else None, "budget": b, "num_images": len(rows),
                 "excluded_out_of_domain": excluded.get(name, {}).get(b, 0)}
        for k in METRICS:
            if rows and k in rows[0]:
                entry[k] = bootstrap_ci([r[k] for r in rows], **boot)
        toks = np.concatenate(tokens_by_key[(name, b)]) if tokens_by_key[(name, b)] else np.zeros(0)
        entry["token_vis_cos_p05"] = float(np.quantile(toks, 0.05)) if len(toks) else float("nan")
        entry["by_aspect"] = {}
        for bin_name, *_ in ASPECT_BINS:
            sub = [r for r in rows if r["aspect_bin"] == bin_name]
            if sub and "lm_cos" in sub[0]:
                entry["by_aspect"][bin_name] = {"num_images": len(sub), "lm_cos": bootstrap_ci([r["lm_cos"] for r in sub], **boot)}
        out["sets"].append(entry)
    return out


def write_markdown(path: str, summary: dict) -> None:
    lines = [f"# E1 fidelity - {summary['run_name']}", "", f"teacher `{summary['teacher']}` | student `{summary['student']}`", "",
             "| set | budget | images | excluded | LM cos [95% CI] | LM rel. err | vision cos | CKA | token cos p05 |", "|---|---:|---:|---:|---|---|---|---|---:|"]
    fmt = lambda d: f"{d['mean']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}]" if isinstance(d, dict) else "-"  # noqa: E731
    for e in summary["sets"]:
        lines.append(f"| {e['set']} | {e['budget']} | {e['num_images']:,} | {e['excluded_out_of_domain']} | {fmt(e.get('lm_cos'))} | "
                     f"{fmt(e.get('lm_rel_err'))} | {fmt(e.get('vis_cos'))} | {fmt(e.get('cka'))} | {e['token_vis_cos_p05']:.4f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
