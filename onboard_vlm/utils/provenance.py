"""Run provenance: everything needed to tell exactly what produced a result, written next to it.

``write_provenance(directory, cfg)`` writes:

* ``provenance.json``: git commit / branch / dirty flag, command line, host, SLURM job, Python and package
  versions (torch, CUDA, cuDNN, transformers, Lightning, timm), GPU names, the seeds, the teacher and its
  pinned revision, the student config, and the fingerprints of the data the run read (corpus manifest
  digest and ``preliminary`` flag, blocklist hash, benchmark LMDB build markers);
* ``requirements.freeze.txt``: ``pip freeze`` of the environment;
* ``uncommitted.patch``: ``git diff HEAD`` when the working tree is dirty (results from uncommitted code stay
  reproducible).

Called by every entry point (training and E1 / E2 / E3), on rank 0 only.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except Exception:
        return None


def _versions() -> dict[str, Any]:
    out: dict[str, Any] = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "pytorch_lightning", "timm", "numpy", "hydra", "lmdb", "safetensors"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch

        out["cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    except Exception:
        pass
    return out


def _file_sha256(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_fingerprints(cfg: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    corpus = cfg.get("corpus_root") or os.environ.get("ONBOARD_VLM_CORPUS")
    if corpus:
        m = os.path.join(corpus, "corpus_manifest.json")
        if os.path.isfile(m):
            with open(m) as f:
                man = json.load(f)
            out["corpus"] = {"root": corpus, "kept_sha1_digest": man.get("kept_sha1_digest"), "total_kept": man.get("total_kept"),
                             "preliminary": man.get("preliminary"), "missing_sources": man.get("missing_sources"), "created": man.get("created")}
        out["blocklist_sha256"] = _file_sha256(os.path.join(corpus, "eval_blocklist.json"))
    lmdb = cfg.get("data_root") or os.environ.get("ONBOARD_VLM_DATA")
    if lmdb and os.path.isdir(lmdb):
        out["benchmarks_built"] = sorted(d for d in os.listdir(lmdb) if os.path.isfile(os.path.join(lmdb, d, ".done")))
    return out


def is_rank_zero() -> bool:
    return all(os.environ.get(k, "0") == "0" for k in ("RANK", "SLURM_PROCID", "LOCAL_RANK"))


def write_provenance(directory: str, cfg: Any = None) -> str | None:
    """Write the provenance files into ``directory`` (rank 0 only); returns the JSON path."""
    if not is_rank_zero():
        return None
    from omegaconf import DictConfig, OmegaConf

    c = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg or {})
    os.makedirs(directory, exist_ok=True)
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    prov = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "command": " ".join([sys.executable, *sys.argv]),
        "host": socket.gethostname(),
        "slurm": {k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NNODES", "SLURM_NTASKS", "SLURM_JOB_PARTITION") if k in os.environ},
        "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "dirty": dirty},
        "versions": _versions(),
        "seeds": {k: c.get(k) for k in ("seed", "init_seed", "data_seed") if k in c},
        "teacher": {k: (c.get("teacher") or {}).get(k) for k in ("model_id", "revision", "key")} if isinstance(c.get("teacher"), dict) else None,
        "student": c.get("student"),
        "run_id": c.get("run_id") or c.get("run_name"),
        "data": _data_fingerprints(c),
    }
    path = os.path.join(directory, "provenance.json")
    with open(path, "w") as f:
        json.dump(prov, f, indent=1, default=str)
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
        with open(os.path.join(directory, "requirements.freeze.txt"), "w") as f:
            f.write(freeze)
    except Exception:
        pass
    if dirty:
        diff = _git("diff", "HEAD")
        if diff:
            with open(os.path.join(directory, "uncommitted.patch"), "w") as f:
                f.write(diff + "\n")
    return path
