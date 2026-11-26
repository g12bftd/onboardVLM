"""Locate a conda environment's Python by name, the way ``conda activate <name>`` would (``envs_dirs`` order), so
environments can live anywhere (e.g. on scratch, outside ``$CONDA_BASE/envs``)."""

from __future__ import annotations

import os
import shutil
import subprocess


def conda_exe() -> str | None:
    if os.environ.get("CONDA_EXE"):
        return os.environ["CONDA_EXE"]
    if os.environ.get("CONDA_BASE") and os.path.isfile(os.path.join(os.environ["CONDA_BASE"], "bin", "conda")):
        return os.path.join(os.environ["CONDA_BASE"], "bin", "conda")
    return shutil.which("conda")


def conda_env_python(name: str, timeout: float = 120.0) -> str | None:
    """``<prefix>/bin/python`` of the conda env ``name`` (None if conda or the env is not there)."""
    exe = conda_exe()
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "run", "-n", name, "python", "-c", "import sys; print(sys.executable)"],
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return lines[-1] if r.returncode == 0 and lines and os.path.isfile(lines[-1]) else None
