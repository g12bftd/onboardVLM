"""Config helpers."""

from __future__ import annotations

from typing import Any


def to_plain(x: Any) -> Any:
    """OmegaConf containers -> plain (resolved) python containers; anything else unchanged."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(x, (DictConfig, ListConfig)):
            return OmegaConf.to_container(x, resolve=True)
    except ImportError:  # pragma: no cover
        pass
    return x
