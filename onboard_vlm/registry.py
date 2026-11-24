"""Name -> class registries for teachers, students, losses and datasets.

Configs refer to components by ``name`` and pass the remaining keys as constructor kwargs::

    student:
      name: compact_vit
      embed_dim: 384
      depth: 12
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items and self._items[name] is not obj:
                raise KeyError(f"{self.kind} '{name}' registered twice")
            self._items[name] = obj  # type: ignore[assignment]
            return obj

        return deco

    def get(self, name: str) -> Callable[..., Any]:
        _import_builtin_components()
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name]

    def build(self, cfg: Mapping[str, Any] | None = None, /, **overrides: Any) -> Any:
        """Instantiate from ``{"name": ..., **kwargs}`` (OmegaConf configs are converted first)."""
        cfg = _to_plain_dict(cfg or {})
        cfg.update(overrides)
        name = cfg.pop("name")
        return self.get(name)(**cfg)

    def names(self) -> list[str]:
        _import_builtin_components()
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        _import_builtin_components()
        return name in self._items


def _to_plain_dict(cfg: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except ImportError:  # pragma: no cover
        pass
    return dict(cfg)


_IMPORTED = False


def _import_builtin_components() -> None:
    """Import modules whose decorators populate the registries (lazy to keep import cheap)."""
    global _IMPORTED
    if _IMPORTED:
        return
    _IMPORTED = True
    import onboard_vlm.data.datasets
    import onboard_vlm.distill.losses
    import onboard_vlm.students
    import onboard_vlm.teachers  # noqa: F401


TEACHERS = Registry("teacher")
STUDENTS = Registry("student")
LOSSES = Registry("loss")
DATASETS = Registry("dataset")

register_teacher = TEACHERS.register
register_student = STUDENTS.register
register_loss = LOSSES.register
register_dataset = DATASETS.register

build_teacher = TEACHERS.build
build_student = STUDENTS.build
build_loss = LOSSES.build
build_dataset = DATASETS.build
