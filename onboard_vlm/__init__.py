"""onboard-vlm: distilling Gemma 4 vision encoders into AR-class students."""

__version__ = "0.1.0"


def _register_resolvers() -> None:
    """OmegaConf resolvers used by the configs (registered when the package is imported).

    Skipped when omegaconf cannot be imported: the lmms-eval environment only loads exported students (no Hydra
    configs), and lmms-eval 0.7.4 installs an antlr4 runtime that omegaconf 2.3 cannot load."""
    try:
        from omegaconf import OmegaConf
    except Exception:  # noqa: BLE001  (ImportError, or antlr4's version check raising a bare Exception)
        return

    def seed_tag(seed, init_seed, data_seed) -> str:
        """``s0`` when every source of randomness uses ``seed``, else ``s0_i<init>_d<data>`` (run ids never collide)."""
        s, i, d = int(seed), int(init_seed), int(data_seed)
        return f"s{s}" if i == s and d == s else f"s{s}_i{i}_d{d}"

    OmegaConf.register_new_resolver("seed_tag", seed_tag, replace=True)
    OmegaConf.register_new_resolver("join", lambda xs, sep="-": sep.join(str(x) for x in xs), replace=True)


_register_resolvers()
