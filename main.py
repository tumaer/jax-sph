import os

from jax import config
from omegaconf import DictConfig, OmegaConf

from jax_sph.defaults import defaults


def check_subset(superset, subset, full_key=""):
    """Check that the keys of 'subset' are a subset of 'superset'."""
    for k, v in subset.items():
        key = full_key + k
        if isinstance(v, dict):
            check_subset(superset[k], v, key + ".")
        else:
            msg = f"cli_args must be a subset of the defaults. Wrong cli key: '{key}'"
            assert k in superset, msg


def load_embedded_configs(cli_args: DictConfig) -> DictConfig:
    """Loads all 'extends' embedded configs and merge them with the cli overwrites."""

    cfgs = [OmegaConf.load(cli_args.config)]
    while "extends" in cfgs[0]:
        extends_path = cfgs[0]["extends"]
        del cfgs[0]["extends"]

        # go to parents configs until the defaults are reached
        if extends_path != "JAX_SPH_DEFAULTS":
            cfgs = [OmegaConf.load(extends_path)] + cfgs
        else:
            cfgs = [defaults] + cfgs

            # Assert that the cli_args and all inherited config files are a subset of
            # the defaults if inheritance from defaults is used.
            # Exclude case.special from this check as it is case-specific.
            for cfg in cfgs[1:] + [cli_args]:
                cfg = cfg.copy()
                if "case" in cfg and "special" in cfg.case:
                    del cfg.case.special
                check_subset(defaults, cfg)

            break

    # merge all embedded configs and give highest priority to cli_args
    cfg = OmegaConf.merge(*cfgs, cli_args)
    return cfg


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    assert "config" in cli_args, "A configuration file must be specified."

    cfg = load_embedded_configs(cli_args)

    # Specify cuda device. These setting must be done before importing jax-md.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 from TensorFlow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cfg.xla_mem_fraction)
    # for reproducibility
    os.environ[
        "XLA_FLAGS"
    ] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    if cfg.no_jit:
        config.update("jax_disable_jit", True)
    if cfg.dtype == "float64":
        config.update("jax_enable_x64", True)

    from jax_sph.simulate import simulate

    simulate(cfg)
