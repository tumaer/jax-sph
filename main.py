"""Main file for starting simulations"""

import os

from jax_sph.args import Args

if __name__ == "__main__":
    args = Args().args

    # specify cuda device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

    if not args.no_f64:
        from jax import config

        config.update("jax_enable_x64", True)

    from jax_sph.simulate import simulate

    simulate(args)
