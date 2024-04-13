"""Parsing input arguments."""

import argparse


class Args:
    """Parsing input arguments."""

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.add_args()
        self.args = self.parser.parse_args()

    def add_args(self) -> None:
        self.parser.add_argument(
            "--case",
            type=str,
            default="TGV",
            help="Simulation setup",
        )
        self.parser.add_argument(
            "--mode",
            type=str,
            default="sim",
            choices=["sim", "rlx"],
            help="run a simulation or relax an initial state",
        )
        self.parser.add_argument(
            "--solver",
            type=str,
            default="SPH",
            choices=["SPH", "RIE"],
            help="SPH means standard SPH, and RIE the Riemann SPH algorithm",
        )
        self.parser.add_argument(
            "--kernel",
            type=str,
            default="QSK",
            choices=["QSK", "WC2K"],
            help="choose kernel function",
        )
        self.parser.add_argument(
            "--tvf",
            type=float,
            default=0.0,
            help="Transport velocity inclusion factor [0,...,1",
        )
        self.parser.add_argument(
            "--density-evolution",
            action="store_true",
            help="Density evolution vs density summation",
        )
        self.parser.add_argument(
            "--density-renormalize",
            action="store_true",
            help="Density renormalization when density evolution",
        )
        self.parser.add_argument(
            "--dim", type=int, choices=[2, 3], default=3, help="Dimension"
        )
        self.parser.add_argument(
            "--dx",
            type=float,
            default=0.05,
            help="Average distance between particles [0.001, 0.1]",
        )
        self.parser.add_argument(
            "--dt",
            type=float,
            default=None,
            help="If the user wants to specify the time step explicitly",
        )
        self.parser.add_argument(
            "--t-end",
            type=float,
            default=0.2,
            help="Physical time length of simulation",
        )
        self.parser.add_argument(
            "--viscosity",
            type=float,
            default=0.01,
            help="Dynamic viscosity. Inversely proportional to Re",
        )
        self.parser.add_argument(
            "--state0-path",
            type=str,
            default="none",
            help="Initalize the state from a .h5 file. Overrides `--r0-type`."
            "Can be useful to restart a simulation",
        )
        self.parser.add_argument(
            "--r0-type",
            type=str,
            default="cartesian",
            choices=["cartesian", "relaxed"],
            help="Position initialization type. Cartesian can have `--r0-noise-factor`"
            "Relaxed requires a relaxed state to be present in `data_relaxed`",
        )
        self.parser.add_argument(
            "--r0-noise-factor",
            type=float,
            default=0.0,
            help="How much Gaussian noise to add to r0. ( _ * dx)",
        )
        self.parser.add_argument(
            "--p-bg-factor",
            type=float,
            default=None,
            help="Background pressure factor w.r.t. p_ref.",
        )
        self.parser.add_argument(
            "--g-ext-magnitude",
            type=float,
            default=None,
            help="Magnitude of external force field",
        )
        self.parser.add_argument(
            "--u_ref",
            type=float,
            default=None,
            help="Estimatet max flow velocity to calculate artificial speed of sound",
        )
        self.parser.add_argument(
            "--artificial-alpha",
            type=float,
            default=0.0,
            help="Parameter alpha of artificial viscosity term.",
        )
        self.parser.add_argument(
            "--free-slip",
            action="store_true",
            help="Whether to turn on free-slip boundary condition",
        )
        self.parser.add_argument(
            "--nl-backend",
            default="jaxmd_vmap",
            choices=["jaxmd_vmap", "jaxmd_scan", "matscipy"],
            help="Which backend to use for neighbor list",
        )
        self.parser.add_argument("--num-partitions", type=int, default=1)
        self.parser.add_argument(
            "--seed", type=int, default=123, help="Seed for random number generator"
        )
        self.parser.add_argument(
            "--no-jit", action="store_true", help="Disable jitting compilation"
        )
        self.parser.add_argument(
            "--write-h5", "-h5", action="store_true", help="Whether to write .h5 files."
        )
        self.parser.add_argument(
            "--write-vtk",
            "-vtk",
            action="store_true",
            help="Whether to output .vtk files",
        )
        self.parser.add_argument(
            "--write-every",
            type=int,
            default=1,
            help="Every `write_every` step will be saved",
        )
        self.parser.add_argument(
            "--data-path", type=str, default="./", help="Where to write and read data"
        )
        self.parser.add_argument(
            "--gpu", type=int, default=0, help="Which GPU to use. -1 for CPU"
        )
        self.parser.add_argument(
            "--no-f64", action="store_true", help="Whether to use 64 bit precision"
        )
        self.parser.add_argument(
            "--eta-limiter",
            type=float,
            default=3,
            help="Riemann dissipation limiter parameter, -1 = off",
        )
        self.parser.add_argument(
            "--kappa",
            type=float,
            default=0,
            help="Thermal conductivity (non-dimensional)",
        )
        self.parser.add_argument(
            "--heat-conduction",
            action="store_true",
            help="Whether to apply the heat conduction term",
        )
        self.parser.add_argument(
            "--outlet-temperature-derivative",
            action="store_true",
            help="Whether to set dT/dt near the outlet to 0",
        )
