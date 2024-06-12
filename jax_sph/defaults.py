"""Default jax-sph configs."""

from omegaconf import DictConfig, OmegaConf


def set_defaults(cfg: DictConfig = OmegaConf.create({})) -> DictConfig:
    """Set default lagrangebench configs."""

    ### global and hardware-related configs

    # .yaml case configuration file
    cfg.config = None
    # Seed for random number generator
    cfg.seed = 123
    # Whether to disable jitting compilation
    cfg.no_jit = False
    # Which GPU to use. -1 for CPU
    cfg.gpu = 0
    # Data type. One of "float32" or "float64"
    cfg.dtype = "float64"
    # XLA memory fraction to be preallocated. The JAX default is 0.75.
    # Should be specified before importing the library.
    cfg.xla_mem_fraction = 0.75

    ### psysical case configuration
    cfg.case = OmegaConf.create({})

    # Wchich Python case file to configure.
    cfg.case.source = None
    # Simulation mode. One of "sim" (run simulation) or "rlx" (run relaxation)
    cfg.case.mode = "sim"
    # Dimension of the simulation. One of 2 or 3
    cfg.case.dim = 3
    # Average distance between particles [0.001, 0.1]
    cfg.case.dx = 0.05
    # Initial state h5 path. Overrides `r0_type`. Can be useful to restart a simulation.
    cfg.case.state0_path = None
    # Which properties to adopt from state0_path. Include all to restart a simulation.
    cfg.case.state0_keys = ["r"]
    # Position initialization type. One of "cartesian" or "relaxed". Cartesian can have
    # `r0_noise_factor` and relaxed requires a state to be present in `data_relaxed`.
    cfg.case.r0_type = "cartesian"
    # How much Gaussian noise to add to r0. ( _ * dx)
    cfg.case.r0_noise_factor = 0.0
    # Magnitude of external force field
    cfg.case.g_ext_magnitude = 0.0
    # Reference dynamic viscosity. Inversely proportional to Re.
    cfg.case.viscosity = 0.01
    # Estimate max flow velocity to calculate artificial speed of sound.
    cfg.case.u_ref = 1.0
    # Reference speed of sound factor w.r.t. u_ref.
    cfg.case.c_ref_factor = 10.0
    # Reference density
    cfg.case.rho_ref = 1.0
    # Reference temperature
    cfg.case.T_ref = 1.0
    # Reference thermal conductivity
    cfg.case.kappa_ref = 0.0
    # Reference heat capacity at constant pressure
    cfg.case.Cp_ref = 0.0
    # case-specific variables
    cfg.case.special = OmegaConf.create({})

    ### solver
    cfg.solver = OmegaConf.create({})

    # Solver name. One of "SPH" (standard SPH) or "RIE" (Riemann SPH)
    cfg.solver.name = "SPH"
    # Transport velocity inclusion factor [0,...,1]
    cfg.solver.tvf = 0.0
    # CFL condition factor
    cfg.solver.cfl = 0.25
    # Density evolution vs density summation
    cfg.solver.density_evolution = False
    # Density renormalization when density evolution
    cfg.solver.density_renormalize = False
    # Integration time step. If None, it is calculated from the CFL condition.
    cfg.solver.dt = None
    # Physical time length of simulation
    cfg.solver.t_end = 0.2
    # Parameter alpha of artificial viscosity term
    cfg.solver.artificial_alpha = 0.0
    # Whether to turn on free-slip boundary condition
    cfg.solver.free_slip = False
    # Riemann dissipation limiter parameter, -1 = off
    cfg.solver.eta_limiter = 3
    # Thermal conductivity (non-dimensional)
    cfg.solver.kappa = 0
    # Number of wall boundary particle layers
    cfg.solver.n_walls = 3
    # Whether to apply the heat conduction term
    cfg.solver.heat_conduction = False
    # Whether to apply boundaty conditions
    cfg.solver.is_bc_trick = False  # new

    ### kernel
    cfg.kernel = OmegaConf.create({})

    # Kernel name, choose one of:
    # "CSK" (cubic spline kernel)
    # "QSK" (quintic spline kernel)
    # "WC2K" (Wendland C2 kernel)
    # "WC4K" (Wendland C4 kernel)
    # "WC6K" (Wendland C4 kernel)
    # "GK" (gaussian kernel)
    # "SGK" (super gaussian kernel)
    cfg.kernel.name = "QSK"
    # Smoothing length factor
    cfg.kernel.h_factor = 1.0  # new. Should default to 1.3 WC2K and 1.0 QSK

    ### equation of state
    cfg.eos = OmegaConf.create({})

    # EoS name. One of "Tait" or "RIEMANN"
    cfg.eos.name = "Tait"
    # power in the Tait equation of state
    cfg.eos.gamma = 1.0
    # background pressure factor w.r.t. p_ref
    cfg.eos.p_bg_factor = 0.0

    ### neighbor list
    cfg.nl = OmegaConf.create({})

    # Neighbor list backend. One of "jaxmd_vmap", "jaxmd_scan", "matscipy"
    cfg.nl.backend = "jaxmd_vmap"
    # Number of partitions for neighbor list. Applies to jaxmd_scan only.
    cfg.nl.num_partitions = 1

    ### output writing
    cfg.io = OmegaConf.create({})

    # In which format to write states. A subset of ["h5", "vtk"]
    cfg.io.write_type = []
    # Every `write_every` step will be saved
    cfg.io.write_every = 1
    # Where to write and read data
    cfg.io.data_path = "./"
    # What to print to stdout. As list of possible properties.
    cfg.io.print_props = ["Ekin", "u_max"]

    return cfg


defaults = set_defaults()


def check_cfg(cfg: DictConfig) -> None:
    """Check if the configs are valid."""

    assert cfg.config is not None, "A configuration file must be specified."

    assert cfg.case.source is not None, "A Python case file must be specified."
    assert cfg.case.mode in ["sim", "rlx"], "Mode must be one of 'sim' or 'rlx'."
    assert cfg.case.dim in [2, 3], "Dimension must be 2 or 3."
    assert cfg.case.dx >= 0.0, "dx must be > 0.0."
    assert cfg.case.viscosity >= 0.0, "viscosity must be >= 0.0."
    _r0_types = ["cartesian", "relaxed"]
    assert cfg.case.r0_type in _r0_types, f"r0_type must be in {_r0_types}."
    assert cfg.case.r0_noise_factor >= 0.0, "r0_noise_factor must be >= 0.0."
    assert cfg.case.g_ext_magnitude is not None, "g_ext_magnitude must be specified."
    assert cfg.case.u_ref is not None, "u_ref must be specified."

    assert cfg.solver.name in ["SPH", "RIE"], "Solver must be one of 'SPH' or 'RIE'."
    assert 0.0 <= cfg.solver.tvf <= 1.0, "tvf must be in [0.0, 1.0]."
    assert cfg.solver.dt is None or cfg.solver.dt > 0.0, "dt must be > 0.0."
    assert cfg.solver.t_end > 0.0, "t_end must be > 0.0."
    assert cfg.solver.artificial_alpha >= 0.0, "artificial_alpha must be >= 0.0."
    assert (
        cfg.solver.eta_limiter == -1 or cfg.solver.eta_limiter >= 0
    ), "eta_limiter must be -1 or >0."
    assert cfg.solver.kappa >= 0.0, "kappa must be >= 0.0."

    assert cfg.kernel.name in ["QSK", "WC2K"], "Kernel must be one of 'QSK' or 'WC2K'."
    assert cfg.kernel.h_factor > 0.0, "h_factor must be > 0.0."

    _eos_name = ["Tait", "RIEMANN"]
    assert cfg.eos.name in _eos_name, f"EoS must be one of {_eos_name}."
    assert cfg.eos.gamma > 0.0, "gamma must be > 0.0."
    assert cfg.eos.p_bg_factor >= 0.0, "p_bg_factor must be >= 0.0."

    _nl_backends = ["jaxmd_vmap", "jaxmd_scan", "matscipy"]
    assert cfg.nl.backend in _nl_backends, f"nl_backend must be in {_nl_backends}."

    _io_trite_type = ["h5", "vtk"]
    assert all(
        [w in _io_trite_type for w in cfg.io.write_type]
    ), f"write_type must be in {_io_trite_type}."
    assert cfg.io.write_every > 0, "write_every must be > 0."
