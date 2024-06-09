Defaults
===================================

The defaults are defined through a function ``jax_sph.defaults.set_defaults()``, which
takes a potentially empty ``omegaconf.DictConfig`` object and creates or overwrites the
default values. One can also directly call ``from jax_sph.defaults import defaults``,
with ``defaults=set_defaults()``, to get the default DictConfig, which we unpack below.

.. exec_code::
    :hide_code:
    :linenos_output:
    :language_output: python
    :caption: JAX-SPH default values


    with open("jax_sph/defaults.py", "r") as file:
        defaults_full = file.read()

    # parse defaults: remove imports, only keep the set_defaults function

    defaults_full = defaults_full.split("\n")

    # remove imports
    defaults_full = [line for line in defaults_full if not line.startswith("import")]
    defaults_full = [line for line in defaults_full if len(line.replace(" ", "")) > 0]

    # remove other functions
    keep = False
    defaults = []
    for i, line in enumerate(defaults_full):
        if line.startswith("def"):
            if "set_defaults" in line:
                keep = True
            else:
                keep = False
        
        if keep:
            defaults.append(line)

    # remove function declaration and return
    defaults = defaults[2:-2]

    # remove indent
    defaults = [line[4:] for line in defaults]


    print("\n".join(defaults))
        