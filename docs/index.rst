.. JAX-SPH documentation master file, created by
   sphinx-quickstart on Sat Apr 12 20:55:01 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

JAX-SPH
========

.. image:: https://s9.gifyu.com/images/SUwUD.gif
   :alt: GIF


What is ``JAX-SPH``?
--------------------

JAX-SPH `(Toshev et al., 2024) <https://arxiv.org/abs/2403.04750>`_ is a Smoothed Particle Hydrodynamics (SPH) code written in `JAX <https://jax.readthedocs.io/>`_. JAX-SPH is designed to be simple, fast, and compatible with deep learning workflows. We currently support the following SPH routines:

* Standard SPH `(Adami et al., 2012) <https://www.sciencedirect.com/science/article/pii/S002199911200229X>`_
* Transport velocity SPH `(Adami et al., 2013) <https://www.sciencedirect.com/science/article/pii/S002199911300096X>`_
* Riemann SPH `(Zhang et al., 2017) <https://www.sciencedirect.com/science/article/abs/pii/S0021999117300438>`_

Check out our `GitHub repository <https://github.com/tumaer/jax-sph>`_ for more information including installation instructions and tutorial notebooks.

.. toctree::
   :maxdepth: 2
   :caption: API

   pages/defaults
   pages/case_setup
   pages/solver
   pages/simulate
   pages/utils