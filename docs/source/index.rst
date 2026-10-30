.. liesel-ptm documentation master file, created by
   sphinx-quickstart on Mon Jul  3 09:59:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Penalized Transformation Models in Liesel
==========================================

Penalized Transformation Models are a class of flexible structured additive
distributional regression models. This is a Python library for estimating these models
with Markov Chain Monte Carlo (MCMC) methods.

Learn more in the paper:

Brachem, J., Wiemann, P. F. V., & Kneib, T. (2024). Bayesian penalized transformation models: Structured additive location-scale regression for arbitrary conditional distributions (No. arXiv:2404.07440). arXiv. `https://doi.org/10.48550/arXiv.2404.07440 <https://doi.org/10.48550/arXiv.2404.07440>`_

Installation
------------

The library can be installed from GitHub:

.. code:: bash

    $ pip install git+https://github.com/liesel-devs/liesel-ptm.git


Getting Started
---------------

This model has a smooth location effect, constant scale, and shared transformation
shape. Data are synthetic; construct predictors and terms with ``liesel_gam``.

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import liesel.model as lsl
    import liesel_gam as gam

    import liesel_ptm as ptm

    x = jnp.linspace(-2.0, 2.0, 100)
    y = jnp.sin(x) + jax.random.normal(jax.random.key(42), x.shape)
    builder = gam.TermBuilder(gam.DictRegistry({"x": x}))

    loc = gam.AdditivePredictor("loc")
    loc += builder.ps("x", k=10)
    scale = gam.AdditivePredictor("scale", inv_link=jnp.exp)

    knots = ptm.PTMKnots(-4.0, 4.0, nparam=6)
    shape = gam.MVAdditivePredictor.from_random_walk(
        "delta",
        ndim=knots.nparam,
        intercept=False,
    ).constrain("sumzero_coef")
    shape_builder = gam.MVTermBuilder.from_predictor(shape, term_builder=builder)
    shape.intercept = shape_builder.intercept(scale=gam.scale_wb(0.1, scale=0.05))

    response = lsl.Var.new_obs(
        y,
        dist=ptm.PTMDist(
            knots.knots,
            loc=loc,
            scale=scale,
            shape=shape,
            bspline="ptm",
            centered=True,
            scaled=True,
        ),
        name="y",
    )
    model = lsl.Model(response)

.. toctree::
    :maxdepth: 2

    term-distributions


API Reference
-------------

.. rubric:: Model

.. autosummary::
    :toctree: generated
    :caption: API
    :recursive:
    :nosignatures:

    ~liesel_ptm.PTMDist
    ~liesel_ptm.TransformationDist
    ~liesel_ptm.LocScaleTransformationDist
    ~liesel_ptm.PTMKnots
    ~liesel_ptm.LocationIntercept
    ~liesel_ptm.LogScaleIntercept
    ~liesel_ptm.LegacyPTM

.. rubric:: Distribution summaries and plots

.. autosummary::
    :toctree: generated
    :nosignatures:

    ~liesel_ptm.summarise_intercept_dist
    ~liesel_ptm.summarise_1d_smooth_dist
    ~liesel_ptm.summarise_nd_smooth_dist
    ~liesel_ptm.summarise_cluster_dist
    ~liesel_ptm.summarise_conditional_dist
    ~liesel_ptm.plot_intercept_dist
    ~liesel_ptm.plot_1d_smooth_dist
    ~liesel_ptm.plot_2d_smooth_dist
    ~liesel_ptm.plot_3d_smooth_dist
    ~liesel_ptm.plot_3d_smooth_dist_stacked
    ~liesel_ptm.plot_cluster_dist
    ~liesel_ptm.plot_conditional_dist
    ~liesel_ptm.plot_regions_dist

Acknowledgements and Funding
--------------------------------

Liesel-PTM is developed by Johannes Brachem with support from Paul Wiemann and
Thomas Kneib at the `University of Göttingen <https://www.uni-goettingen.de/en>`_.
As a specialized extension, Liesel-PTM belongs to the Liesel project.
We are
grateful to the `German Research Foundation (DFG) <https://www.dfg.de/en>`_ for funding the development
through grant 443179956.

.. image:: https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/uni-goe.svg
   :alt: University of Göttingen

.. image:: https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/funded-by-dfg.svg
   :alt: Funded by DFG


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
