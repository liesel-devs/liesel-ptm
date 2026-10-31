.. liesel-ptm documentation master file, created by
   sphinx-quickstart on Mon Jul  3 09:59:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Penalized Transformation Models in Liesel
==========================================

Penalized Transformation Models are a class of flexible structured additive
distributional regression models. This is a Python library for estimating these models
with optimization, variational inference, and Markov Chain Monte Carlo (MCMC).
Liesel provides inference and model graphs; liesel-gam provides predictors, terms,
priors, and Gaussian IWLS proposals. Liesel-PTM provides transformations,
distributions, computed intercepts, and distribution summaries and plots.

Learn more in the paper:

Brachem, J., Wiemann, P. F. V., & Kneib, T. (2026).
`Bayesian Penalized Transformation Models: Structured Additive Location-Scale Regression for Arbitrary Conditional Distributions <https://doi.org/10.1080/10618600.2026.2703269>`_.
*Journal of Computational and Graphical Statistics*.

Installation
------------

Requires Python 3.13. Until Liesel releases the required optimization/VI APIs,
install its development branch alongside this library:

.. code:: bash

    pip install "liesel @ git+https://github.com/liesel-devs/liesel.git@vi-modernization" \
      "git+https://github.com/liesel-devs/liesel-ptm.git@codex/cleanup"


Getting Started
---------------

This model has a smooth location effect, constant scale, and shared transformation
shape. Data are synthetic; construct predictors and terms with ``liesel_gam``.

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import liesel_gam as gam

    import liesel_ptm as ptm

    x = jnp.linspace(-2.0, 2.0, 100)
    y = jnp.sin(x) + jax.random.normal(jax.random.key(42), x.shape)
    builder = gam.TermBuilder(gam.DictRegistry({"x": x}))

    model = ptm.PTM(y, nshape=6, bounds=(-4.0, 4.0))
    model.pred.loc += builder.ps("x", k=10)

:class:`~liesel_ptm.PTM` is a Liesel model with native liesel-gam predictors in
:attr:`~liesel_ptm.PTM.pred`. Add terms through ``.pred.loc``, ``.pred.scale`` and
``.pred.shape`` before creating inference objects. Location and scale represent the
conditional mean and standard deviation. ``nshape`` counts shape coefficients before
the sum-to-zero constraint; ``bounds`` defaults to ``(-4, 4)`` on the standardized
residual scale. The shape predictor includes a zero-initialized random-walk intercept.
Pass ``shape_scale`` a native liesel-gam scale variable or a fixed float to configure
its prior. Supplied ``loc``, ``scale`` and ``shape`` predictors are also accepted.

For staged optimization, VI, covariate-dependent shape and prediction, see the
`modern reference example <https://github.com/liesel-devs/liesel-ptm/blob/codex/cleanup/demo/reference_modern.py>`_.
For manual construction, :class:`~liesel_ptm.PTMDist` remains available.


Legacy PTM
----------

The original PTM uses shared shape and computed location/scale intercepts.
:class:`~liesel_ptm.LegacyPTM` applies those intercepts separately from ``.pred.loc``
and ``.pred.scale``. The latter returns relative positive scale; use its
``linear_predictor`` for log-scale effects. The
`complete legacy example <https://github.com/liesel-devs/liesel-ptm/blob/codex/cleanup/demo/reference_legacy.py>`_
includes a smoothing-variance Gibbs update. Its Weibull variance prior is unclipped.

Configure Gaussian IWLS proposals for location/scale and joint shape NUTS with
:meth:`~liesel_ptm.LegacyPTM.setup_mcmc`::

    legacy = ptm.LegacyPTM(y, nshape=6)
    legacy_terms = gam.TermBuilder.from_dict({"x": x})
    legacy.pred.loc += legacy_terms.slin("x", scale=1.0, name="m.x")
    legacy.pred.scale += legacy_terms.slin("x", scale=0.5, name="s.x")
    engine_builder = legacy.setup_mcmc(seed=43, num_chains=2)
    engine_builder.add_adaptation(500)
    engine_builder.add_posterior(1000)
    engine = engine_builder.build()
    engine.sample_all_epochs()
    samples = engine.get_results().get_posterior_samples()
    prediction = legacy.predict(
        samples,
        predict=["loc", "scale", "delta"],
        newdata={"x": jnp.array([-0.5, 0.5])},
    )

Both computed intercepts are recorded for every posterior draw.
:meth:`~liesel_ptm.LegacyPTM.predict` uses those records for new covariates without
changing the fitted graph. Use longer chains and inspect diagnostics for inference.

Migrating the legacy constructor: ``LegacyPTM(y, nshape=...)`` replaces the earlier
``LegacyPTM(model)`` wrapper. Add terms through ``.pred``; ``setup_mcmc`` now obtains
the predictors directly from the model.

Distribution effects
--------------------

The `plotting notebook <https://github.com/liesel-devs/liesel-ptm/blob/codex/cleanup/demo/term-distribution-plots.ipynb>`_
illustrates smooth, categorical, tensor and spatial effects with simulated coefficient
draws. See the guide for summary and plotting options:

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

    ~liesel_ptm.PTM
    ~liesel_ptm.PTMPredictors
    ~liesel_ptm.PTMDist
    ~liesel_ptm.TransformationDist
    ~liesel_ptm.LocScaleTransformationDist
    ~liesel_ptm.PTMKnots
    ~liesel_ptm.OnionKnots
    ~liesel_ptm.OnionSpline
    ~liesel_ptm.onion_dist
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

.. rubric:: Censoring and survival

.. autosummary::
    :toctree: generated
    :nosignatures:

    ~liesel_ptm.survival_dist.CensoredDistribution
    ~liesel_ptm.survival_dist.LeftCensoredDistribution
    ~liesel_ptm.survival_dist.RightCensoredDistribution
    ~liesel_ptm.survival_dist.IntervalCensoredDistribution
    ~liesel_ptm.survival_dist.uncensored
    ~liesel_ptm.survival_dist.left_censored
    ~liesel_ptm.survival_dist.right_censored
    ~liesel_ptm.survival_dist.interval_censored
    ~liesel_ptm.survival_dist.setup_censored_vars
    ~liesel_ptm.survival_dist.subset_var

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
