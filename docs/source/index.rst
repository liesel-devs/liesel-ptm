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

This is a simple example for first steps:

.. code-block:: python

    import liesel_ptm as ptm
    import jax

    y = jax.random.normal(jax.random.key(0), (50,))

    model = ptm.LocScalePTM.new_ptm(y)
    results = model.run_mcmc(seed=1, warmup=300, posterior=500)
    samples = results.get_posterior_samples()

    model.plot(samples)

    dist = model.init_dist(samples) # initialize a distribution object


API Reference
-------------

.. rubric:: Model

.. autosummary::
    :toctree: generated
    :caption: API
    :recursive:
    :nosignatures:

    ~liesel_ptm.LocScalePTM
    ~liesel_ptm.TransformationDist
    ~liesel_ptm.LocScaleTransformationDist
    ~liesel_ptm.term
    ~liesel_ptm.term_ri
    ~liesel_ptm.ps
    ~liesel_ptm.lin
    ~liesel_ptm.ri
    ~liesel_ptm.PTMKnots
    ~liesel_ptm.PTMCoef
    ~liesel_ptm.ScaleWeibull
    ~liesel_ptm.ScaleInverseGamma

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
