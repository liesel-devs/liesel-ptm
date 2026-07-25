Term-induced distributions
==========================

The distribution summary functions evaluate one multivariate transformation-predictor
term from :mod:`liesel_gam` on the standardized response scale. Other predictor terms
are excluded. An intercept or lower-order tensor marginals are included only when
their concrete term objects are passed explicitly::

   summary = ptm.summarise_nd_smooth_dist(
       ptm.onion_dist(nparam=20),
       tensor,
       samples,
       marginals=(main_x, main_z),
       intercept=predictor.intercept,
   )

``intercept=True`` is intentionally unsupported because a term does not reliably
identify its owning predictor. The distribution constructor may instead be supplied
as a Liesel ``Dist`` or response ``Var``.

Summaries
---------

Every summary returns a tidy dataframe containing ``density``, ``cdf``,
``transformation``, and ``transformation_raw`` in the ``quantity`` column. Posterior
statistics are stored in the ``mean``, ``sd``, ``var``, ``hdi_low``, ``hdi_high``,
``q_*``, and ``sample_size`` columns.

An integer ``rgrid`` creates a grid on ``[-5, 5]``. Continuous covariates use five
values by default, while categorical covariates use every mapped level. Supplied
``newdata`` is interpreted row-wise unless ``newdata_meshgrid=True``.

Plots
-----

The high-level plotting functions accept the distribution, term, and samples directly
and return a :class:`plotnine.ggplot`. Density plots use ridge baselines; CDF and
transformation plots overlay curves. Posterior trajectories are disabled by default
and can be enabled reproducibly with ``show_n_samples=`` and ``seed=``. Quantile
ribbons are shown on ordinary distribution plots by default; ``hdi_prob=`` adds HDI
display. Both uncertainty displays are opt-in for three-input and region glyphs.
Panel grids are removed by default.

Density plots deliberately do not map color. A caller can add color and fill mappings
afterward because the original covariate columns remain in the plot data::

   plot = ptm.plot_2d_smooth_dist(
       dist,
       tensor,
       samples,
       facet_by="season",
   )
   plot + p9.aes(color="day", fill="day") + p9.scale_color_viridis_c()

The one-dimensional density ridgeline hides its y-axis by default. Pass
``show_y_axis=True`` to restore the conditioning-value labels.

Three-input tensors
-------------------

For a three-input tensor, two inputs supply paired anchor points and the third selects
the local densities. Caller order is retained in the stack, facets, and legend::

   ptm.plot_3d_smooth_dist(
       dist,
       tensor,
       samples,
       x="longitude",
       y="latitude",
       ridge_by="z",
       points={"longitude": longitude, "latitude": latitude},
       ridge_values=[0.1, 0.5, 0.9],
       layout="stack",  # or "facet"
       point_size=3.5,
       point_shape="+",
       point_color="black",
   )

Anchor points default to black crosses. Their size, shape, and color can be changed
directly with the ``point_*`` arguments.

Region maps
-----------

``plot_regions_dist`` places one density glyph per category. Polygons stored on a
native liesel_gam MRF term are discovered automatically. Polygon centroids are used
as glyph locations and can be overridden selectively::

   ptm.plot_regions_dist(
       dist,
       region_term,
       samples,
       polys=polygons,
       locations={"island": (10.2, 54.1)},
   )

When no polygon is available for a category, an explicit location is required.
Pass ``show_density_fill=False`` to draw only the density outlines. This does not
disable CI or HDI ribbons requested separately.
