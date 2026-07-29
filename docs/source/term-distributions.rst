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
display. Both uncertainty displays are opt-in for stacked three-input and region
glyphs. Panel grids are removed by default.

Grouped distribution plots map color, and uncertainty ribbons map fill, to the
conditioning variable by default. This provides a legend identifying each ridge or
curve. Numeric ridge values are ordered from low to high, so higher values have
higher vertical offsets and use the high end of the color scale. Callers can replace
the default scales directly. Reference distributions are always drawn as gray dotted
lines::

   plot = ptm.plot_2d_smooth_dist(
       dist,
       tensor,
       samples,
       facet_by="season",
   )
   plot + p9.scale_color_viridis_c() + p9.scale_fill_viridis_c()

The one-dimensional density ridgeline hides its y-axis by default. Pass
``show_y_axis=True`` to restore the conditioning-value labels. Two- and three-input
density ridges use an otherwise empty ``Density`` axis because the ridge values are
identified by the legend.

Three-input tensors
-------------------

For a three-input tensor, two inputs define a facet grid and the third defines the
density ridges within each facet. The conditioning data follows the same row-wise or
mesh-grid convention as the two-input plot::

   ptm.plot_3d_smooth_dist(
       dist,
       tensor,
       samples,
       x="longitude",
       y="latitude",
       ridge_by="z",
       newdata={
           "longitude": [10.1, 10.5, 10.9],
           "latitude": [50.1, 50.5, 50.9],
           "z": [0.1, 0.5, 0.9],
       },
       newdata_meshgrid=True,
   )

Use the stacked variant to draw spatial density glyphs at paired anchor points.
Ridge values use the same ascending order as the faceted plots::

   ptm.plot_3d_smooth_dist_stacked(
       dist,
       tensor,
       samples,
       x="longitude",
       y="latitude",
       ridge_by="z",
       points={"longitude": longitude, "latitude": latitude},
       ridge_values=[0.1, 0.5, 0.9],
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
