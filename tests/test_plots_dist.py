from typing import Any

import jax.numpy as jnp
import liesel.model as lsl
import liesel_gam as gam
import numpy as np
import pandas as pd
import plotnine as p9
import pytest

import liesel_ptm as ptm


def _assert_gray_reference_lines(plot: p9.ggplot) -> None:
    reference_layers = [
        layer
        for layer in plot.layers
        if isinstance(layer.geom, p9.geom_line)
        and layer.geom.aes_params.get("linetype") == "dotted"
    ]
    assert reference_layers
    assert all(layer.geom.aes_params["color"] == "gray" for layer in reference_layers)


def _basis(x):
    x = jnp.squeeze(x, axis=-1)
    return jnp.column_stack((jnp.ones_like(x), x))


def _smooth():
    builder = gam.MVTermBuilder.from_df(
        pd.DataFrame({"x": jnp.linspace(0.0, 1.0, 6)}), jnp.eye(4)
    )
    term = builder.f(
        "x",
        basis_fn=_basis,
        penalty=jnp.eye(2),
        scale=1.0,
        dimension_scale=1.0,
        use_callback=False,
    )
    model = lsl.Model([term])
    return term, model


def _tensor2():
    data = pd.DataFrame(
        {"x": jnp.linspace(0.0, 1.0, 6), "z": jnp.linspace(1.0, 2.0, 6)}
    )
    scalar_builder = gam.TermBuilder.from_df(data)
    builder = gam.MVTermBuilder.from_term_builder(scalar_builder, jnp.eye(4))
    sx = scalar_builder.f(
        "x",
        basis_fn=_basis,
        penalty=jnp.eye(2),
        scale=1.0,
        use_callback=False,
    )
    sz = scalar_builder.f(
        "z",
        basis_fn=_basis,
        penalty=jnp.eye(2),
        scale=1.0,
        use_callback=False,
    )
    term = builder.tx(sx, sz, dimension_scale=1.0)
    model = lsl.Model([term])
    return term, model


def _tensor3():
    data = pd.DataFrame(
        {
            "longitude": jnp.linspace(10.0, 11.0, 6),
            "latitude": jnp.linspace(50.0, 51.0, 6),
            "z": jnp.linspace(0.0, 1.0, 6),
        }
    )
    scalar_builder = gam.TermBuilder.from_df(data)
    builder = gam.MVTermBuilder.from_term_builder(scalar_builder, jnp.eye(4))

    def marginal(name):
        return scalar_builder.f(
            name,
            basis_fn=_basis,
            penalty=jnp.eye(2),
            scale=1.0,
            use_callback=False,
        )

    term = builder.tx(
        marginal("longitude"),
        marginal("latitude"),
        marginal("z"),
        dimension_scale=1.0,
    )
    model = lsl.Model([term])
    return term, model


def _cluster():
    groups = pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"])
    builder = gam.MVTermBuilder.from_df(pd.DataFrame({"group": groups}), jnp.eye(4))
    term = builder.ri("group", scale=1.0, dimension_scale=1.0)
    model = lsl.Model([term])
    return term, model


def test_plot_intercept_dist_renders_mean_band_and_reference() -> None:
    builder = gam.MVTermBuilder.from_df(pd.DataFrame({"x": [0.0, 1.0]}), jnp.eye(4))
    term = builder.intercept(scale=1.0)
    model = lsl.Model([term])
    assert term.model is model
    samples = {term.coef.name: jnp.zeros((2, 4))}

    plot = ptm.plot_intercept_dist(ptm.onion_dist(nparam=4), term, samples, rgrid=11)
    figure = plot.draw()

    assert isinstance(plot, p9.ggplot)
    assert len(figure.axes) == 1
    assert len(figure.axes[0].lines) >= 2
    _assert_gray_reference_lines(plot)
    assert not any(
        line.get_visible()
        for line in figure.axes[0].get_xgridlines() + figure.axes[0].get_ygridlines()
    )


def test_plot_intercept_dist_adds_sampled_trajectories_only_on_request() -> None:
    builder = gam.MVTermBuilder.from_df(pd.DataFrame({"x": [0.0, 1.0]}), jnp.eye(4))
    term = builder.intercept(scale=1.0)
    model = lsl.Model([term])
    assert term.model is model
    samples = {
        term.coef.name: jnp.array(
            [[0.0, 0.0, 0.0, 0.0], [0.2, -0.1, 0.1, 0.3], [-0.2, 0.1, 0.0, 0.2]]
        )
    }

    default = ptm.plot_intercept_dist(
        ptm.onion_dist(nparam=4), term, samples, rgrid=11
    ).draw()
    sampled = ptm.plot_intercept_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=11,
        show_n_samples=2,
        seed=2,
    ).draw()

    assert len(sampled.axes[0].lines) > len(default.axes[0].lines)


def test_plot_1d_smooth_dist_uses_covariate_ridge_baselines() -> None:
    term, model = _smooth()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    plot = ptm.plot_1d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=11,
        newdata={"x": np.asarray([1.0, 0.5, 0.0])},
    )
    figure = plot.draw()
    shown_axis = ptm.plot_1d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=11,
        newdata={"x": np.asarray([1.0, 0.5, 0.0])},
        show_y_axis=True,
    ).draw()

    assert isinstance(plot, p9.ggplot)
    assert isinstance(plot.data, pd.DataFrame)
    assert not figure.axes[0].get_yticklabels()
    assert len(shown_axis.axes[0].get_yticklabels()) == 3
    assert plot.mapping["color"] == "x"
    assert plot.labels.color == "x"
    _assert_gray_reference_lines(plot)
    baselines = plot.data.groupby("x")["baseline"].first().sort_index()
    assert np.all(np.diff(baselines) > 0)
    assert len(plot.layers) >= 4


def test_plot_1d_smooth_dist_supports_opt_in_trajectories() -> None:
    term, model = _smooth()
    assert term.model is model
    samples = {
        term.coef.name: jnp.arange(3 * term.coef.value.size, dtype=float).reshape(3, -1)
        / 50.0
    }

    default = ptm.plot_1d_smooth_dist(
        ptm.onion_dist(nparam=4), term, samples, rgrid=9, ngrid=2
    ).draw()
    sampled = ptm.plot_1d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=9,
        ngrid=2,
        show_n_samples=2,
    ).draw()

    assert len(sampled.axes[0].lines) > len(default.axes[0].lines)


def test_density_ridges_add_opt_in_hdi() -> None:
    term, model = _smooth()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    default = ptm.plot_1d_smooth_dist(
        ptm.onion_dist(nparam=4), term, samples, rgrid=9, ngrid=2
    )
    hdi = ptm.plot_1d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=9,
        ngrid=2,
        hdi_prob=0.8,
    )

    assert len(hdi.layers) > len(default.layers)
    assert len(hdi.draw().axes) == 1


def test_plot_2d_smooth_dist_colors_ridges_and_hides_y_tick_labels() -> None:
    term, model = _tensor2()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    plot = ptm.plot_2d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=7,
        newdata={
            "x": np.asarray([1.0, 0.5, 0.0]),
            "z": np.asarray([2.0, 1.5, 1.0]),
        },
        newdata_meshgrid=True,
    )
    figure = plot.draw()

    assert isinstance(plot, p9.ggplot)
    assert len(figure.axes) == 3
    assert plot.mapping["color"] == "x"
    assert plot.labels.color == "x"
    assert plot.labels.y == "Density"
    _assert_gray_reference_lines(plot)
    assert all(not axis.get_yticklabels() for axis in figure.axes)
    assert isinstance(plot.data, pd.DataFrame)
    baselines = plot.data.groupby("x")["baseline"].first().sort_index()
    assert np.all(np.diff(baselines) > 0)


@pytest.mark.parametrize(
    ("quantity", "label"),
    [
        ("cdf", "CDF"),
        ("transformation", "Transformation"),
        ("transformation_raw", "Raw transformation"),
    ],
)
def test_plot_2d_smooth_dist_labels_non_density_quantities(
    quantity: str, label: str
) -> None:
    term, model = _tensor2()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    plot = ptm.plot_2d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        quantity=quantity,
        rgrid=7,
        ngrid=2,
    )

    assert plot.labels.y == label
    _assert_gray_reference_lines(plot)


def test_plot_2d_smooth_dist_supports_opt_in_trajectories() -> None:
    term, model = _tensor2()
    assert term.model is model
    samples = {
        term.coef.name: jnp.arange(3 * term.coef.value.size, dtype=float).reshape(3, -1)
        / 100.0
    }

    default = ptm.plot_2d_smooth_dist(
        ptm.onion_dist(nparam=4), term, samples, rgrid=7, ngrid=2
    ).draw()
    sampled = ptm.plot_2d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=7,
        ngrid=2,
        show_n_samples=2,
    ).draw()

    assert sum(len(axis.lines) for axis in sampled.axes) > sum(
        len(axis.lines) for axis in default.axes
    )


def test_plot_3d_smooth_dist_stacked_orders_ridges_and_marks_points() -> None:
    term, model = _tensor3()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    plot = ptm.plot_3d_smooth_dist_stacked(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        x="longitude",
        y="latitude",
        ridge_by="z",
        points={
            "longitude": np.asarray([10.2, 10.8]),
            "latitude": np.asarray([50.3, 50.7]),
        },
        ridge_values=[0.8, 0.2],
        rgrid=9,
        point_size=4.0,
        point_shape="o",
        point_color="red",
    )
    figure = plot.draw()

    assert isinstance(plot, p9.ggplot)
    assert isinstance(plot.data, pd.DataFrame)
    assert pd.api.types.is_numeric_dtype(plot.data["z"])
    baselines = (
        plot.data.groupby(["longitude", "latitude", "z"])["baseline"]
        .first()
        .unstack("z")
    )
    assert np.all(baselines[0.8] > baselines[0.2])
    assert plot.mapping["color"] == "z"
    assert plot.layers[-1].geom.aes_params == {
        "color": "red",
        "shape": "o",
        "size": 4.0,
    }
    assert len(figure.axes) == 1
    assert len(figure.axes[0].collections) >= 1


def test_plot_3d_smooth_dist_uses_automatic_facet_grid() -> None:
    term, model = _tensor3()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}
    generic_term: lsl.Var = term
    generic_marginals: list[lsl.Var] = []

    plot = ptm.plot_3d_smooth_dist(
        ptm.onion_dist(nparam=4),
        generic_term,
        samples,
        x="longitude",
        y="latitude",
        ridge_by="z",
        rgrid=7,
        ngrid=2,
        marginals=generic_marginals,
    )

    assert isinstance(plot.facet, p9.facet_grid)
    assert plot.facet.rows == ["longitude"]
    assert plot.facet.cols == ["latitude"]
    assert isinstance(plot.data, pd.DataFrame)
    assert len(plot.data[["longitude", "latitude", "z"]].drop_duplicates()) == 8


def test_plot_3d_smooth_dist_supports_rowwise_newdata() -> None:
    term, model = _tensor3()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    plot = ptm.plot_3d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        x="longitude",
        y="latitude",
        ridge_by="z",
        newdata={
            "longitude": np.asarray([10.2, 10.8, 10.2]),
            "latitude": np.asarray([50.3, 50.7, 50.3]),
            "z": np.asarray([0.8, 0.2, 0.2]),
        },
        rgrid=7,
        ci_quantiles=None,
    )

    assert isinstance(plot.data, pd.DataFrame)
    assert len(plot.data[["longitude", "latitude", "z"]].drop_duplicates()) == 3


def test_plot_3d_smooth_dist_meshgrid_draws_ordered_density_ridges() -> None:
    term, model = _tensor3()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    plot = ptm.plot_3d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        x="longitude",
        y="latitude",
        ridge_by="z",
        newdata={
            "longitude": np.asarray([10.2, 10.8]),
            "latitude": np.asarray([50.3, 50.7]),
            "z": np.asarray([0.8, 0.2]),
        },
        newdata_meshgrid=True,
        rgrid=7,
    )
    figure = plot.draw()

    assert isinstance(plot.data, pd.DataFrame)
    assert len(plot.data[["longitude", "latitude", "z"]].drop_duplicates()) == 8
    assert plot.mapping["x"] == "r"
    assert plot.mapping["color"] == "z"
    assert plot.labels.color == "z"
    _assert_gray_reference_lines(plot)
    assert plot.labels.y == "Density"
    assert any(isinstance(layer.geom, p9.geom_ribbon) for layer in plot.layers)
    assert len(figure.axes) == 4
    assert all(not axis.get_yticklabels() for axis in figure.axes)
    baselines = plot.data.groupby("z")["baseline"].first().sort_index()
    assert np.all(np.diff(baselines) > 0)


def test_plot_3d_smooth_dist_adds_opt_in_hdi_and_trajectories() -> None:
    term, model = _tensor3()
    assert term.model is model
    samples = {
        term.coef.name: jnp.arange(3 * term.coef.value.size, dtype=float).reshape(3, -1)
        / 100.0
    }
    kwargs: dict[str, Any] = {
        "x": "longitude",
        "y": "latitude",
        "ridge_by": "z",
        "newdata": {
            "longitude": np.asarray([10.2, 10.2]),
            "latitude": np.asarray([50.3, 50.3]),
            "z": np.asarray([0.8, 0.2]),
        },
        "rgrid": 7,
        "ci_quantiles": None,
    }

    default = ptm.plot_3d_smooth_dist(ptm.onion_dist(nparam=4), term, samples, **kwargs)
    hdi = ptm.plot_3d_smooth_dist(
        ptm.onion_dist(nparam=4), term, samples, hdi_prob=0.8, **kwargs
    )
    sampled = ptm.plot_3d_smooth_dist(
        ptm.onion_dist(nparam=4), term, samples, show_n_samples=2, **kwargs
    )

    assert len(hdi.layers) == len(default.layers) + 1
    assert len(sampled.layers) == len(default.layers) + 1
    assert len(sampled.draw().axes) == 1


def test_plot_3d_smooth_dist_stacked_adds_opt_in_hdi() -> None:
    term, model = _tensor3()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}
    kwargs: dict[str, Any] = {
        "x": "longitude",
        "y": "latitude",
        "ridge_by": "z",
        "points": {
            "longitude": np.asarray([10.2]),
            "latitude": np.asarray([50.3]),
        },
        "ridge_values": [0.8, 0.2],
        "rgrid": 9,
    }

    default = ptm.plot_3d_smooth_dist_stacked(
        ptm.onion_dist(nparam=4), term, samples, **kwargs
    )
    hdi = ptm.plot_3d_smooth_dist_stacked(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        ci_quantiles=(0.05, 0.95),
        hdi_prob=0.8,
        **kwargs,
    )

    assert len(hdi.layers) > len(default.layers)
    assert len(hdi.draw().axes) == 1


def test_plot_cluster_dist_can_show_and_hide_unobserved_levels() -> None:
    term, model = _cluster()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    shown_plot = ptm.plot_cluster_dist(ptm.onion_dist(nparam=4), term, samples, rgrid=9)
    shown = shown_plot.draw()
    hidden = ptm.plot_cluster_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=9,
        show_unobserved=False,
    ).draw()

    assert len(shown.axes[0].get_yticks()) == 3
    assert len(hidden.axes[0].get_yticks()) == 2
    assert shown_plot.mapping["color"] == "group"
    _assert_gray_reference_lines(shown_plot)


def test_plot_cluster_dist_supports_opt_in_trajectories() -> None:
    term, model = _cluster()
    assert term.model is model
    samples = {
        term.coef.name: jnp.arange(3 * term.coef.value.size, dtype=float).reshape(3, -1)
        / 100.0
    }

    default = ptm.plot_cluster_dist(
        ptm.onion_dist(nparam=4), term, samples, rgrid=9
    ).draw()
    sampled = ptm.plot_cluster_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=9,
        show_n_samples=2,
    ).draw()

    assert len(sampled.axes[0].lines) > len(default.axes[0].lines)


def test_plot_regions_dist_uses_polygon_centroids_for_density_glyphs() -> None:
    term, model = _cluster()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    polys = {
        "a": square,
        "b": square + np.array([2, 0]),
        "c": square + np.array([4, 0]),
    }

    plot = ptm.plot_regions_dist(
        ptm.onion_dist(nparam=4), term, samples, polys=polys, rgrid=9
    )
    figure = plot.draw()

    assert isinstance(plot, p9.ggplot)
    assert isinstance(plot.data, pd.DataFrame)
    np.testing.assert_allclose(
        plot.data.groupby("group", observed=False)[["anchor_x", "anchor_y"]]
        .first()
        .to_numpy(),
        np.asarray([[0.5, 0.5], [2.5, 0.5], [4.5, 0.5]]),
    )
    assert len(figure.axes) == 1


def test_plot_regions_dist_adds_opt_in_hdi() -> None:
    term, model = _cluster()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}
    locations = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (2.0, 0.0)}

    default = ptm.plot_regions_dist(
        ptm.onion_dist(nparam=4), term, samples, locations=locations, rgrid=9
    )
    hdi = ptm.plot_regions_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        locations=locations,
        rgrid=9,
        ci_quantiles=(0.05, 0.95),
        hdi_prob=0.8,
    )

    assert len(hdi.layers) > len(default.layers)
    assert len(hdi.draw().axes) == 1


def test_plot_regions_dist_can_hide_density_fill() -> None:
    term, _ = _cluster()
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}
    locations = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (2.0, 0.0)}

    filled = ptm.plot_regions_dist(
        ptm.onion_dist(nparam=4), term, samples, locations=locations, rgrid=9
    )
    unfilled = ptm.plot_regions_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        locations=locations,
        rgrid=9,
        show_density_fill=False,
    )

    assert len(unfilled.layers) == len(filled.layers) - 1


def test_plot_regions_dist_uses_native_liesel_gam_polygons() -> None:
    groups = pd.Categorical(["a", "b", "c"], categories=["a", "b", "c"])
    polys = {
        label: np.array([[i, 0.0], [i + 1.0, 0.0], [i + 1.0, 1.0], [i, 1.0], [i, 0.0]])
        for i, label in enumerate(groups.categories)
    }
    polys["b"] = np.vstack((polys["b"], [np.nan, np.nan]))
    builder = gam.MVTermBuilder.from_df(pd.DataFrame({"group": groups}), jnp.eye(4))
    term = builder.mrf("group", polys=polys, scale=1.0, dimension_scale=1.0)
    model = lsl.Model([term])
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    plot = ptm.plot_regions_dist(ptm.onion_dist(nparam=4), term, samples, rgrid=9)

    assert isinstance(plot.data, pd.DataFrame)
    assert set(plot.data["group"]) == {"a", "b", "c"}
    assert np.isfinite(plot.data[["glyph_x", "glyph_y"]]).all().all()
    assert len(plot.draw().axes) == 1
