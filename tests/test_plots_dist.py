from typing import Any

import jax.numpy as jnp
import liesel.model as lsl
import liesel_gam as gam
import numpy as np
import pandas as pd
import plotnine as p9
import pytest
from matplotlib.text import Text

import liesel_ptm as ptm

_OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]


def _assert_gray_reference_lines(plot: p9.ggplot, *, shown: bool = True) -> None:
    reference_layers = [
        layer
        for layer in plot.layers
        if isinstance(layer.geom, p9.geom_line)
        and layer.geom.aes_params.get("color") == "gray"
    ]
    assert len(reference_layers) == int(shown)
    if shown:
        assert reference_layers[0].geom.aes_params["linetype"] == "dotted"
        assert reference_layers[0].geom.aes_params["alpha"] == 0.5


def _assert_cluster_linetypes(plot: p9.ggplot) -> None:
    scale = plot.scales.get_scales("linetype")
    assert scale is not None
    assert scale.map([True, False]) == ["solid", "dashed"]


def _assert_okabe_ito_scales(plot: p9.ggplot, n_categories: int) -> None:
    color = plot.scales.get_scales("color")
    fill = plot.scales.get_scales("fill")
    assert isinstance(color, p9.scale_color_manual)
    assert isinstance(fill, p9.scale_fill_manual)
    assert color.map(color.final_limits) == _OKABE_ITO[:n_categories]
    assert fill.map(fill.final_limits) == _OKABE_ITO[:n_categories]


def _assert_ridge_baselines(plot: p9.ggplot, *, shown: bool) -> None:
    baseline_layers = [
        layer for layer in plot.layers if isinstance(layer.geom, p9.geom_hline)
    ]
    assert len(baseline_layers) == int(shown)
    if shown:
        layer = baseline_layers[0]
        assert layer.geom.aes_params == {"alpha": 0.35, "linetype": "dotted"}
        assert isinstance(plot.data, pd.DataFrame)
        np.testing.assert_allclose(
            np.sort(layer._data["yintercept"]),
            np.sort(plot.data["baseline"].drop_duplicates()),
        )


def _assert_ridge_ribbons_are_fill_only(plot: p9.ggplot) -> None:
    ribbons = [layer for layer in plot.layers if isinstance(layer.geom, p9.geom_ribbon)]
    assert ribbons
    assert all(layer.geom.aes_params["color"] == "none" for layer in ribbons)


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


def _categorical_linear():
    categories = pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"])
    builder = gam.MVTermBuilder.from_df(pd.DataFrame({"myvar": categories}), jnp.eye(4))
    term = builder.lin("C(myvar, contr.sum)", dimension_scale=1.0)
    model = lsl.Model([term])
    return term, model


def _conditional_response():
    x = lsl.Var.new_obs(jnp.array([0.0, 1.0]), name="conditional_x")
    z = lsl.Var.new_obs(jnp.array([0.0, 1.0]), name="conditional_z")
    loc_coef = lsl.Var.new_param(1.0, name="conditional_loc_coef")
    scale_coef = lsl.Var.new_param(jnp.log(2.0), name="conditional_scale_coef")
    loc = lsl.Var.new_calc(
        lambda x, coef: x * coef, x, loc_coef, name="conditional_loc"
    )
    scale = lsl.Var.new_calc(
        lambda z, coef: jnp.exp(z * coef),
        z,
        scale_coef,
        name="conditional_scale",
    )
    coef = lsl.Var.new_param(jnp.zeros(4), name="conditional_shape")
    response = lsl.Var.new_obs(
        jnp.zeros(2),
        lsl.Dist(ptm.onion_dist(nparam=4), coef=coef, loc=loc, scale=scale),
        name="conditional_response",
    )
    model = lsl.Model([response])
    samples = {
        coef.name: jnp.zeros((1, 1, 4)),
        loc_coef.name: jnp.ones((1, 1)),
        scale_coef.name: jnp.full((1, 1), jnp.log(2.0)),
    }
    return response, samples, model


@pytest.mark.parametrize("n_categories", range(1, 9))
def test_discrete_scales_use_okabe_ito_up_to_eight_categories(
    n_categories: int,
) -> None:
    groups = [f"group-{i}" for i in range(n_categories)]
    data = pd.DataFrame(
        {
            "group": pd.Categorical(
                groups,
                categories=[*groups, "unused"],
            )
        }
    )
    plot = p9.ggplot(data) + ptm.plots._rounded_scales(
        data, color="group", fill="group"
    )

    _assert_okabe_ito_scales(plot, n_categories)


def test_discrete_scales_use_viridis_from_nine_categories() -> None:
    groups = [f"group-{i}" for i in range(9)]
    data = pd.DataFrame({"group": groups})
    plot = p9.ggplot(data) + ptm.plots._rounded_scales(
        data, color="group", fill="group"
    )
    color = plot.scales.get_scales("color")
    fill = plot.scales.get_scales("fill")

    assert isinstance(color, p9.scale_color_cmap_d)
    assert isinstance(fill, p9.scale_fill_cmap_d)
    assert color.map(color.final_limits) == fill.map(fill.final_limits)
    assert color.map(color.final_limits)[::8] == ["#440154", "#fde725"]


def test_numeric_color_and_fill_scales_remain_continuous() -> None:
    data = pd.DataFrame({"group": np.arange(9)})
    plot = p9.ggplot(data) + ptm.plots._rounded_scales(
        data, color="group", fill="group"
    )

    assert isinstance(plot.scales.get_scales("color"), p9.scale_color_continuous)
    assert isinstance(plot.scales.get_scales("fill"), p9.scale_fill_continuous)


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


def test_plot_conditional_density_uses_unique_condition_ridges() -> None:
    response, samples, model = _conditional_response()
    assert response.model is model
    newdata = [
        {"conditional_x": 0.0, "conditional_z": 0.0},
        {"conditional_x": 1.0, "conditional_z": 1.0},
    ]

    plot = ptm.plot_conditional_dist(
        response,
        samples,
        newdata=newdata,
        rgrid=jnp.array([-1.0, 0.0, 1.0]),
        include_loc=True,
        include_scale=True,
        ci_quantiles=None,
    )
    figure = plot.draw()

    assert isinstance(plot.data, pd.DataFrame)
    data = plot.data
    assert list(data["_condition"].cat.categories) == [
        "conditional_x=0, conditional_z=0",
        "conditional_x=1, conditional_z=1",
    ]
    baselines = data["baseline"].drop_duplicates().to_numpy()
    np.testing.assert_allclose(baselines, [0.0, 1.15 * float(data["mean"].max())])
    assert not figure.axes[0].get_yticklabels()
    _assert_ridge_baselines(plot, shown=False)
    _assert_okabe_ito_scales(plot, 2)

    spaced = ptm.plot_conditional_dist(
        response,
        samples,
        newdata=newdata,
        rgrid=jnp.array([-1.0, 0.0, 1.0]),
        ridge_spacing=0.7,
        show_ridge_baselines=True,
        ci_quantiles=None,
    )
    assert isinstance(spaced.data, pd.DataFrame)
    np.testing.assert_allclose(
        spaced.data["baseline"].drop_duplicates().to_numpy(), [0.0, 0.7]
    )
    _assert_ridge_baselines(spaced, shown=True)

    overlaid = ptm.plot_conditional_dist(
        response,
        samples,
        newdata=newdata,
        rgrid=jnp.array([-1.0, 0.0, 1.0]),
        ridge_spacing=0.0,
        show_ridge_baselines=True,
        ci_quantiles=None,
    )
    assert not overlaid.draw().axes[0].get_yticklabels()
    _assert_ridge_baselines(overlaid, shown=True)


@pytest.mark.parametrize(
    ("newdata", "quantity"),
    [
        (
            {"conditional_x": [0.0, 1.0], "conditional_z": [0.0, 1.0]},
            "density",
        ),
        (
            [
                {"conditional_x": 0.0, "conditional_z": 0.0},
                {"conditional_x": 1.0, "conditional_z": 1.0},
            ],
            "cdf",
        ),
    ],
)
def test_plot_conditional_dist_accepts_custom_condition_labels(
    newdata, quantity: str
) -> None:
    response, samples, model = _conditional_response()
    assert response.model is model

    plot = ptm.plot_conditional_dist(
        response,
        samples,
        newdata=newdata,
        condition_labels=["Control", "Treatment"],
        quantity=quantity,
        rgrid=jnp.array([-1.0, 0.0, 1.0]),
        ci_quantiles=None,
    )
    figure = plot.draw()

    assert isinstance(plot.data, pd.DataFrame)
    assert list(plot.data["_condition"].cat.categories) == ["Control", "Treatment"]
    assert plot.mapping["color"] == "_condition"
    assert plot.labels.color == "Condition"
    assert {"Control", "Treatment"} <= {
        text.get_text() for text in figure.findobj(Text)
    }
    _assert_okabe_ito_scales(plot, 2)


@pytest.mark.parametrize(
    ("condition_labels", "error", "message"),
    [
        ("Control", TypeError, "sequence of strings"),
        (["Control", 1], TypeError, "sequence of strings"),
        (["Control"], ValueError, "one label per condition"),
        (["Same", "Same"], ValueError, "unique"),
    ],
)
def test_plot_conditional_dist_validates_condition_labels(
    condition_labels, error: type[Exception], message: str
) -> None:
    response, samples, model = _conditional_response()
    assert response.model is model

    with pytest.raises(error, match=message):
        ptm.plot_conditional_dist(
            response,
            samples,
            newdata={"conditional_x": [0.0, 1.0], "conditional_z": [0.0, 1.0]},
            condition_labels=condition_labels,
            rgrid=jnp.array([-1.0, 0.0, 1.0]),
        )


def test_plot_conditional_dist_rejects_duplicate_conditions() -> None:
    response, samples, model = _conditional_response()
    assert response.model is model

    with pytest.raises(ValueError, match="duplicate condition rows"):
        ptm.plot_conditional_dist(
            response,
            samples,
            newdata={"conditional_x": [0.0, 0.0], "conditional_z": [1.0, 1.0]},
            condition_labels=["Control", "Control"],
            rgrid=jnp.array([-1.0, 0.0, 1.0]),
        )


def test_plot_conditional_non_density_overlays_grouped_curves() -> None:
    response, samples, model = _conditional_response()
    assert response.model is model

    plot = ptm.plot_conditional_dist(
        response,
        samples,
        newdata={"conditional_x": [0.0, 1.0], "conditional_z": [0.0, 1.0]},
        quantity="cdf",
        rgrid=jnp.array([-1.0, 0.0, 1.0]),
        include_loc=True,
        include_scale=True,
        show_ridge_baselines=True,
        ci_quantiles=None,
    )
    figure = plot.draw()

    assert plot.mapping["group"] == "_condition"
    assert plot.mapping["color"] == "_condition"
    assert len(figure.axes[0].lines) == 2
    _assert_ridge_baselines(plot, shown=False)
    assert not any(
        isinstance(layer.geom, p9.geom_line)
        and layer.geom.aes_params.get("linetype") == "dotted"
        for layer in plot.layers
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

    default_plot = ptm.plot_intercept_dist(
        ptm.onion_dist(nparam=4), term, samples, rgrid=11
    )
    sampled_plot = ptm.plot_intercept_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=11,
        show_reference_dist=False,
        show_n_samples=2,
        seed=2,
    )
    default = default_plot.draw()
    sampled = sampled_plot.draw()

    assert len(sampled.axes[0].lines) > len(default.axes[0].lines)
    _assert_gray_reference_lines(default_plot)
    _assert_gray_reference_lines(sampled_plot, shown=False)


def test_plot_1d_smooth_dist_uses_covariate_ridge_baselines() -> None:
    term, model = _smooth()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    plot = ptm.plot_1d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=11,
        newdata={"x": np.asarray([1.5, 0.126, 0.004])},
    )
    figure = plot.draw()
    hidden_axis_plot = ptm.plot_1d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=11,
        newdata={"x": np.asarray([1.5, 0.126, 0.004])},
        show_reference_dist=False,
        show_ridge_baselines=True,
        show_y_axis=False,
    )
    hidden_axis = hidden_axis_plot.draw()

    assert isinstance(plot, p9.ggplot)
    assert isinstance(plot.data, pd.DataFrame)
    assert [label.get_text() for label in figure.axes[0].get_yticklabels()] == [
        "0",
        "0.13",
        "1.5",
    ]
    assert not hidden_axis.axes[0].get_yticklabels()
    assert plot.mapping["color"] == "x"
    assert plot.labels.color == "x"
    _assert_gray_reference_lines(plot)
    _assert_gray_reference_lines(hidden_axis_plot, shown=False)
    _assert_ridge_baselines(plot, shown=False)
    _assert_ridge_baselines(hidden_axis_plot, shown=True)
    _assert_ridge_ribbons_are_fill_only(plot)
    assert any(
        isinstance(layer.geom, p9.geom_line) and not layer.geom.aes_params
        for layer in plot.layers
    )
    baselines = plot.data.groupby("x")["baseline"].first().sort_index()
    assert np.all(np.diff(baselines) > 0)


def test_plot_1d_smooth_dist_uses_density_axis_with_zero_spacing() -> None:
    term, model = _smooth()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}
    kwargs: dict[str, Any] = {
        "rgrid": 7,
        "newdata": {"x": np.asarray([0.0, 0.5, 1.0])},
        "ridge_spacing": 0.0,
        "show_ridge_baselines": True,
        "ci_quantiles": None,
    }

    shown = ptm.plot_1d_smooth_dist(ptm.onion_dist(nparam=4), term, samples, **kwargs)
    hidden = ptm.plot_1d_smooth_dist(
        ptm.onion_dist(nparam=4), term, samples, show_y_axis=False, **kwargs
    )
    hidden_axis = hidden.draw().axes[0]
    shown_figure = shown.draw()
    shown_axis = shown_figure.axes[0]

    assert not hidden_axis.get_yticklabels()
    assert shown.labels.y == "Density"
    assert "Density" in [text.get_text() for text in shown_figure.texts]
    assert shown_axis.get_yticklabels()
    assert len(np.unique(shown_axis.get_yticks())) == len(shown_axis.get_yticks())
    assert len(shown_axis.get_yticks()) > 1
    _assert_ridge_baselines(shown, shown=True)


def test_plot_labels_round_numeric_values_to_two_decimals() -> None:
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
        rgrid=7,
        ngrid=4,
        show_reference_dist=False,
        show_ridge_baselines=True,
    )
    figure = plot.draw()

    numeric_labels = [
        text.get_text()
        for text in figure.findobj(Text)
        if any(character.isdigit() for character in text.get_text())
    ]
    assert numeric_labels
    assert all(
        len(label.rpartition(".")[2]) <= 2 for label in numeric_labels if "." in label
    )
    _assert_ridge_baselines(plot, shown=True)
    _assert_gray_reference_lines(plot, shown=False)


def test_plot_1d_smooth_dist_supports_opt_in_trajectories() -> None:
    term, model = _smooth()
    assert term.model is model
    samples = {
        term.coef.name: jnp.arange(3 * term.coef.value.size, dtype=float).reshape(3, -1)
        / 50.0
    }

    default_plot = ptm.plot_1d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        quantity="cdf",
        rgrid=9,
        ngrid=2,
    )
    sampled_plot = ptm.plot_1d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        quantity="cdf",
        rgrid=9,
        ngrid=2,
        show_reference_dist=False,
        show_n_samples=2,
    )
    default = default_plot.draw()
    sampled = sampled_plot.draw()

    assert len(sampled.axes[0].lines) > len(default.axes[0].lines)
    _assert_gray_reference_lines(default_plot)
    _assert_gray_reference_lines(sampled_plot, shown=False)


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
    _assert_ridge_ribbons_are_fill_only(hdi)
    assert len(hdi.draw().axes) == 1


def test_plot_2d_smooth_dist_labels_continuous_ridges() -> None:
    term, model = _tensor2()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    plot = ptm.plot_2d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=7,
        newdata={
            "x": np.asarray([1.5, 0.126, 0.004]),
            "z": np.asarray([2.0, 1.5, 1.0]),
        },
        newdata_meshgrid=True,
    )
    figure = plot.draw()

    assert isinstance(plot, p9.ggplot)
    assert len(figure.axes) == 3
    assert plot.mapping["color"] == "x"
    assert plot.labels.color == "x"
    assert plot.labels.y == "x"
    _assert_gray_reference_lines(plot)
    _assert_ridge_baselines(plot, shown=False)
    _assert_ridge_ribbons_are_fill_only(plot)
    ylabels = [
        [label.get_text() for label in axis.get_yticklabels()]
        for axis in figure.axes
        if axis.get_yticklabels()
    ]
    assert ylabels
    assert all(labels == ["0", "0.13", "1.5"] for labels in ylabels)
    assert isinstance(plot.data, pd.DataFrame)
    baselines = plot.data.groupby("x")["baseline"].first().sort_index()
    assert np.all(np.diff(baselines) > 0)

    overlaid = ptm.plot_2d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=7,
        ngrid=2,
        ridge_spacing=0.0,
        ci_quantiles=None,
    )
    overlaid_figure = overlaid.draw()
    assert overlaid.labels.y == "Density"
    assert "Density" in [text.get_text() for text in overlaid_figure.texts]
    assert all(
        len(np.unique(axis.get_yticks())) == len(axis.get_yticks()) > 1
        for axis in overlaid_figure.axes
    )


@pytest.mark.parametrize(
    ("quantity", "label", "show_reference_dist"),
    [
        ("cdf", "CDF", True),
        ("transformation", "Transformation", False),
        ("transformation_raw", "Raw transformation", True),
    ],
)
def test_plot_2d_smooth_dist_labels_non_density_quantities(
    quantity: str, label: str, show_reference_dist: bool
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
        show_reference_dist=show_reference_dist,
    )

    assert plot.labels.y == label
    _assert_gray_reference_lines(plot, shown=show_reference_dist)


def test_plot_2d_smooth_dist_supports_opt_in_trajectories() -> None:
    term, model = _tensor2()
    assert term.model is model
    samples = {
        term.coef.name: jnp.arange(3 * term.coef.value.size, dtype=float).reshape(3, -1)
        / 100.0
    }

    default_plot = ptm.plot_2d_smooth_dist(
        ptm.onion_dist(nparam=4), term, samples, rgrid=7, ngrid=2
    )
    sampled_plot = ptm.plot_2d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=7,
        ngrid=2,
        show_reference_dist=False,
        show_n_samples=2,
        show_ridge_baselines=True,
    )
    default = default_plot.draw()
    sampled = sampled_plot.draw()

    assert sum(len(axis.lines) for axis in sampled.axes) > sum(
        len(axis.lines) for axis in default.axes
    )
    _assert_ridge_baselines(default_plot, shown=False)
    _assert_ridge_baselines(sampled_plot, shown=True)
    _assert_gray_reference_lines(default_plot)
    _assert_gray_reference_lines(sampled_plot, shown=False)


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
    _assert_ridge_baselines(plot, shown=False)
    _assert_ridge_ribbons_are_fill_only(plot)
    assert plot.labels.y == "z"
    assert any(isinstance(layer.geom, p9.geom_ribbon) for layer in plot.layers)
    assert len(figure.axes) == 4
    ylabels = [
        [label.get_text() for label in axis.get_yticklabels()]
        for axis in figure.axes
        if axis.get_yticklabels()
    ]
    assert ylabels
    assert all(labels == ["0.2", "0.8"] for labels in ylabels)
    baselines = plot.data.groupby("z")["baseline"].first().sort_index()
    assert np.all(np.diff(baselines) > 0)

    overlaid = ptm.plot_3d_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        x="longitude",
        y="latitude",
        ridge_by="z",
        rgrid=7,
        ngrid=2,
        ridge_spacing=0.0,
        ci_quantiles=None,
    )
    overlaid_figure = overlaid.draw()
    assert overlaid.labels.y == "Density"
    assert "Density" in [text.get_text() for text in overlaid_figure.texts]
    assert all(
        len(np.unique(axis.get_yticks())) == len(axis.get_yticks()) > 1
        for axis in overlaid_figure.axes
    )


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
    _assert_ridge_ribbons_are_fill_only(hdi)
    assert len(hdi.draw().axes) == 1


def test_plot_cluster_dist_can_show_and_hide_unobserved_levels() -> None:
    term, model = _cluster()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    shown_plot = ptm.plot_cluster_dist(ptm.onion_dist(nparam=4), term, samples, rgrid=9)
    shown = shown_plot.draw()
    hidden_plot = ptm.plot_cluster_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=9,
        show_reference_dist=False,
        show_unobserved=False,
        show_ridge_baselines=True,
    )
    hidden = hidden_plot.draw()

    assert len(shown.axes[0].get_yticks()) == 3
    assert len(hidden.axes[0].get_yticks()) == 2
    assert [label.get_text() for label in shown.axes[0].get_yticklabels()] == [
        "a",
        "b",
        "c",
    ]
    assert shown_plot.labels.y == "group"
    assert shown_plot.mapping["color"] == "group"
    _assert_cluster_linetypes(shown_plot)
    _assert_cluster_linetypes(hidden_plot)
    _assert_okabe_ito_scales(shown_plot, 3)
    _assert_okabe_ito_scales(hidden_plot, 2)
    _assert_gray_reference_lines(shown_plot)
    _assert_gray_reference_lines(hidden_plot, shown=False)
    _assert_ridge_baselines(shown_plot, shown=False)
    _assert_ridge_baselines(hidden_plot, shown=True)
    _assert_ridge_ribbons_are_fill_only(shown_plot)


def test_plot_cluster_dist_uses_density_axis_with_zero_spacing() -> None:
    term, model = _cluster()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    plot = ptm.plot_cluster_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=7,
        ridge_spacing=0.0,
        show_ridge_baselines=True,
    )
    figure = plot.draw()
    axis = figure.axes[0]
    ticks = axis.get_yticks()
    labels = [label.get_text() for label in axis.get_yticklabels()]

    assert plot.labels.y == "Density"
    assert "Density" in [text.get_text() for text in figure.texts]
    assert len(np.unique(ticks)) == len(ticks)
    assert len(ticks) > 1
    assert not set(labels) & {"a", "b", "c"}
    assert plot.labels.color == "group"
    _assert_ridge_baselines(plot, shown=True)


def test_plot_cluster_dist_uses_cluster_linetypes_for_non_density_curves() -> None:
    term, model = _cluster()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    plot = ptm.plot_cluster_dist(
        ptm.onion_dist(nparam=4), term, samples, quantity="cdf", rgrid=9
    )

    assert len(plot.draw().axes) == 1
    _assert_cluster_linetypes(plot)
    _assert_gray_reference_lines(plot)


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


def test_plot_cluster_dist_supports_categorical_linear_term() -> None:
    term, model = _categorical_linear()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    plot = ptm.plot_cluster_dist(ptm.onion_dist(nparam=4), term, samples, rgrid=9)
    figure = plot.draw()
    cdf = ptm.plot_cluster_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        quantity="cdf",
        rgrid=9,
        show_reference_dist=False,
    )

    assert len(figure.axes[0].get_yticks()) == 3
    assert plot.mapping["color"] == "myvar"
    _assert_gray_reference_lines(cdf, shown=False)


def test_categorical_linear_trajectories_use_the_selected_category() -> None:
    term, model = _categorical_linear()
    assert term.model is model
    samples = {
        term.coef.name: jnp.arange(term.coef.value.size, dtype=float)[None, :] / 10.0
    }

    figure = ptm.plot_cluster_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        quantity="cdf",
        rgrid=9,
        newdata={"myvar": ["c"]},
        ci_quantiles=None,
        show_n_samples=1,
    ).draw()

    trajectory, mean, _reference = figure.axes[0].lines
    np.testing.assert_allclose(
        np.asarray(trajectory.get_ydata(), dtype=float),
        np.asarray(mean.get_ydata(), dtype=float),
    )


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
