import jax.numpy as jnp
import liesel.model as lsl
import liesel_gam as gam
import numpy as np
import pandas as pd
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm


def _basis(x):
    x = jnp.squeeze(x, axis=-1)
    return jnp.column_stack((jnp.ones_like(x), x))


def _intercept():
    builder = gam.MVTermBuilder.from_df(pd.DataFrame({"x": [0.0, 1.0]}), jnp.eye(4))
    term = builder.intercept(scale=1.0)
    model = lsl.Model([term])
    return term, model


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


def _scalar_smooth(builder, name):
    return builder.f(
        name,
        basis_fn=_basis,
        penalty=jnp.eye(2),
        scale=1.0,
        use_callback=False,
    )


def _tensor2():
    data = pd.DataFrame(
        {"x": jnp.linspace(0.0, 1.0, 6), "z": jnp.linspace(1.0, 2.0, 6)}
    )
    scalar_builder = gam.TermBuilder.from_df(data)
    builder = gam.MVTermBuilder.from_term_builder(scalar_builder, jnp.eye(4))
    term = builder.tx(
        _scalar_smooth(scalar_builder, "x"),
        _scalar_smooth(scalar_builder, "z"),
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


def _mixed_tensor():
    groups = pd.Categorical(["a", "b", "a", "b"], categories=["a", "b", "c"])
    data = pd.DataFrame({"x": jnp.linspace(0.0, 1.0, 4), "group": groups})
    scalar_builder = gam.TermBuilder.from_df(data)
    builder = gam.MVTermBuilder.from_term_builder(scalar_builder, jnp.eye(4))
    term = builder.tx(
        _scalar_smooth(scalar_builder, "x"),
        scalar_builder.ri("group", scale=1.0),
        dimension_scale=1.0,
    )
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


def _tensor_with_marginal():
    data = pd.DataFrame(
        {"x": jnp.linspace(0.0, 1.0, 6), "z": jnp.linspace(1.0, 2.0, 6)}
    )
    scalar_builder = gam.TermBuilder.from_df(data)
    builder = gam.MVTermBuilder.from_term_builder(scalar_builder, jnp.eye(4))
    tensor = builder.tx(
        _scalar_smooth(scalar_builder, "x"),
        _scalar_smooth(scalar_builder, "z"),
        dimension_scale=1.0,
    )
    marginal = builder.f(
        "x",
        basis_fn=_basis,
        penalty=jnp.eye(2),
        scale=1.0,
        dimension_scale=1.0,
        use_callback=False,
    )
    model = lsl.Model([tensor, marginal])
    return tensor, marginal, model


def test_summarise_intercept_dist_reports_all_quantities() -> None:
    dist = ptm.onion_dist(a=-4.0, b=4.0, nparam=4, loc_scale=False)
    term, model = _intercept()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(4)}

    summary = ptm.summarise_intercept_dist(dist, term, samples, rgrid=jnp.array([0.0]))

    expected_dist = dist(coef=jnp.zeros(4))
    expected = {
        "density": expected_dist.prob(jnp.array([0.0]))[0],
        "cdf": expected_dist.cdf(jnp.array([0.0]))[0],
        "transformation": expected_dist.transformation_and_logdet(jnp.array([0.0]))[0][
            0
        ],
        "transformation_raw": expected_dist.transformation_and_logdet_spline(
            jnp.array([0.0])
        )[0][0],
    }
    assert set(summary["quantity"]) == set(expected)
    assert summary["sample_size"].eq(1).all()
    assert summary["r"].eq(0.0).all()
    assert summary.set_index("quantity")["mean"].to_dict() == pytest.approx(expected)


@pytest.mark.parametrize(
    "newdata",
    [
        {"conditional_x": [0.0, 1.0], "conditional_z": [1.0, 0.0]},
        [
            {"conditional_x": 0.0, "conditional_z": 1.0},
            {"conditional_x": 1.0, "conditional_z": 0.0},
        ],
    ],
)
def test_summarise_conditional_dist_preserves_condition_order(newdata) -> None:
    response, samples, model = _conditional_response()
    assert response.model is model

    summary = ptm.summarise_conditional_dist(
        response, samples, newdata=newdata, rgrid=jnp.array([-1.0, 1.0])
    )

    assert summary.shape[0] == 4 * 2 * 2
    assert summary["quantity"].drop_duplicates().tolist() == [
        "density",
        "cdf",
        "transformation",
        "transformation_raw",
    ]
    conditions = summary.loc[
        summary["quantity"] == "density", ["conditional_x", "conditional_z"]
    ].drop_duplicates()
    assert conditions.to_records(index=False).tolist() == [(0.0, 1.0), (1.0, 0.0)]


def test_summarise_conditional_dist_transforms_response_distribution() -> None:
    response, samples, model = _conditional_response()
    assert response.model is model
    rgrid = jnp.array([0.5, 1.0, 2.0])
    bijector = tfb.Exp()

    summary = ptm.summarise_conditional_dist(
        response,
        samples,
        newdata={"conditional_x": [0.0], "conditional_z": [0.0]},
        rgrid=rgrid,
        response_bijector=bijector,
    )

    constructor = ptm.onion_dist(nparam=4)
    fitted = constructor(coef=jnp.zeros(4), loc=0.0, scale=1.0)
    raw = constructor(
        coef=jnp.zeros(4), loc=0.0, scale=1.0, centered=False, scaled=False
    )
    reported = tfd.TransformedDistribution(distribution=fitted, bijector=bijector)
    model_rgrid = bijector.inverse(rgrid)
    expected = {
        "density": reported.prob(rgrid),
        "cdf": reported.cdf(rgrid),
        "transformation": fitted.transformation_and_logdet(model_rgrid)[0],
        "transformation_raw": raw.transformation_and_logdet_spline(model_rgrid)[0],
    }

    np.testing.assert_allclose(summary["r"].unique(), rgrid)
    for quantity, values in expected.items():
        actual = summary.loc[summary["quantity"] == quantity, "mean"]
        np.testing.assert_allclose(actual, values, rtol=1e-5)


def test_summarise_conditional_dist_none_bijector_preserves_results() -> None:
    response, samples, model = _conditional_response()
    assert response.model is model
    newdata = {"conditional_x": [0.0], "conditional_z": [0.0]}
    rgrid = jnp.array([-1.0, 0.0, 1.0])

    default = ptm.summarise_conditional_dist(
        response, samples, newdata=newdata, rgrid=rgrid
    )
    explicit_none = ptm.summarise_conditional_dist(
        response,
        samples,
        newdata=newdata,
        rgrid=rgrid,
        response_bijector=None,
    )

    pd.testing.assert_frame_equal(default, explicit_none)


@pytest.mark.parametrize(
    ("response_bijector", "rgrid", "error", "message"),
    [
        (object(), jnp.array([1.0]), TypeError, "JAX TFP bijector"),
        (
            tfb.ScaleMatvecDiag(jnp.ones(2)),
            jnp.array([1.0]),
            ValueError,
            "scalar-event bijector",
        ),
        (tfb.Exp(), 10, ValueError, "rgrid must be an explicit array"),
    ],
)
def test_summarise_conditional_dist_validates_response_bijector(
    response_bijector, rgrid, error: type[Exception], message: str
) -> None:
    response, samples, model = _conditional_response()
    assert response.model is model

    with pytest.raises(error, match=message):
        ptm.summarise_conditional_dist(
            response,
            samples,
            newdata={"conditional_x": [0.0], "conditional_z": [0.0]},
            rgrid=rgrid,
            response_bijector=response_bijector,
        )


def test_summarise_conditional_dist_retains_noninjective_tfp_error() -> None:
    response, samples, model = _conditional_response()
    assert response.model is model

    with pytest.raises(NotImplementedError, match="cdf.*not implemented"):
        ptm.summarise_conditional_dist(
            response,
            samples,
            newdata={"conditional_x": [0.0], "conditional_z": [0.0]},
            rgrid=jnp.array([0.5, 1.0]),
            response_bijector=tfb.AbsoluteValue(),
        )


@pytest.mark.parametrize(
    "newdata",
    [
        {"conditional_x": [0.0, 1.0, 1.0], "conditional_z": [1.0, 0.0, 0.0]},
        [
            {"conditional_x": 0.0, "conditional_z": 1.0},
            {"conditional_x": 1.0, "conditional_z": 0.0},
            {"conditional_x": 1.0, "conditional_z": 0.0},
        ],
    ],
)
def test_summarise_conditional_dist_rejects_duplicate_conditions(newdata) -> None:
    response, samples, model = _conditional_response()
    assert response.model is model

    with pytest.raises(ValueError, match="duplicate condition rows"):
        ptm.summarise_conditional_dist(response, samples, newdata=newdata)


@pytest.mark.parametrize(
    ("include_loc", "include_scale"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_summarise_conditional_dist_location_scale_switches(
    include_loc, include_scale
) -> None:
    response, samples, model = _conditional_response()
    assert response.model is model
    newdata = {"conditional_x": [0.0, 1.0], "conditional_z": [0.0, 1.0]}

    summary = ptm.summarise_conditional_dist(
        response,
        samples,
        newdata=newdata,
        rgrid=jnp.array([0.0]),
        include_loc=include_loc,
        include_scale=include_scale,
    )

    loc = jnp.array([0.0, 1.0]) if include_loc else 0.0
    scale = jnp.array([1.0, 2.0]) if include_scale else 1.0
    expected = ptm.onion_dist(nparam=4)(coef=jnp.zeros(4), loc=loc, scale=scale).prob(
        jnp.zeros(2)
    )
    density = summary.loc[summary["quantity"] == "density", "mean"]
    np.testing.assert_allclose(density, expected, rtol=1e-5)


@pytest.mark.parametrize(
    ("include_loc", "include_scale"), [(True, False), (False, True), (True, True)]
)
def test_summarise_conditional_dist_requires_explicit_full_scale_grid(
    include_loc, include_scale
) -> None:
    response, samples, model = _conditional_response()
    assert response.model is model

    with pytest.raises(ValueError, match="rgrid must be an explicit array"):
        ptm.summarise_conditional_dist(
            response,
            samples,
            newdata={"conditional_x": [0.0], "conditional_z": [0.0]},
            include_loc=include_loc,
            include_scale=include_scale,
        )


@pytest.mark.parametrize(
    "newdata",
    [
        {},
        {"conditional_x": [0.0], "conditional_z": [0.0, 1.0]},
        [{"conditional_x": 0.0}, {"conditional_z": 1.0}],
        [{"conditional_x": [0.0]}],
    ],
)
def test_summarise_conditional_dist_rejects_malformed_newdata(newdata) -> None:
    response, samples, model = _conditional_response()
    assert response.model is model

    with pytest.raises((TypeError, ValueError)):
        ptm.summarise_conditional_dist(response, samples, newdata=newdata)


def test_summarise_conditional_dist_requires_built_response_distribution() -> None:
    response, samples, model = _conditional_response()
    assert response.model is model
    newdata = {"conditional_x": [0.0], "conditional_z": [0.0]}

    with pytest.raises(TypeError, match="no distribution"):
        ptm.summarise_conditional_dist(
            lsl.Var.new_obs(jnp.zeros(1), name="bare_response"),
            samples,
            newdata=newdata,
        )

    response = lsl.Var.new_obs(
        jnp.zeros(1),
        lsl.Dist(
            ptm.onion_dist(nparam=4),
            coef=jnp.zeros(4),
            loc=0.0,
            scale=1.0,
        ),
        name="unbuilt_response",
    )
    with pytest.raises(ValueError, match="built model"):
        ptm.summarise_conditional_dist(response, {}, newdata=newdata)


@pytest.mark.parametrize("as_response", [False, True])
def test_summarise_intercept_dist_accepts_liesel_dist_or_response(
    as_response,
) -> None:
    constructor = ptm.onion_dist(nparam=4)
    coef = lsl.Var.new_param(jnp.zeros(4), name="shape")
    dist = lsl.Dist(constructor, coef=coef, loc=0.0, scale=1.0)
    supplied_dist = (
        lsl.Var.new_obs(jnp.zeros(1), dist, name="response") if as_response else dist
    )
    term, model = _intercept()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(4)}

    summary = ptm.summarise_intercept_dist(
        supplied_dist, term, samples, rgrid=jnp.array([0.0])
    )

    assert set(summary["quantity"]) == {
        "density",
        "cdf",
        "transformation",
        "transformation_raw",
    }


def test_summarise_1d_smooth_dist_uses_small_default_grid() -> None:
    term, model = _smooth()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    summary = ptm.summarise_1d_smooth_dist(
        ptm.onion_dist(nparam=4), term, samples, rgrid=2
    )

    assert summary.shape[0] == 4 * 5 * 2
    assert summary["x"].unique() == pytest.approx(jnp.linspace(0.0, 1.0, 5))


def test_summarise_nd_smooth_dist_treats_newdata_as_rows() -> None:
    term, model = _tensor2()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}
    newdata = {"x": jnp.array([0.1, 0.2]), "z": jnp.array([1.8, 1.9])}

    summary = ptm.summarise_nd_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=jnp.array([0.0]),
        newdata=newdata,
    )

    assert summary.shape[0] == 4 * 2
    np.testing.assert_allclose(
        summary[["x", "z"]].drop_duplicates().to_numpy(),
        [[0.1, 1.8], [0.2, 1.9]],
    )


def test_summarise_cluster_dist_includes_unobserved_categories() -> None:
    term, model = _cluster()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    summary = ptm.summarise_cluster_dist(
        ptm.onion_dist(nparam=4), term, samples, rgrid=jnp.array([0.0])
    )

    assert summary.shape[0] == 4 * 3
    assert list(summary["group"].cat.categories) == ["a", "b", "c"]
    observed = summary.groupby("group", observed=False)["observed"].first().to_dict()
    assert observed == {"a": True, "b": True, "c": False}


def test_summarise_cluster_dist_supports_categorical_linear_term() -> None:
    categories = pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"])
    builder = gam.MVTermBuilder.from_df(pd.DataFrame({"myvar": categories}), jnp.eye(4))
    term = builder.lin("C(myvar, contr.sum)", dimension_scale=1.0)
    model = lsl.Model([term])
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    summary = ptm.summarise_cluster_dist(
        ptm.onion_dist(nparam=4), term, samples, rgrid=jnp.array([0.0])
    )

    assert summary.shape[0] == 4 * 3
    assert list(summary["myvar"].cat.categories) == ["a", "b", "c"]
    observed = summary.groupby("myvar", observed=False)["observed"].first().to_dict()
    assert observed == {"a": True, "b": True, "c": False}


def test_summarise_nd_smooth_dist_uses_all_mapped_categories() -> None:
    term, model = _mixed_tensor()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    summary = ptm.summarise_nd_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=jnp.array([0.0]),
        ngrid=2,
    )

    assert summary.shape[0] == 4 * 2 * 3
    assert list(summary["group"].cat.categories) == ["a", "b", "c"]


def test_summaries_reject_boolean_intercept_shorthand() -> None:
    term, model = _smooth()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    with pytest.raises(TypeError, match="intercept"):
        ptm.summarise_1d_smooth_dist(  # ty: ignore[no-matching-overload]
            ptm.onion_dist(nparam=4),
            term,
            samples,
            rgrid=2,
            intercept=True,
        )


def test_summarise_nd_smooth_dist_adds_only_explicit_marginals() -> None:
    term, marginal, model = _tensor_with_marginal()
    assert term.model is model
    generic_terms: dict[str, lsl.Var] = {"term": term, "marginal": marginal}
    samples = {
        term.coef.name: jnp.zeros(term.coef.value.shape),
        marginal.coef.name: jnp.full(marginal.coef.value.shape, 0.2),
    }
    newdata = {"x": jnp.array([0.5]), "z": jnp.array([1.5])}

    isolated = ptm.summarise_nd_smooth_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=jnp.array([0.0]),
        newdata=newdata,
    )
    composed = ptm.summarise_nd_smooth_dist(
        ptm.onion_dist(nparam=4),
        generic_terms["term"],
        samples,
        rgrid=jnp.array([0.0]),
        newdata=newdata,
        marginals=(generic_terms["marginal"],),
    )

    assert not np.allclose(isolated["mean"], composed["mean"])


@pytest.mark.parametrize(
    ("sample_shape", "sample_size"),
    [((), 1), ((3,), 3), ((2, 3), 6)],
)
def test_summarise_intercept_dist_normalises_sample_layouts(
    sample_shape, sample_size
) -> None:
    term, model = _intercept()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(sample_shape + term.coef.value.shape)}

    summary = ptm.summarise_intercept_dist(
        ptm.onion_dist(nparam=4), term, samples, rgrid=jnp.array([0.0])
    )

    assert summary["sample_size"].eq(sample_size).all()


def test_summarise_cluster_dist_accepts_explicit_category_mapping() -> None:
    term, model = _cluster()
    assert term.model is model
    mapping = term.marginal_terms[0].mapping
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    summary = ptm.summarise_cluster_dist(
        ptm.onion_dist(nparam=4),
        term,
        samples,
        rgrid=jnp.array([0.0]),
        newdata={"group": np.asarray(["c", "a"])},
        labels=mapping,
    )

    assert list(summary["group"].cat.categories) == ["a", "b", "c"]
    observed = summary[["group", "observed"]].drop_duplicates().set_index("group")
    assert observed["observed"].to_dict() == {"c": False, "a": True}


def test_summaries_reject_non_vector_rgrid() -> None:
    term, model = _intercept()
    assert term.model is model
    samples = {term.coef.name: jnp.zeros(term.coef.value.shape)}

    with pytest.raises(ValueError, match="one-dimensional"):
        ptm.summarise_intercept_dist(
            ptm.onion_dist(nparam=4),
            term,
            samples,
            rgrid=jnp.zeros((2, 2)),
        )
