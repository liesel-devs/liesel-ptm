import jax.numpy as jnp
import liesel.model as lsl
import liesel_gam as gam
import numpy as np
import pandas as pd
import pytest

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
        ptm.summarise_1d_smooth_dist(
            ptm.onion_dist(nparam=4),
            term,
            samples,
            rgrid=2,
            intercept=True,
        )


def test_summarise_nd_smooth_dist_adds_only_explicit_marginals() -> None:
    term, marginal, model = _tensor_with_marginal()
    assert term.model is model
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
        term,
        samples,
        rgrid=jnp.array([0.0]),
        newdata=newdata,
        marginals=(marginal,),
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
