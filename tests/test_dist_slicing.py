"""PTM distributions can be copied, batch-sliced and passed into jax.jit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm

N = 5
NPARAM = 6
SPLINE = ptm.OnionSpline(ptm.OnionKnots(-4.0, 4.0, nparam=NPARAM).knots)
INDICES = jnp.array([0, 2, 2, 4])
VALUES = jnp.linspace(-1.0, 1.5, N)


def coef(shape=()):
    base = jnp.linspace(-0.3, 0.4, NPARAM)
    rows = jnp.arange(int(np.prod(shape))).reshape(shape + (1,)) * 0.05
    return base + rows


def loc_scale(coef_value, loc, scale=1.0):
    return ptm.LocScaleTransformationDist(
        coef=coef_value,
        loc=loc,
        scale=scale,
        bspline=SPLINE,
        centered=True,
        scaled=True,
    )


def transformation(parametric_distribution, coef_value, **kwargs):
    return ptm.TransformationDist(
        coef=coef_value,
        bspline=SPLINE,
        parametric_distribution=parametric_distribution,
        **kwargs,
    )


CASES = {
    "rowwise coef": lambda: loc_scale(coef((N,)), jnp.linspace(0.0, 1.0, N)),
    "shared coef": lambda: loc_scale(coef(), jnp.linspace(0.0, 1.0, N), 0.7),
    "rowwise scale": lambda: loc_scale(coef(), 0.2, jnp.linspace(0.5, 1.5, N)),
    "exponential layer": lambda: transformation(
        tfd.Exponential, coef(), rate=jnp.linspace(0.5, 2.0, N)
    ),
    "two-piece t layer": lambda: transformation(
        tfd.TwoPieceStudentT,
        coef((N,)),
        df=5.0,
        loc=jnp.linspace(0.0, 1.0, N),
        scale=1.0,
        skewness=1.2,
    ),
}


@pytest.mark.parametrize("make", CASES.values(), ids=CASES.keys())
def test_copy_rebuilds_an_equal_distribution(make):
    dist = make()
    copied = dist.copy()
    assert type(copied) is type(dist)
    np.testing.assert_allclose(copied.log_prob(VALUES), dist.log_prob(VALUES))


@pytest.mark.parametrize("make", CASES.values(), ids=CASES.keys())
def test_integer_array_slices_select_rows(make):
    dist = make()
    sliced = dist[INDICES]
    assert type(sliced) is type(dist)
    assert sliced.batch_shape == (len(INDICES),)
    np.testing.assert_allclose(
        sliced.log_prob(VALUES[INDICES]),
        dist.log_prob(VALUES)[INDICES],
        rtol=1e-6,
    )


def test_slices_posterior_batches_along_the_last_axis():
    dist = loc_scale(coef((2, 3, N)), jnp.linspace(0.0, 1.0, N))
    sliced = dist[..., INDICES]
    assert sliced.batch_shape == (2, 3, len(INDICES))
    np.testing.assert_allclose(
        sliced.log_prob(VALUES[INDICES]),
        dist.log_prob(VALUES)[..., INDICES],
        rtol=1e-6,
    )


def test_size_one_axes_broadcast_when_sliced():
    dist = loc_scale(coef((1,)), jnp.zeros(1))
    assert dist[jnp.array([0, 0, 0])].batch_shape == (3,)


def test_batched_reference_distribution_cannot_be_sliced():
    dist = ptm.LocScaleTransformationDist(
        coef=coef((N,)),
        loc=0.0,
        scale=1.0,
        bspline=SPLINE,
        reference_distribution=tfd.Normal(jnp.zeros(N), 1.0),
    )
    with pytest.raises(ValueError, match="reference distribution"):
        dist[INDICES]


def test_batched_parameter_without_properties_cannot_be_sliced():
    class RenamedNormal(tfd.Normal):
        def __init__(self, mu, sigma):
            super().__init__(loc=mu, scale=sigma)

    dist = transformation(RenamedNormal, coef(), mu=jnp.zeros(N), sigma=1.0)
    with pytest.raises(ValueError, match="'mu'"):
        dist[INDICES]


@pytest.mark.parametrize("make", CASES.values(), ids=CASES.keys())
def test_distribution_passes_into_jit(make):
    dist = make()

    @jax.jit
    def log_prob(dist, value):
        return dist.log_prob(value)

    np.testing.assert_allclose(log_prob(dist, VALUES), dist.log_prob(VALUES), rtol=1e-6)


def test_gradients_flow_through_a_jitted_distribution_argument():
    def log_lik(dist):
        return dist.log_prob(VALUES).sum()

    dist = loc_scale(coef((N,)), jnp.linspace(0.0, 1.0, N))
    grads = jax.jit(jax.grad(log_lik))(dist)
    assert isinstance(grads, ptm.LocScaleTransformationDist)
    assert np.isfinite(grads.coef).all()


def test_loc_scale_parameter_properties():
    properties = ptm.LocScaleTransformationDist.parameter_properties()
    assert properties.keys() == {"coef", "loc", "scale"}
    assert properties["coef"].event_ndims == 1


def test_shared_coef_stays_shared_when_sliced():
    # Expanding it to one row each would recompute the spline moments per row.
    dist = loc_scale(coef(), jnp.linspace(0.0, 1.0, N))
    assert dist[INDICES].coef.shape == (NPARAM,)
