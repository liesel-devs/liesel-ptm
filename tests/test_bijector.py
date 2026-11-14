import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
from liesel_ptm.bspline import OnionKnots, OnionSpline, PTMKnots, PTMSpline

NPARAM = 10
N = 7


def random_walk_coef(key, batch_shape: tuple[int, ...], scale=0.35):
    innovations = jax.random.normal(key, batch_shape + (NPARAM,)) * scale
    return jnp.cumsum(innovations, axis=-1)


SPLINES = {
    "onion": OnionSpline.from_bounds(-4.0, 4.0, nparam=NPARAM),
    "ptm": PTMSpline.from_bounds(-4.0, 4.0, nparam=NPARAM),
}


@pytest.fixture(params=list(SPLINES), ids=list(SPLINES))
def spline(request):
    return SPLINES[request.param]


NORMALIZATION = {
    "raw": {},
    "centered": {"centered": True},
    "scaled": {"scaled": True},
    "centered_scaled": {"centered": True, "scaled": True},
}


@pytest.fixture(params=list(NORMALIZATION), ids=list(NORMALIZATION))
def normalization(request):
    return NORMALIZATION[request.param]


COEF_SHAPES = {"shared": (), "batched": (3,), "per_observation": (N,)}


@pytest.fixture(params=list(COEF_SHAPES), ids=list(COEF_SHAPES))
def coef(request):
    return random_walk_coef(jax.random.key(1), COEF_SHAPES[request.param])


class TestConstruction:
    def test_spline_returns_its_bijector(self):
        coef = random_walk_coef(jax.random.key(0), ())
        onion = SPLINES["onion"].bijector(coef)
        ptm_bijector = SPLINES["ptm"].bijector(coef, centered=True)

        assert type(onion) is ptm.OnionSplineBijector
        assert type(ptm_bijector) is ptm.SplineBijector
        assert isinstance(onion, ptm.SplineBijector)
        assert isinstance(onion, tfb.Bijector)
        assert ptm_bijector.centered and not ptm_bijector.scaled
        assert onion.bspline is SPLINES["onion"]

    def test_plain_constructor(self, spline):
        coef = random_walk_coef(jax.random.key(0), ())
        bijector = ptm.SplineBijector(coef, bspline=spline, scaled=True)
        expected = spline.bijector(coef, scaled=True)
        y = jnp.linspace(-3.0, 3.0, 5)
        np.testing.assert_allclose(bijector.inverse(y), expected.inverse(y))

    def test_from_bounds_matches_knots(self):
        onion = OnionSpline.from_bounds(-3.0, 2.0, nparam=8, order=3)
        np.testing.assert_array_equal(
            onion.knots, OnionKnots(-3.0, 2.0, nparam=8, order=3).knots
        )
        ptm_spline = PTMSpline.from_bounds(-3.0, 2.0, nparam=8, eps=0.2)
        np.testing.assert_array_equal(
            ptm_spline.knots, PTMKnots(-3.0, 2.0, nparam=8).knots
        )
        assert ptm_spline.transition_width == pytest.approx(
            0.2 * (ptm_spline.max_knot - ptm_spline.min_knot)
        )
        assert type(onion) is OnionSpline and type(ptm_spline) is PTMSpline

    def test_ptm_spline_is_exported(self):
        assert ptm.PTMSpline is PTMSpline

    def test_has_no_full_moments(self, spline):
        bijector = spline.bijector(random_walk_coef(jax.random.key(0), ()))
        assert not hasattr(bijector, "mean")
        assert not hasattr(bijector, "variance")

    def test_rejects_integer_coef(self, spline):
        with pytest.raises(TypeError, match="floating"):
            spline.bijector(jnp.zeros(NPARAM, dtype=jnp.int32))

    def test_rejects_batched_reference_for_normalization(self, spline):
        coef = random_walk_coef(jax.random.key(0), ())
        reference = tfd.Normal(jnp.zeros(3), 1.0)
        with pytest.raises(ValueError, match="batch shape"):
            spline.bijector(coef, centered=True, reference_distribution=reference)
        bijector = spline.bijector(coef, reference_distribution=reference)
        with pytest.raises(ValueError, match="batch shape"):
            bijector.mean_raw()

    @pytest.mark.parametrize(
        "order,error", [(0, ValueError), (-1, ValueError), (1.5, TypeError)]
    )
    def test_rejects_invalid_quadrature_order(self, spline, order, error):
        coef = random_walk_coef(jax.random.key(0), ())
        with pytest.raises(error, match="gauss_legendre_order"):
            spline.bijector(coef, gauss_legendre_order=order)


class TestBijection:
    def test_inverse_recovers_forward(self, spline, normalization, coef):
        bijector = spline.bijector(coef, **normalization)
        z = jnp.linspace(-6.0, 6.0, 4 * N).reshape(4, N)
        if coef.shape[:-1] == (3,):
            z = z[:, :3]
        # Copy, so TFP's bijector cache does not return z itself.
        x = jnp.array(bijector.forward(z), copy=True)
        np.testing.assert_allclose(bijector.inverse(x), z, atol=1e-4)

    def test_inverse_log_det_jacobian_matches_autodiff(
        self, spline, normalization, coef
    ):
        bijector = spline.bijector(coef, **normalization)
        y = jnp.linspace(-6.0, 6.0, 11)
        y = jnp.broadcast_to(
            jnp.reshape(y, (11,) + (1,) * (coef.ndim - 1)), (11,) + coef.shape[:-1]
        )

        # The bijector acts elementwise, so the gradient of the sum is h'.
        deriv = jax.grad(lambda y: jnp.sum(bijector.inverse(y)))(y)
        np.testing.assert_allclose(
            bijector.inverse_log_det_jacobian(y), jnp.log(deriv), rtol=1e-4, atol=1e-4
        )

    def test_forward_log_det_jacobian_is_consistent(self, spline):
        bijector = spline.bijector(
            random_walk_coef(jax.random.key(2), ()), centered=True, scaled=True
        )
        z = jnp.linspace(-3.0, 3.0, 9)
        np.testing.assert_allclose(
            bijector.forward_log_det_jacobian(z),
            -bijector.inverse_log_det_jacobian(
                jnp.array(bijector.forward(z), copy=True)
            ),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_nan_is_masked(self, spline):
        bijector = spline.bijector(random_walk_coef(jax.random.key(2), ()))
        y = jnp.array([jnp.nan, 0.5])
        assert jnp.isnan(bijector.inverse(y)[0])
        assert jnp.isnan(bijector.inverse_log_det_jacobian(y)[0])
        assert jnp.isfinite(bijector.inverse_log_det_jacobian(y)[1])

    def test_composes_with_tfp_bijectors(self, spline):
        coef = random_walk_coef(jax.random.key(3), ())
        spline_bijector = spline.bijector(coef, centered=True)
        dist = tfd.TransformedDistribution(
            tfd.Normal(0.0, 1.0), tfb.Chain([tfb.Exp(), spline_bijector])
        )
        t = jnp.linspace(0.1, 5.0, 6)
        expected = (
            tfd.Normal(0.0, 1.0).log_prob(spline_bijector.inverse(jnp.log(t)))
            + spline_bijector.inverse_log_det_jacobian(jnp.log(t))
            - jnp.log(t)
        )
        np.testing.assert_allclose(dist.log_prob(t), expected, rtol=1e-5)


class TestShapes:
    def test_batch_shape(self, spline):
        coef = random_walk_coef(jax.random.key(4), (2, 3))
        bijector = spline.bijector(coef)
        assert bijector.experimental_batch_shape() == (2, 3)
        assert bijector.inverse(jnp.zeros((5, 2, 3))).shape == (5, 2, 3)
        assert bijector.inverse(0.5).shape == (2, 3)
        assert bijector.forward(jnp.zeros((5, 1, 3))).shape == (5, 2, 3)

    def test_slicing(self, spline):
        coef = random_walk_coef(jax.random.key(4), (N,))
        bijector = spline.bijector(coef, centered=True, scaled=True)
        rows = jnp.array([4, 0, 0])
        sliced = bijector[rows]

        assert type(sliced) is type(bijector)
        assert sliced.experimental_batch_shape() == (3,)
        assert sliced.centered and sliced.scaled
        y = jnp.linspace(-2.0, 2.0, N)
        np.testing.assert_allclose(sliced.inverse(y[rows]), bijector.inverse(y)[rows])
        np.testing.assert_allclose(sliced.mean_raw(), bijector.mean_raw()[rows])

    def test_passes_into_jit(self, spline, normalization):
        coef = random_walk_coef(jax.random.key(5), (3,))
        bijector = spline.bijector(coef, **normalization)
        y = jnp.linspace(-2.0, 2.0, 3)

        @jax.jit
        def evaluate(bijector, y):
            return (
                bijector.inverse(y),
                bijector.inverse_log_det_jacobian(y),
                bijector.forward(y),
            )

        for jitted, eager in zip(
            evaluate(bijector, y),
            (
                bijector.inverse(y),
                bijector.inverse_log_det_jacobian(y),
                bijector.forward(y),
            ),
            strict=True,
        ):
            np.testing.assert_allclose(jitted, eager, rtol=1e-5, atol=1e-6)

    def test_float64_coef_keeps_float64(self, spline):
        with jax.enable_x64(True):
            coef = jnp.asarray(random_walk_coef(jax.random.key(6), ()), jnp.float64)
            bijector = spline.bijector(coef, centered=True, scaled=True)
            assert bijector.dtype == jnp.float64
            assert bijector.inverse(0.5).dtype == jnp.float64
            assert bijector.forward(0.5).dtype == jnp.float64
            assert bijector.mean_raw().dtype == jnp.float64

            dist = ptm.LocScaleTransformationDist(
                coef, loc=jnp.zeros(3), scale=1.0, bspline=spline, centered=True
            )
            assert dist.dtype == jnp.float64
            assert dist.log_prob(jnp.float32(0.5)).dtype == jnp.float64
            assert dist.sample(2, seed=jax.random.key(0)).dtype == jnp.float64
            assert dist.spline_bijector.dtype == jnp.float64


class TestMatchesDistribution:
    """The bijector reproduces the spline part of TransformationDist."""

    def test_spline_only_distribution(self, spline, normalization, coef):
        dist = ptm.TransformationDist(coef, bspline=spline, **normalization)
        bijector = spline.bijector(coef, **normalization)
        reference = tfd.Normal(0.0, 1.0)
        batch = coef.shape[:-1] or (N,)
        y = jnp.linspace(-6.0, 6.0, int(np.prod(batch))).reshape(batch)
        p = jnp.linspace(0.01, 0.99, int(np.prod(batch))).reshape(batch)

        np.testing.assert_allclose(
            dist.log_prob(y),
            reference.log_prob(bijector.inverse(y))
            + bijector.inverse_log_det_jacobian(y),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            dist.cdf(y), reference.cdf(bijector.inverse(y)), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            dist.quantile(p),
            bijector.forward(reference.quantile(p)),
            rtol=1e-4,
            atol=1e-4,
        )
        transformed = tfd.TransformedDistribution(reference, bijector)
        np.testing.assert_allclose(
            dist.log_prob(y), transformed.log_prob(y), rtol=1e-5, atol=1e-5
        )

    def test_loc_scale_distribution(self, spline, normalization, coef):
        loc = jnp.linspace(-1.0, 1.0, N)
        scale = jnp.linspace(0.5, 2.0, N)
        if coef.shape[:-1] == (3,):
            coef = coef[:1]
        dist = ptm.LocScaleTransformationDist(
            coef, loc=loc, scale=scale, bspline=spline, **normalization
        )
        bijector = spline.bijector(coef, **normalization)
        y = jnp.linspace(-6.0, 6.0, N)
        r = (y - loc) / scale

        np.testing.assert_allclose(
            dist.log_prob(y),
            tfd.Normal(0.0, 1.0).log_prob(bijector.inverse(r))
            + bijector.inverse_log_det_jacobian(r)
            - jnp.log(scale),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_spline_bijector_property(self, spline, normalization):
        coef = random_walk_coef(jax.random.key(7), ())
        reference = tfd.Normal(0.0, 1.0)
        dist = ptm.LocScaleTransformationDist(
            coef,
            loc=jnp.zeros(N),
            scale=1.0,
            bspline=spline,
            reference_distribution=reference,
            gauss_legendre_order=6,
            integration_bounds=(-5.0, 5.0),
            **normalization,
        )
        bijector = dist.spline_bijector

        assert type(bijector) is type(spline.bijector(coef))
        np.testing.assert_array_equal(bijector.coef, dist.coef)
        assert bijector.bspline is spline
        assert bijector.reference_distribution is reference
        assert bijector.centered == dist.centered
        assert bijector.scaled == dist.scaled
        assert bijector.gauss_legendre_order == 6
        assert bijector.integration_bounds == (-5.0, 5.0)

    def test_raw_moments_match_distribution(self, spline, coef):
        dist = ptm.TransformationDist(coef, bspline=spline)
        bijector = spline.bijector(coef)

        np.testing.assert_allclose(
            bijector.mean_raw(), dist.transformation_spline_mean(), rtol=1e-6
        )
        np.testing.assert_allclose(
            bijector.variance_raw(), dist.transformation_spline_variance(), rtol=1e-6
        )
        mean = jnp.zeros(coef.shape[:-1])
        np.testing.assert_allclose(
            bijector.variance_raw(mean),
            dist.transformation_spline_variance(mean),
            rtol=1e-6,
        )
        assert bijector.mean_raw().shape == coef.shape[:-1]

    def test_raw_moments_ignore_normalization(self, spline):
        coef = random_walk_coef(jax.random.key(8), ())
        raw = spline.bijector(coef)
        normalized = spline.bijector(coef, centered=True, scaled=True)
        np.testing.assert_allclose(normalized.mean_raw(), raw.mean_raw())
        np.testing.assert_allclose(normalized.variance_raw(), raw.variance_raw())

    def test_normalized_bijector_is_centered_and_scaled(self, spline):
        coef = random_walk_coef(jax.random.key(8), ())
        bijector = spline.bijector(coef, centered=True, scaled=True)
        mean, sd = bijector.mean_raw(), jnp.sqrt(bijector.variance_raw())
        y = jnp.linspace(-2.0, 2.0, 5)
        raw = spline.bijector(coef)
        np.testing.assert_allclose(
            bijector.inverse(y), raw.inverse(y * sd + mean), rtol=1e-6
        )

    def test_moment_quadrature_diagnostic_matches_distribution(self, spline):
        coef = random_walk_coef(jax.random.key(9), (3,))
        dist = ptm.TransformationDist(coef, bspline=spline)
        diagnostic = spline.bijector(coef).moment_quadrature_diagnostic()
        expected = dist.moment_quadrature_diagnostic()

        assert diagnostic.keys() == expected.keys()
        for key, value in expected.items():
            np.testing.assert_allclose(
                diagnostic[key], value, rtol=1e-5, atol=1e-6, err_msg=key
            )
        assert bool(diagnostic["ok"])
        with pytest.raises(ValueError, match="reference_order"):
            spline.bijector(coef).moment_quadrature_diagnostic(reference_order=8)

    def test_sampling_stream_uses_quantiles_of_uniforms(self, spline):
        coef = random_walk_coef(jax.random.key(10), ())
        dist = ptm.LocScaleTransformationDist(
            coef,
            loc=jnp.linspace(-1.0, 1.0, N),
            scale=2.0,
            bspline=spline,
            centered=True,
            scaled=True,
        )
        seed = jax.random.key(11)
        eps = jnp.finfo(dist.dtype).eps
        u = jax.random.uniform(seed, (4, N), minval=eps, maxval=1.0 - eps)
        np.testing.assert_array_equal(dist.sample(4, seed=seed), dist.quantile(u))


class TestOnionSplineBijector:
    spline = SPLINES["onion"]

    def bijector(self, batch_shape=(), **kwargs):
        coef = random_walk_coef(jax.random.key(12), batch_shape, scale=0.6)
        return self.spline.bijector(coef, **kwargs)

    def test_core_maps_onto_itself(self):
        bijector = self.bijector((3,))
        lo, hi = float(self.spline.min_knot), float(self.spline.max_knot)
        knots = jnp.array([[lo], [hi]])
        np.testing.assert_allclose(
            bijector.inverse(knots), jnp.broadcast_to(knots, (2, 3)), atol=1e-5
        )
        np.testing.assert_array_equal(
            bijector.forward(knots), jnp.broadcast_to(knots, (2, 3))
        )

    def test_tails_are_exactly_identity(self):
        bijector = self.bijector((3,))
        lo, hi = float(self.spline.min_knot), float(self.spline.max_knot)
        tails = jnp.array([lo - 10.0, lo - 1e-3, hi + 1e-3, hi + 10.0])[:, None]
        tails = jnp.broadcast_to(tails, (4, 3))

        np.testing.assert_array_equal(bijector.forward(tails), tails)
        np.testing.assert_array_equal(bijector.inverse(tails), tails)
        np.testing.assert_array_equal(bijector.inverse_log_det_jacobian(tails), 0.0)

    def test_forward_is_accurate_in_core(self):
        bijector = self.bijector((3,))
        x = jnp.linspace(float(self.spline.min_knot), float(self.spline.max_knot), 101)[
            :, None
        ]
        z = jnp.array(bijector.inverse(x), copy=True)
        np.testing.assert_allclose(
            bijector.forward(z), jnp.broadcast_to(x, z.shape), atol=1e-4
        )

    def test_forward_has_no_while_loop(self):
        bijector = self.bijector((3,))
        jaxpr = jax.make_jaxpr(bijector.forward)(jnp.zeros((5, 3)))
        assert "while" not in str(jaxpr)
