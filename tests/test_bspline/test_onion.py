import jax
import jax.numpy as jnp
import pytest

from liesel_ptm.bspline.onion import OnionKnots, OnionSpline

knots = OnionKnots(-4.0, 4.0, nparam=10)
coef = jax.random.normal(jax.random.key(1), (1, knots.nparam))
bs = OnionSpline(knots.knots)


class TestLowLevelContraction:
    def test_shared_unbatched_basis_with_batched_shared_coef(self):
        x = jnp.linspace(-2.0, 2.0, 5)
        raw_coef = jax.random.normal(jax.random.key(1), (2, 1, knots.nparam))
        constrained_coef = bs.compute_coef(raw_coef)

        fx, fxd = bs.bspline.dot_and_deriv_n(x, constrained_coef)

        assert fx.shape == (2, 5)
        assert fxd.shape == (2, 5)
        assert jnp.all(jnp.isfinite(fx))
        assert jnp.all(jnp.isfinite(fxd))

    def test_rowwise_unbatched_basis_with_batched_rowwise_coef(self):
        x = jnp.linspace(-2.0, 2.0, 5)
        raw_coef = jax.random.normal(jax.random.key(2), (2, 5, knots.nparam))
        constrained_coef = bs.compute_coef(raw_coef)

        fx, fxd = bs.bspline.dot_and_deriv_n(x, constrained_coef)

        assert fx.shape == (2, 5)
        assert fxd.shape == (2, 5)
        assert jnp.all(jnp.isfinite(fx))
        assert jnp.all(jnp.isfinite(fxd))

    def test_mismatched_rowwise_coef_axis_raises(self):
        x = jnp.linspace(-2.0, 2.0, 5)
        raw_coef = jax.random.normal(jax.random.key(3), (2, 4, knots.nparam))
        constrained_coef = bs.compute_coef(raw_coef)

        with pytest.raises(ValueError, match="n_coef"):
            bs.bspline.dot_and_deriv_n(x, constrained_coef)


class TestDotAndDeriv:
    def test_scalar_x(self):
        fx, fxd = bs.dot_and_deriv(1.0, coef)

        # shape fits
        assert fx.shape == ()
        assert fxd.shape == ()

        # no nans
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

        # increasing
        assert jnp.all(fxd > 0.0)

    def test_rejects_legacy_coef_shape(self):
        legacy_coef = jax.random.normal(jax.random.key(1), (knots.nparam,))

        with pytest.raises(ValueError, match="n_coef"):
            bs.dot_and_deriv(1.0, legacy_coef)

    def test_rowwise_coef_requires_matching_observation_axis(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        coef = jax.random.normal(jax.random.key(1), (4, knots.nparam))

        with pytest.raises(ValueError, match="n_coef"):
            bs.dot_and_deriv(jnp.zeros((3,)), coef)

        with pytest.raises(ValueError, match="n_coef"):
            bs.dot_inverse(jnp.zeros((3,)), coef)

    def test_subscripts_argument_is_accepted_as_compatibility_noop(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs_default = OnionSpline(knots.knots)
        bs_compat = OnionSpline(knots.knots, subscripts="...nj,...nj->...n")
        x = jnp.linspace(-2.0, 2.0, 5)
        coef = jax.random.normal(jax.random.key(2), (5, knots.nparam))

        fx_default, fxd_default = bs_default.dot_and_deriv(x, coef)
        fx_compat, fxd_compat = bs_compat.dot_and_deriv(x, coef)

        assert jnp.allclose(fx_default, fx_compat)
        assert jnp.allclose(fxd_default, fxd_compat)

    def test_vector_x(self):
        x = jnp.linspace(-8.0, 8.0, 300)
        fx, fxd = bs.dot_and_deriv(x, coef)

        # shape fits
        assert fx.shape == x.shape
        assert fxd.shape == x.shape

        # no nans
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

        # increasing
        assert jnp.all(fxd > 0.0)
        assert jnp.all(jnp.diff(fx) > 0.0)

    def test_vector_x_long(self):
        x = jnp.linspace(-8.0, 8.0, 10_000)
        fx, fxd = bs.dot_and_deriv(x, coef)

        # shape fits
        assert fx.shape == x.shape
        assert fxd.shape == x.shape

        # no nans
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

        # increasing
        assert jnp.all(fxd > 0.0)
        assert jnp.all(jnp.diff(fx) > 0.0)

    def test_batched_x_single_coef(self):
        x = jax.random.normal(jax.random.key(1), (4, 200))
        fx, fxd = bs.dot_and_deriv(x, coef)

        # shape fits
        assert fx.shape == (4, 200)
        assert fxd.shape == (4, 200)

        # no nans
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

        # increasing
        assert jnp.all(fxd > 0.0)

    def test_2batched_x_single_coef(self):
        x = jax.random.normal(jax.random.key(1), (3, 4, 200))
        fx, fxd = bs.dot_and_deriv(x, coef)

        # shape fits
        assert fx.shape == (3, 4, 200)
        assert fxd.shape == (3, 4, 200)

        # no nans
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

        # increasing
        assert jnp.all(fxd > 0.0)

    def test_scalar_x_batched_coef(self):
        x = 1.0
        coef = jax.random.normal(jax.random.key(1), (3, 1, knots.nparam))
        fx, fxd = bs.dot_and_deriv(x, coef)

        # shape fits
        assert fx.shape == (3,)
        assert fxd.shape == (3,)

        # no nans
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

        # increasing
        assert jnp.all(fxd > 0.0)

    def test_vector_x_batched_coef(self):
        x = jax.random.normal(jax.random.key(1), (200,))
        coef = jax.random.normal(jax.random.key(1), (3, 1, knots.nparam))
        fx, fxd = bs.dot_and_deriv(x, coef)

        # shape fits
        assert fx.shape == (3, 200)
        assert fxd.shape == (3, 200)

        # no nans
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

        # increasing
        assert jnp.all(fxd > 0.0)

    def test_batched_x_batched_coef(self):
        x = jax.random.normal(
            jax.random.key(1),
            (
                3,
                200,
            ),
        )
        coef = jax.random.normal(jax.random.key(1), (3, 1, knots.nparam))
        fx, fxd = bs.dot_and_deriv(x, coef)

        # shape fits
        assert fx.shape == (3, 200)
        assert fxd.shape == (3, 200)

        # no nans
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

        # increasing
        assert jnp.all(fxd > 0.0)

    def test_2batched_x_2batched_coef(self):
        x = jax.random.normal(
            jax.random.key(1),
            (
                2,
                3,
                200,
            ),
        )
        coef = jax.random.normal(jax.random.key(1), (2, 3, 1, knots.nparam))
        fx, fxd = bs.dot_and_deriv(x, coef)

        # shape fits
        assert fx.shape == (2, 3, 200)
        assert fxd.shape == (2, 3, 200)

        # no nans
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

        # increasing
        assert jnp.all(fxd > 0.0)


class TestDotAndDerivInverse:
    def test_scalar_x(self):
        fx, _ = bs.dot_and_deriv(1.0, coef)
        x = bs.dot_inverse(fx, coef)

        # shape fits
        assert x.shape == ()

        # no nans
        assert not jnp.any(jnp.isnan(x))

        assert jnp.allclose(x, 1.0, atol=1e-5)

    def test_vector_x(self):
        x = jnp.linspace(-8.0, 8.0, 300)
        fx, _ = bs.dot_and_deriv(x, coef)

        x2 = bs.dot_inverse(fx, coef)

        # shape fits
        assert x2.shape == (300,)

        # no nans
        assert not jnp.any(jnp.isnan(x2))

        assert jnp.allclose(x2, x, atol=1e-4)

    def test_batched_x_single_coef(self):
        x = jax.random.normal(jax.random.key(1), (4, 200))
        fx, _ = bs.dot_and_deriv(x, coef)

        x2 = bs.dot_inverse(fx, coef)

        # shape fits
        assert x2.shape == x.shape

        # no nans
        assert not jnp.any(jnp.isnan(x2))

        assert jnp.allclose(x2, x, atol=1e-4)

    def test_2batched_x_single_coef(self):
        x = jax.random.normal(jax.random.key(1), (3, 4, 200))
        fx, _ = bs.dot_and_deriv(x, coef)

        x2 = bs.dot_inverse(fx, coef)

        # shape fits
        assert x2.shape == x.shape

        # no nans
        assert not jnp.any(jnp.isnan(x2))

        assert jnp.allclose(x2, x, atol=1e-4)

    def test_scalar_x_batched_coef(self):
        x = 1.0
        coef = jax.random.normal(jax.random.key(1), (3, 1, knots.nparam))
        fx, _ = bs.dot_and_deriv(x, coef)

        x2 = bs.dot_inverse(fx, coef)

        # shape fits
        assert x2.shape == (3,)

        # no nans
        assert not jnp.any(jnp.isnan(x2))

        assert jnp.allclose(x2, x, atol=1e-4)

    def test_vector_x_batched_coef(self):
        x = jax.random.normal(jax.random.key(1), (200,))
        coef = jax.random.normal(jax.random.key(1), (3, 1, knots.nparam))
        fx, _ = bs.dot_and_deriv(x, coef)

        x2 = bs.dot_inverse(fx, coef)

        # shape fits
        assert x2.shape == (3, 200)

        # no nans
        assert not jnp.any(jnp.isnan(x2))

        for i in range(x2.shape[0]):
            assert jnp.allclose(x2[i, ...], x, atol=1e-4)

    def test_batched_x_batched_coef(self):
        x = jax.random.normal(
            jax.random.key(1),
            (
                3,
                200,
            ),
        )
        coef = jax.random.normal(jax.random.key(1), (3, 1, knots.nparam))
        fx, _ = bs.dot_and_deriv(x, coef)

        x2 = bs.dot_inverse(fx, coef)

        # shape fits
        assert x2.shape == x.shape

        # no nans
        assert not jnp.any(jnp.isnan(x2))

        for i in range(x2.shape[0]):
            assert jnp.allclose(x2[i, ...], x[i, ...], atol=1e-4)

    def test_2batched_x_2batched_coef(self):
        x = jax.random.normal(
            jax.random.key(1),
            (
                2,
                3,
                200,
            ),
        )
        coef = jax.random.normal(jax.random.key(1), (2, 3, 1, knots.nparam))
        fx, _ = bs.dot_and_deriv(x, coef)

        x2 = bs.dot_inverse(fx, coef)

        # shape fits
        assert x2.shape == x.shape

        # no nans
        assert not jnp.any(jnp.isnan(x2))

        for i in range(x2.shape[0]):
            for j in range(x2.shape[1]):
                assert jnp.allclose(x2[i, j, ...], x[i, j, ...], atol=1e-4)


class TestDotAndDerivNFullBatch:
    def test_scalar_x(self):
        fx, fxd = bs.dot_and_deriv_n_fullbatch(1.0, coef)

        # shape fits
        assert fx.shape == ()
        assert fxd.shape == ()

        # no nans
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

        # increasing
        assert jnp.all(fxd > 0.0)

    def test_vector_x(self):
        x = jnp.linspace(-8.0, 8.0, 300)
        fx, fxd = bs.dot_and_deriv_n_fullbatch(x, coef)

        # shape fits
        assert fx.shape == x.shape
        assert fxd.shape == x.shape

        # no nans
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

        # increasing
        assert jnp.all(fxd > 0.0)
        assert jnp.all(jnp.diff(fx) > 0.0)

    def test_batched_x_single_coef(self):
        x = jax.random.normal(jax.random.key(1), (4, 200))
        fx, fxd = bs.dot_and_deriv_n_fullbatch(x, coef)

        # shape fits
        assert fx.shape == (4, 200)
        assert fxd.shape == (4, 200)

        # no nans
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))

        # increasing
        assert jnp.all(fxd > 0.0)

    def test_scalar_x_batched_coef(self):
        x = 1.0
        coef = jax.random.normal(jax.random.key(1), (3, 1, knots.nparam))
        bs.dot_and_deriv_n_fullbatch(x, coef)

    def test_vector_x_batched_coef(self):
        x = jax.random.normal(jax.random.key(1), (200,))
        coef = jax.random.normal(jax.random.key(1), (3, 1, knots.nparam))
        bs.dot_and_deriv_n_fullbatch(x, coef)

    def test_gptm_dot_and_deriv(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        k1 = jax.random.key(1)
        k1, k2 = jax.random.split(k1)
        bs = OnionSpline(knots.knots)

        n = 17

        coef = jax.random.normal(k1, (n, knots.nparam))
        x = jax.random.normal(k2, (n,))

        dot, deriv = bs.dot_and_deriv_n_fullbatch(x, coef)
        assert dot.shape == (n,)
        assert deriv.shape == (n,)

        b = 13
        coef = jax.random.normal(k1, (b, n, knots.nparam))

        dot, deriv = bs.dot_and_deriv_n_fullbatch(x, coef)

        assert dot.shape == (b, n)
        assert deriv.shape == (b, n)

        coef = jax.random.normal(k1, (1, b, n, knots.nparam))

        dot, deriv = bs.dot_and_deriv_n_fullbatch(x, coef)

        assert dot.shape == (1, b, n)
        assert deriv.shape == (1, b, n)

    def test_gptm_inverse(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        k1 = jax.random.key(1)
        k1, k2 = jax.random.split(k1)
        bs = OnionSpline(knots.knots)

        n = 17

        coef = jax.random.normal(k1, (n, knots.nparam))
        x = jax.random.normal(k2, (n,))

        dot = bs.dot_inverse_n_fullbatch(x, coef)
        assert dot.shape == (n,)

        b = 13
        coef = jax.random.normal(k1, (b, n, knots.nparam))

        dot = bs.dot_inverse_n_fullbatch(x, coef)

        assert dot.shape == (b, n)

        coef = jax.random.normal(k1, (1, b, n, knots.nparam))
        dot = bs.dot_inverse_n_fullbatch(x, coef)

        assert dot.shape == (1, b, n)


class TestPublicMethodEquivalence:
    def test_forward_methods_agree_rowwise(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        n = 5
        x = jnp.linspace(-2.0, 2.0, n)
        coef = jax.random.normal(jax.random.key(2), (n, knots.nparam))

        fx, fxd = bs.dot_and_deriv(x, coef)
        fx_n, fxd_n = bs.dot_and_deriv_n(x, coef)
        fx_full, fxd_full = bs.dot_and_deriv_n_fullbatch(x, coef)

        assert jnp.allclose(fx, fx_n)
        assert jnp.allclose(fxd, fxd_n)
        assert jnp.allclose(fx, fx_full)
        assert jnp.allclose(fxd, fxd_full)

    def test_inverse_methods_agree_rowwise(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        n = 5
        x = jnp.linspace(-2.0, 2.0, n)
        coef = jax.random.normal(jax.random.key(2), (n, knots.nparam))
        fx, _ = bs.dot_and_deriv(x, coef)

        x_inv = bs.dot_inverse(fx, coef)
        x_inv_n = bs.dot_inverse_n(fx, coef)
        x_inv_full = bs.dot_inverse_n_fullbatch(fx, coef)

        assert x_inv.shape == x.shape
        assert jnp.allclose(x_inv, x, atol=1e-4)
        assert jnp.allclose(x_inv, x_inv_n)
        assert jnp.allclose(x_inv, x_inv_full)

    def test_jvp_through_rowwise_forward(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        n = 5
        x = jnp.linspace(-2.0, 2.0, n)
        coef = jax.random.normal(jax.random.key(2), (n, knots.nparam))

        def fn(value):
            fx, fxd = bs.dot_and_deriv(value, coef)
            return jnp.sum(fx + 0.01 * fxd)

        primal, tangent = jax.jvp(fn, (x,), (jnp.ones_like(x),))

        assert jnp.isfinite(primal)
        assert jnp.isfinite(tangent)

    def test_grad_through_shared_coef_forward(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        x = jnp.linspace(-2.0, 2.0, 5)
        coef = jax.random.normal(jax.random.key(2), (3, 1, knots.nparam))

        def fn(raw_coef):
            fx, fxd = bs.dot_and_deriv(x, raw_coef)
            return jnp.sum(fx + 0.01 * fxd)

        grad = jax.grad(fn)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_through_rowwise_coef_forward(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        n = 5
        x = jnp.linspace(-2.0, 2.0, n)
        coef = jax.random.normal(jax.random.key(2), (n, knots.nparam))

        def fn(raw_coef):
            fx, fxd = bs.dot_and_deriv(x, raw_coef)
            return jnp.sum(fx + 0.01 * fxd)

        grad = jax.grad(fn)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))


class TestTfpLayout:
    @staticmethod
    def _full_broadcast_forward_reference(bs, value, coef, batch_shape=None):
        batch_shape = bs._tfp_batch_shape(coef, batch_shape)
        value, sample_shape, result_batch_shape = bs._broadcast_tfp_value(
            value, batch_shape
        )

        value = bs._tfp_to_legacy_batch_last(
            value, result_batch_shape, sample_shape
        )
        legacy_coef = jnp.expand_dims(coef, axis=-2)
        fx, fxd = bs.dot_and_deriv(value, legacy_coef)

        fx = bs._legacy_batch_last_to_tfp(fx, result_batch_shape, sample_shape)
        fxd = bs._legacy_batch_last_to_tfp(fxd, result_batch_shape, sample_shape)

        return fx, fxd

    @staticmethod
    def _full_broadcast_inverse_reference(bs, value, coef, batch_shape=None):
        batch_shape = bs._tfp_batch_shape(coef, batch_shape)
        value, sample_shape, result_batch_shape = bs._broadcast_tfp_value(
            value, batch_shape
        )

        value = bs._tfp_to_legacy_batch_last(
            value, result_batch_shape, sample_shape
        )
        legacy_coef = jnp.expand_dims(coef, axis=-2)
        inverse = bs.dot_inverse(value, legacy_coef)

        return bs._legacy_batch_last_to_tfp(
            inverse, result_batch_shape, sample_shape
        )

    def _assert_tfp_roundtrip(
        self,
        value,
        coef,
        expected_shape,
        batch_shape=None,
    ):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)

        fx, fxd = bs.dot_and_deriv_tfp(value, coef, batch_shape=batch_shape)

        assert fx.shape == expected_shape
        assert fxd.shape == expected_shape
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))
        assert jnp.all(fxd > 0.0)

        fx_ref, fxd_ref = self._full_broadcast_forward_reference(
            bs, value, coef, batch_shape=batch_shape
        )
        assert jnp.allclose(fx, fx_ref)
        assert jnp.allclose(fxd, fxd_ref)

        x = bs.dot_inverse_tfp(fx, coef, batch_shape=batch_shape)
        x_ref = self._full_broadcast_inverse_reference(
            bs, fx, coef, batch_shape=batch_shape
        )

        assert x.shape == expected_shape
        assert not jnp.any(jnp.isnan(x))
        assert jnp.allclose(x, x_ref, atol=1e-4)

        expected = jnp.broadcast_to(jnp.asarray(value), expected_shape)
        assert jnp.allclose(x, expected, atol=1e-4)

    def test_single_coef(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        coef = jax.random.normal(jax.random.key(8), (knots.nparam,))

        self._assert_tfp_roundtrip(0.5, coef, ())
        self._assert_tfp_roundtrip(jnp.linspace(-2.0, 2.0, 5), coef, (5,))

    def test_batched_coef(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        n = 4
        coef = jax.random.normal(jax.random.key(2), (n, knots.nparam))

        self._assert_tfp_roundtrip(0.5, coef, (n,))
        self._assert_tfp_roundtrip(
            jnp.linspace(-2.0, 2.0, 5).reshape((5, 1)),
            coef,
            (5, n),
        )

    def test_batch_one_coef_broadcasts_observations(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        coef = jax.random.normal(jax.random.key(7), (1, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5)

        self._assert_tfp_roundtrip(value, coef, (5,))

    def test_two_dimensional_batched_coef(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        n = 4
        coef = jax.random.normal(jax.random.key(3), (2, n, knots.nparam))

        self._assert_tfp_roundtrip(0.5, coef, (2, n))
        self._assert_tfp_roundtrip(
            jnp.linspace(-2.0, 2.0, 5).reshape((5, 1, 1)),
            coef,
            (5, 2, n),
        )
        self._assert_tfp_roundtrip(
            jnp.broadcast_to(
                jnp.linspace(-2.0, 2.0, n),
                (5, 1, n),
            ),
            coef,
            (5, 2, n),
        )

    def test_inverse_tail_values_match_full_broadcast_reference(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        n = 4
        coef = jax.random.normal(jax.random.key(6), (2, n, knots.nparam))
        value = jnp.asarray([-8.0, -4.0, 0.0, 4.0, 8.0]).reshape((5, 1, 1))

        x = bs.dot_inverse_tfp(value, coef, batch_shape=(2, n))
        x_ref = self._full_broadcast_inverse_reference(
            bs, value, coef, batch_shape=(2, n)
        )

        assert x.shape == (5, 2, n)
        assert jnp.allclose(x, x_ref, atol=1e-4)

    def test_jvp_through_rowwise_shared_grid_tfp(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        n = 4
        coef = jax.random.normal(jax.random.key(4), (2, n, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1, 1))

        def fn(value):
            fx, fxd = bs.dot_and_deriv_tfp(value, coef, batch_shape=(2, n))
            return jnp.sum(fx + 0.01 * fxd)

        primal, tangent = jax.jvp(fn, (value,), (jnp.ones_like(value),))

        assert jnp.isfinite(primal)
        assert jnp.isfinite(tangent)

    def test_grad_through_rowwise_shared_grid_tfp_coef(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        n = 4
        coef = jax.random.normal(jax.random.key(5), (2, n, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1, 1))

        def fn(coef):
            fx, fxd = bs.dot_and_deriv_tfp(value, coef, batch_shape=(2, n))
            return jnp.sum(fx + 0.01 * fxd)

        grad = jax.grad(fn)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))
