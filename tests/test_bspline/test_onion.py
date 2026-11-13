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
        constrained_coef = bs._compute_coef(raw_coef)

        fx, fxd = bs.bspline.dot_and_deriv_n(x, constrained_coef)

        assert fx.shape == (2, 5)
        assert fxd.shape == (2, 5)
        assert jnp.all(jnp.isfinite(fx))
        assert jnp.all(jnp.isfinite(fxd))

    def test_rowwise_unbatched_basis_with_batched_rowwise_coef(self):
        x = jnp.linspace(-2.0, 2.0, 5)
        raw_coef = jax.random.normal(jax.random.key(2), (2, 5, knots.nparam))
        constrained_coef = bs._compute_coef(raw_coef)

        fx, fxd = bs.bspline.dot_and_deriv_n(x, constrained_coef)

        assert fx.shape == (2, 5)
        assert fxd.shape == (2, 5)
        assert jnp.all(jnp.isfinite(fx))
        assert jnp.all(jnp.isfinite(fxd))

    def test_mismatched_rowwise_coef_axis_raises(self):
        x = jnp.linspace(-2.0, 2.0, 5)
        raw_coef = jax.random.normal(jax.random.key(3), (2, 4, knots.nparam))
        constrained_coef = bs._compute_coef(raw_coef)

        with pytest.raises(ValueError, match="n_coef"):
            bs.bspline.dot_and_deriv_n(x, constrained_coef)


class TestMonotonicityAndInverse:
    def test_increasing_over_core_and_tails(self):
        tfp_coef = jnp.squeeze(coef, axis=0)
        x = jnp.linspace(-8.0, 8.0, 10_000)
        fx, fxd = bs.dot_and_deriv_tfp(x, tfp_coef)

        assert fx.shape == x.shape
        assert fxd.shape == x.shape
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))
        assert jnp.all(fxd > 0.0)
        assert jnp.all(jnp.diff(fx) > 0.0)

    def test_inverse_over_core_and_tails(self):
        tfp_coef = jnp.squeeze(coef, axis=0)
        x = jnp.linspace(-8.0, 8.0, 300)
        fx = bs.dot_tfp(x, tfp_coef)

        x2 = bs.dot_inverse_tfp(fx, tfp_coef)

        assert x2.shape == x.shape
        assert not jnp.any(jnp.isnan(x2))
        assert jnp.allclose(x2, x, atol=1e-4)

    def test_subscripts_argument_is_accepted_as_compatibility_noop(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs_default = OnionSpline(knots.knots)
        bs_compat = OnionSpline(knots.knots, subscripts="...nj,...nj->...n")
        x = jnp.linspace(-2.0, 2.0, 5)
        coef = jax.random.normal(jax.random.key(2), (5, knots.nparam))

        fx_default, fxd_default = bs_default.dot_and_deriv_tfp(x, coef)
        fx_compat, fxd_compat = bs_compat.dot_and_deriv_tfp(x, coef)

        assert jnp.allclose(fx_default, fx_compat)
        assert jnp.allclose(fxd_default, fxd_compat)


class TestTfpLayout:
    @staticmethod
    def _full_broadcast_args(bs, value, coef, batch_shape=None):
        batch_shape = bs._tfp_batch_shape(coef, batch_shape)
        value = jnp.asarray(value)
        _, sample_shape, _, result_batch_shape = bs._tfp_value_layout(
            value, batch_shape
        )
        value = jnp.broadcast_to(value, sample_shape + result_batch_shape)
        coef = jnp.broadcast_to(coef, result_batch_shape + coef.shape[-1:])
        return value, coef

    @classmethod
    def _full_broadcast_forward_reference(cls, bs, value, coef, batch_shape=None):
        value, coef = cls._full_broadcast_args(bs, value, coef, batch_shape)
        return bs.dot_and_deriv_tfp(value, coef)

    @classmethod
    def _full_broadcast_inverse_reference(cls, bs, value, coef, batch_shape=None):
        value, coef = cls._full_broadcast_args(bs, value, coef, batch_shape)
        return bs.dot_inverse_tfp(value, coef)

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
        fx_value = bs.dot_tfp(value, coef, batch_shape=batch_shape)

        assert fx.shape == expected_shape
        assert fxd.shape == expected_shape
        assert fx_value.shape == expected_shape
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))
        assert not jnp.any(jnp.isnan(fx_value))
        assert jnp.all(fxd > 0.0)

        fx_ref, fxd_ref = self._full_broadcast_forward_reference(
            bs, value, coef, batch_shape=batch_shape
        )
        # Equivalent float32 evaluation paths differ slightly near zero.
        assert jnp.allclose(fx, fx_ref, atol=1e-6)
        assert jnp.allclose(fx_value, fx_ref, atol=1e-6)
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

    def test_value_only_outside_core_identity(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        n = 4
        coef = jax.random.normal(jax.random.key(8), (2, n, knots.nparam))
        value = jnp.asarray([-8.0, 8.0]).reshape((2, 1, 1))

        fx = bs.dot_tfp(value, coef, batch_shape=(2, n))
        expected = jnp.broadcast_to(value, (2, 2, n))

        assert fx.shape == (2, 2, n)
        assert jnp.allclose(fx, expected)

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

    def test_jvp_through_value_only_rowwise_shared_grid_tfp(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        n = 4
        coef = jax.random.normal(jax.random.key(9), (2, n, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1, 1))

        def fn(value):
            fx = bs.dot_tfp(value, coef, batch_shape=(2, n))
            return jnp.sum(fx)

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

    def test_grad_through_value_only_rowwise_shared_grid_tfp_coef(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        n = 4
        coef = jax.random.normal(jax.random.key(10), (2, n, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1, 1))

        def fn(coef):
            fx = bs.dot_tfp(value, coef, batch_shape=(2, n))
            return jnp.sum(fx)

        grad = jax.grad(fn)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))
