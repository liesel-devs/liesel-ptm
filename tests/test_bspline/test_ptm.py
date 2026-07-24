import jax
import jax.numpy as jnp
import pytest

from liesel_ptm.bspline.ptm import PTMKnots, PTMSpline

knots = PTMKnots(-4.0, 4.0, nparam=10)
coef = jax.random.normal(jax.random.key(1), (1, knots.nparam))
bs = PTMSpline(knots.knots)


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

    def test_rejects_rowwise_coef_for_standard_spline(self):
        rowwise_coef = jax.random.normal(jax.random.key(1), (3, knots.nparam))

        with pytest.raises(ValueError, match="rowwise"):
            bs.dot_and_deriv(jnp.zeros((3,)), rowwise_coef)

        with pytest.raises(ValueError, match="rowwise"):
            bs.dot_inverse(jnp.zeros((3,)), rowwise_coef)

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

        assert jnp.allclose(x2, x, atol=1e-5)

    def test_batched_x_single_coef(self):
        x = jax.random.normal(jax.random.key(1), (4, 200))
        fx, _ = bs.dot_and_deriv(x, coef)

        x2 = bs.dot_inverse(fx, coef)

        # shape fits
        assert x2.shape == x.shape

        # no nans
        assert not jnp.any(jnp.isnan(x2))

        assert jnp.allclose(x2, x, atol=1e-5)

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

        assert fx.shape == (4, 200)
        assert fxd.shape == (4, 200)
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))
        assert jnp.all(fxd > 0.0)

    def test_scalar_x_batched_coef(self):
        x = 1.0
        coef = jax.random.normal(jax.random.key(1), (3, 1, knots.nparam))
        fx, fxd = bs.dot_and_deriv_n_fullbatch(x, coef)

        assert fx.shape == (3,)
        assert fxd.shape == (3,)
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))
        assert jnp.all(fxd > 0.0)

    def test_vector_x_batched_coef(self):
        x = jax.random.normal(jax.random.key(1), (200,))
        coef = jax.random.normal(jax.random.key(1), (3, 1, knots.nparam))
        fx, fxd = bs.dot_and_deriv_n_fullbatch(x, coef)

        assert fx.shape == (3, 200)
        assert fxd.shape == (3, 200)
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))
        assert jnp.all(fxd > 0.0)


class TestPublicMethodEquivalence:
    def test_forward_methods_agree(self):
        x = jax.random.normal(jax.random.key(1), (3, 7))
        coef = jax.random.normal(jax.random.key(2), (3, 1, knots.nparam))

        fx, fxd = bs.dot_and_deriv(x, coef)
        fx_n, fxd_n = bs.dot_and_deriv_n(x, coef)
        fx_full, fxd_full = bs.dot_and_deriv_n_fullbatch(x, coef)

        assert jnp.allclose(fx, fx_n)
        assert jnp.allclose(fxd, fxd_n)
        assert jnp.allclose(fx, fx_full)
        assert jnp.allclose(fxd, fxd_full)

    def test_inverse_methods_agree(self):
        x = jnp.linspace(-2.0, 2.0, 15).reshape((3, 5))
        coef = jax.random.normal(jax.random.key(2), (3, 1, knots.nparam))
        fx, _ = bs.dot_and_deriv(x, coef)

        x_inv = bs.dot_inverse(fx, coef)
        x_inv_n = bs.dot_inverse_n(fx, coef)
        x_inv_full = bs.dot_inverse_n_fullbatch(fx, coef)

        assert x_inv.shape == x.shape
        assert jnp.allclose(x_inv, x, atol=1e-4)
        assert jnp.allclose(x_inv, x_inv_n)
        assert jnp.allclose(x_inv, x_inv_full)

    def test_jvp_through_batched_forward(self):
        x = jnp.linspace(-2.0, 2.0, 6).reshape((2, 3))
        coef = jax.random.normal(jax.random.key(2), (2, 1, knots.nparam))

        def fn(value):
            fx, fxd = bs.dot_and_deriv(value, coef)
            return jnp.sum(fx + 0.01 * fxd)

        primal, tangent = jax.jvp(fn, (x,), (jnp.ones_like(x),))

        assert jnp.isfinite(primal)
        assert jnp.isfinite(tangent)


class TestTfpLayout:
    @staticmethod
    def _full_broadcast_forward_reference(value, coef, batch_shape=None):
        batch_shape = bs._tfp_batch_shape(coef, batch_shape)
        value, sample_shape, result_batch_shape = bs._broadcast_tfp_value(
            value, batch_shape
        )

        value = bs._tfp_to_legacy_batch_last(value, result_batch_shape, sample_shape)
        legacy_coef = jnp.expand_dims(coef, axis=-2)
        fx, fxd = bs.dot_and_deriv(value, legacy_coef)

        fx = bs._legacy_batch_last_to_tfp(fx, result_batch_shape, sample_shape)
        fxd = bs._legacy_batch_last_to_tfp(fxd, result_batch_shape, sample_shape)

        return fx, fxd

    @staticmethod
    def _full_broadcast_inverse_reference(value, coef, batch_shape=None):
        batch_shape = bs._tfp_batch_shape(coef, batch_shape)
        value, sample_shape, result_batch_shape = bs._broadcast_tfp_value(
            value, batch_shape
        )

        value = bs._tfp_to_legacy_batch_last(value, result_batch_shape, sample_shape)
        legacy_coef = jnp.expand_dims(coef, axis=-2)
        inverse = bs.dot_inverse(value, legacy_coef)

        return bs._legacy_batch_last_to_tfp(inverse, result_batch_shape, sample_shape)

    def _assert_tfp_roundtrip(
        self,
        value,
        coef,
        expected_shape,
        batch_shape=None,
    ):
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
            value, coef, batch_shape=batch_shape
        )
        assert jnp.allclose(fx, fx_ref)
        assert jnp.allclose(fx_value, fx_ref)
        assert jnp.allclose(fxd, fxd_ref)

        x = bs.dot_inverse_tfp(fx, coef, batch_shape=batch_shape)
        x_ref = self._full_broadcast_inverse_reference(
            fx, coef, batch_shape=batch_shape
        )

        assert x.shape == expected_shape
        assert not jnp.any(jnp.isnan(x))
        assert jnp.allclose(x, x_ref, atol=1e-4)

        expected = jnp.broadcast_to(jnp.asarray(value), expected_shape)
        assert jnp.allclose(x, expected, atol=1e-4)

    def test_single_coef(self):
        tfp_coef = jnp.squeeze(coef, axis=0)

        self._assert_tfp_roundtrip(0.5, tfp_coef, ())
        self._assert_tfp_roundtrip(jnp.linspace(-2.0, 2.0, 5), tfp_coef, (5,))
        self._assert_tfp_roundtrip(
            jnp.linspace(-2.0, 2.0, 6).reshape((2, 3)),
            tfp_coef,
            (2, 3),
        )

    def test_batched_coef(self):
        coef = jax.random.normal(jax.random.key(2), (2, knots.nparam))

        self._assert_tfp_roundtrip(0.5, coef, (2,))
        self._assert_tfp_roundtrip(
            jnp.linspace(-2.0, 2.0, 5).reshape((5, 1)),
            coef,
            (5, 2),
        )
        self._assert_tfp_roundtrip(
            jnp.stack(
                (
                    jnp.linspace(-2.0, 2.0, 5),
                    jnp.linspace(-1.5, 1.5, 5),
                ),
                axis=-1,
            ),
            coef,
            (5, 2),
        )

    def test_batch_one_coef_broadcasts_observations(self):
        coef = jax.random.normal(jax.random.key(8), (1, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5)

        self._assert_tfp_roundtrip(value, coef, (5,))

    def test_two_dimensional_batched_coef(self):
        coef = jax.random.normal(jax.random.key(3), (3, 2, knots.nparam))

        self._assert_tfp_roundtrip(0.5, coef, (3, 2))
        self._assert_tfp_roundtrip(
            jnp.linspace(-2.0, 2.0, 5).reshape((5, 1, 1)),
            coef,
            (5, 3, 2),
        )

    def test_explicit_broader_batch_shape(self):
        coef = jax.random.normal(jax.random.key(4), (2, knots.nparam))

        self._assert_tfp_roundtrip(
            0.5,
            coef,
            (3, 2),
            batch_shape=(3, 2),
        )

    def test_inverse_tail_values_match_full_broadcast_reference(self):
        coef = jax.random.normal(jax.random.key(7), (2, knots.nparam))
        value = jnp.asarray([-8.0, -4.0, 0.0, 4.0, 8.0]).reshape((5, 1))

        x = bs.dot_inverse_tfp(value, coef, batch_shape=(2,))
        x_ref = self._full_broadcast_inverse_reference(value, coef, batch_shape=(2,))

        assert x.shape == (5, 2)
        assert jnp.allclose(x, x_ref, atol=1e-4)

    def test_jvp_through_shared_grid_tfp(self):
        coef = jax.random.normal(jax.random.key(5), (2, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1))

        def fn(value):
            fx, fxd = bs.dot_and_deriv_tfp(value, coef, batch_shape=(2,))
            return jnp.sum(fx + 0.01 * fxd)

        primal, tangent = jax.jvp(fn, (value,), (jnp.ones_like(value),))

        assert jnp.isfinite(primal)
        assert jnp.isfinite(tangent)

    def test_jvp_through_value_only_shared_grid_tfp(self):
        coef = jax.random.normal(jax.random.key(15), (2, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1))

        def fn(value):
            fx = bs.dot_tfp(value, coef, batch_shape=(2,))
            return jnp.sum(fx)

        primal, tangent = jax.jvp(fn, (value,), (jnp.ones_like(value),))

        assert jnp.isfinite(primal)
        assert jnp.isfinite(tangent)

    def test_grad_through_shared_grid_tfp_coef(self):
        coef = jax.random.normal(jax.random.key(6), (2, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1))

        def fn(coef):
            fx, fxd = bs.dot_and_deriv_tfp(value, coef, batch_shape=(2,))
            return jnp.sum(fx + 0.01 * fxd)

        grad = jax.grad(fn)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_through_value_only_shared_grid_tfp_coef(self):
        coef = jax.random.normal(jax.random.key(16), (2, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1))

        def fn(coef):
            fx = bs.dot_tfp(value, coef, batch_shape=(2,))
            return jnp.sum(fx)

        grad = jax.grad(fn)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))
