import jax
import jax.numpy as jnp
import pytest

from liesel_ptm.bspline.onion import OnionKnots, OnionSpline

knots = OnionKnots(-4.0, 4.0, nparam=10)
coef = jax.random.normal(jax.random.key(1), (knots.nparam,))
bs = OnionSpline(knots.knots)


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
        coef = jax.random.normal(jax.random.key(1), (3, knots.nparam))
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
        coef = jax.random.normal(jax.random.key(1), (3, knots.nparam))
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
        coef = jax.random.normal(jax.random.key(1), (3, knots.nparam))
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
        coef = jax.random.normal(jax.random.key(1), (2, 3, knots.nparam))
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
        coef = jax.random.normal(jax.random.key(1), (3, knots.nparam))
        fx, _ = bs.dot_and_deriv(x, coef)

        # since fx is (3,), this will assume n=3 and batchdim=(3,), so the result is
        # (3, 3)
        # to get the original batched inversed, we then need to do jnp.diag(x2)
        # or vmap the dot inverse explicitly
        x2 = bs.dot_inverse(fx, coef)

        # shape fits
        assert x2.shape == (3, 3)

        # no nans
        assert not jnp.any(jnp.isnan(x2))

        assert jnp.allclose(jnp.diag(x2), x, atol=1e-4)

    def test_vector_x_batched_coef(self):
        x = jax.random.normal(jax.random.key(1), (200,))
        coef = jax.random.normal(jax.random.key(1), (3, knots.nparam))
        fx, _ = bs.dot_and_deriv(x, coef)

        x2 = bs.dot_inverse(fx, coef)

        # shape fits
        assert x2.shape == coef.shape[:-1] + x.shape

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
        coef = jax.random.normal(jax.random.key(1), (3, knots.nparam))
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
        coef = jax.random.normal(jax.random.key(1), (2, 3, knots.nparam))
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
        coef = jax.random.normal(jax.random.key(1), (3, knots.nparam))
        bs.dot_and_deriv_n_fullbatch(x, coef)

    def test_vector_x_batched_coef(self):
        x = jax.random.normal(jax.random.key(1), (200,))
        coef = jax.random.normal(jax.random.key(1), (3, knots.nparam))
        bs.dot_and_deriv_n_fullbatch(x, coef)

    def test_gptm_dot_and_deriv(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        k1 = jax.random.key(1)
        k1, k2 = jax.random.split(k1)
        bs = OnionSpline(knots.knots, subscripts="...nj,...nj->...n")

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
        bs = OnionSpline(knots.knots, subscripts="...nj,...nj->...n")

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
