import jax
import jax.numpy as jnp

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


class TestMonotonicityAndInverse:
    def test_increasing_over_core_and_tails(self):
        shared_coef = jnp.squeeze(coef, axis=0)
        x = jnp.linspace(-8.0, 8.0, 10_000)
        fx, fxd = bs.evaluate_and_deriv(x, shared_coef)

        assert fx.shape == x.shape
        assert fxd.shape == x.shape
        assert not jnp.any(jnp.isnan(fx))
        assert not jnp.any(jnp.isnan(fxd))
        assert jnp.all(fxd > 0.0)
        assert jnp.all(jnp.diff(fx) > 0.0)

    def test_inverse_over_core_and_tails(self):
        shared_coef = jnp.squeeze(coef, axis=0)
        x = jnp.linspace(-8.0, 8.0, 300)
        fx = bs.evaluate(x, shared_coef)

        x2 = bs.inverse(fx, shared_coef)

        assert x2.shape == x.shape
        assert not jnp.any(jnp.isnan(x2))
        assert jnp.allclose(x2, x, atol=1e-4)

    def test_batched_inverse_over_core_and_tails(self):
        # 300 clearly different coefficient rows: more than one inversion block,
        # so blocks are padded and rows in a block share one grid.
        batch_shape = (2, 150)
        steps = 0.4 * jax.random.normal(
            jax.random.key(11), batch_shape + (knots.nparam,)
        )
        batched_coef = jnp.cumsum(steps, axis=-1)
        x = jnp.linspace(-8.0, 8.0, 200).reshape((200, 1, 1))
        fx = bs.evaluate(x, batched_coef)

        x2 = bs.inverse(fx, batched_coef)

        assert x2.shape == (200,) + batch_shape
        assert not jnp.any(jnp.isnan(x2))
        assert jnp.allclose(x2, jnp.broadcast_to(x, x2.shape), atol=1e-4)


class TestBatchLayout:
    @staticmethod
    def _full_broadcast_args(bs, value, coef, batch_shape=None):
        batch_shape = bs._resolve_batch_shape(coef, batch_shape)
        value = jnp.asarray(value)
        _, sample_shape, _, result_batch_shape = bs._value_layout(value, batch_shape)
        value = jnp.broadcast_to(value, sample_shape + result_batch_shape)
        coef = jnp.broadcast_to(coef, result_batch_shape + coef.shape[-1:])
        return value, coef

    @classmethod
    def _full_broadcast_forward_reference(cls, bs, value, coef, batch_shape=None):
        value, coef = cls._full_broadcast_args(bs, value, coef, batch_shape)
        return bs.evaluate_and_deriv(value, coef)

    @classmethod
    def _full_broadcast_inverse_reference(cls, bs, value, coef, batch_shape=None):
        value, coef = cls._full_broadcast_args(bs, value, coef, batch_shape)
        return bs.inverse(value, coef)

    def _assert_roundtrip(
        self,
        value,
        coef,
        expected_shape,
        batch_shape=None,
    ):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)

        fx, fxd = bs.evaluate_and_deriv(value, coef, batch_shape=batch_shape)
        fx_value = bs.evaluate(value, coef, batch_shape=batch_shape)

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

        x = bs.inverse(fx, coef, batch_shape=batch_shape)
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

        self._assert_roundtrip(0.5, coef, ())
        self._assert_roundtrip(jnp.linspace(-2.0, 2.0, 5), coef, (5,))

    def test_batched_coef(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        n = 4
        coef = jax.random.normal(jax.random.key(2), (n, knots.nparam))

        self._assert_roundtrip(0.5, coef, (n,))
        self._assert_roundtrip(
            jnp.linspace(-2.0, 2.0, 5).reshape((5, 1)),
            coef,
            (5, n),
        )

    def test_batch_one_coef_broadcasts_observations(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        coef = jax.random.normal(jax.random.key(7), (1, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5)

        self._assert_roundtrip(value, coef, (5,))

    def test_two_dimensional_batched_coef(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        n = 4
        coef = jax.random.normal(jax.random.key(3), (2, n, knots.nparam))

        self._assert_roundtrip(0.5, coef, (2, n))
        self._assert_roundtrip(
            jnp.linspace(-2.0, 2.0, 5).reshape((5, 1, 1)),
            coef,
            (5, 2, n),
        )
        self._assert_roundtrip(
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

        x = bs.inverse(value, coef, batch_shape=(2, n))
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

        fx = bs.evaluate(value, coef, batch_shape=(2, n))
        expected = jnp.broadcast_to(value, (2, 2, n))

        assert fx.shape == (2, 2, n)
        assert jnp.allclose(fx, expected)

    def test_jvp_with_batched_coef(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        n = 4
        coef = jax.random.normal(jax.random.key(4), (2, n, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1, 1))

        def fn(value):
            fx, fxd = bs.evaluate_and_deriv(value, coef, batch_shape=(2, n))
            return jnp.sum(fx + 0.01 * fxd)

        primal, tangent = jax.jvp(fn, (value,), (jnp.ones_like(value),))

        assert jnp.isfinite(primal)
        assert jnp.isfinite(tangent)

    def test_jvp_value_only_with_batched_coef(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        n = 4
        coef = jax.random.normal(jax.random.key(9), (2, n, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1, 1))

        def fn(value):
            fx = bs.evaluate(value, coef, batch_shape=(2, n))
            return jnp.sum(fx)

        primal, tangent = jax.jvp(fn, (value,), (jnp.ones_like(value),))

        assert jnp.isfinite(primal)
        assert jnp.isfinite(tangent)

    def test_grad_wrt_batched_coef(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        n = 4
        coef = jax.random.normal(jax.random.key(5), (2, n, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1, 1))

        def fn(coef):
            fx, fxd = bs.evaluate_and_deriv(value, coef, batch_shape=(2, n))
            return jnp.sum(fx + 0.01 * fxd)

        grad = jax.grad(fn)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_value_only_wrt_batched_coef(self):
        knots = OnionKnots(-4.0, 4.0, nparam=11)
        bs = OnionSpline(knots.knots)
        n = 4
        coef = jax.random.normal(jax.random.key(10), (2, n, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1, 1))

        def fn(coef):
            fx = bs.evaluate(value, coef, batch_shape=(2, n))
            return jnp.sum(fx)

        grad = jax.grad(fn)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))
