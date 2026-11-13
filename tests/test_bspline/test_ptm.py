import jax
import jax.numpy as jnp

from liesel_ptm.bspline.approx import _grid_index
from liesel_ptm.bspline.ptm import PTMKnots, PTMSpline

from .test_pytree import count_compiles

knots = PTMKnots(-4.0, 4.0, nparam=10)
coef = jax.random.normal(jax.random.key(1), (1, knots.nparam))
bs = PTMSpline(knots.knots)


def test_inverse_reuses_compilation_without_freezing_inputs():
    spline = PTMSpline(PTMKnots(-4.0, 4.0, nparam=6).knots)
    value = jnp.linspace(-1.0, 1.0, 5)
    coef = jnp.zeros(6)

    shifted = value + 0.1
    changed_coef = jnp.linspace(-0.2, 0.2, 6)
    first = spline.inverse(value, coef)
    results = []
    compiles = count_compiles(
        lambda: results.extend(
            [spline.inverse(shifted, coef), spline.inverse(value, changed_coef)]
        )
    )
    second, third = results

    assert compiles == 0
    assert jnp.allclose(first, value, atol=1e-4)
    assert jnp.allclose(second, value + 0.1, atol=1e-4)
    assert not jnp.allclose(third, first)
    assert jnp.allclose(spline.evaluate(third, changed_coef), value, atol=1e-4)


def test_algebraic_grid_lookup_matches_binary_search():
    approx = bs.bspline
    nodes = approx.grid[1:-1]
    x = jnp.concatenate(
        (
            jnp.linspace(approx.min_knot, approx.max_knot, 10_000),
            nodes,
            jnp.nextafter(nodes, -jnp.inf),
            jnp.nextafter(nodes, jnp.inf),
        )
    )
    expected = jnp.searchsorted(approx.grid, x, side="right") - 1
    expected = jnp.clip(expected, 0, approx.ngrid)

    assert jnp.array_equal(_grid_index(approx, x), expected)


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
        assert jnp.allclose(x2, x, atol=1e-5)


class TestBatchLayout:
    @staticmethod
    def _full_broadcast_args(value, coef, batch_shape=None):
        batch_shape = bs._resolve_batch_shape(coef, batch_shape)
        value = jnp.asarray(value)
        _, sample_shape, _, result_batch_shape = bs._value_layout(value, batch_shape)
        value = jnp.broadcast_to(value, sample_shape + result_batch_shape)
        coef = jnp.broadcast_to(coef, result_batch_shape + coef.shape[-1:])
        return value, coef

    @classmethod
    def _full_broadcast_forward_reference(cls, value, coef, batch_shape=None):
        value, coef = cls._full_broadcast_args(value, coef, batch_shape)
        return bs.evaluate_and_deriv(value, coef)

    @classmethod
    def _full_broadcast_inverse_reference(cls, value, coef, batch_shape=None):
        value, coef = cls._full_broadcast_args(value, coef, batch_shape)
        return bs.inverse(value, coef)

    def _assert_roundtrip(
        self,
        value,
        coef,
        expected_shape,
        batch_shape=None,
    ):
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
            value, coef, batch_shape=batch_shape
        )
        assert jnp.allclose(fx, fx_ref)
        assert jnp.allclose(fx_value, fx_ref)
        assert jnp.allclose(fxd, fxd_ref)

        x = bs.inverse(fx, coef, batch_shape=batch_shape)
        x_ref = self._full_broadcast_inverse_reference(
            fx, coef, batch_shape=batch_shape
        )

        assert x.shape == expected_shape
        assert not jnp.any(jnp.isnan(x))
        assert jnp.allclose(x, x_ref, atol=1e-4)

        expected = jnp.broadcast_to(jnp.asarray(value), expected_shape)
        assert jnp.allclose(x, expected, atol=1e-4)

    def test_single_coef(self):
        shared_coef = jnp.squeeze(coef, axis=0)

        self._assert_roundtrip(0.5, shared_coef, ())
        self._assert_roundtrip(jnp.linspace(-2.0, 2.0, 5), shared_coef, (5,))
        self._assert_roundtrip(
            jnp.linspace(-2.0, 2.0, 6).reshape((2, 3)),
            shared_coef,
            (2, 3),
        )

    def test_batched_coef(self):
        coef = jax.random.normal(jax.random.key(2), (2, knots.nparam))

        self._assert_roundtrip(0.5, coef, (2,))
        self._assert_roundtrip(
            jnp.linspace(-2.0, 2.0, 5).reshape((5, 1)),
            coef,
            (5, 2),
        )
        self._assert_roundtrip(
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

        self._assert_roundtrip(value, coef, (5,))

    def test_two_dimensional_batched_coef(self):
        coef = jax.random.normal(jax.random.key(3), (3, 2, knots.nparam))

        self._assert_roundtrip(0.5, coef, (3, 2))
        self._assert_roundtrip(
            jnp.linspace(-2.0, 2.0, 5).reshape((5, 1, 1)),
            coef,
            (5, 3, 2),
        )

    def test_explicit_broader_batch_shape(self):
        coef = jax.random.normal(jax.random.key(4), (2, knots.nparam))

        self._assert_roundtrip(
            0.5,
            coef,
            (3, 2),
            batch_shape=(3, 2),
        )

    def test_inverse_tail_values_match_full_broadcast_reference(self):
        coef = jax.random.normal(jax.random.key(7), (2, knots.nparam))
        value = jnp.asarray([-8.0, -4.0, 0.0, 4.0, 8.0]).reshape((5, 1))

        x = bs.inverse(value, coef, batch_shape=(2,))
        x_ref = self._full_broadcast_inverse_reference(value, coef, batch_shape=(2,))

        assert x.shape == (5, 2)
        assert jnp.allclose(x, x_ref, atol=1e-4)

    def test_jvp_through_shared_grid(self):
        coef = jax.random.normal(jax.random.key(5), (2, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1))

        def fn(value):
            fx, fxd = bs.evaluate_and_deriv(value, coef, batch_shape=(2,))
            return jnp.sum(fx + 0.01 * fxd)

        primal, tangent = jax.jvp(fn, (value,), (jnp.ones_like(value),))

        assert jnp.isfinite(primal)
        assert jnp.isfinite(tangent)

    def test_jvp_through_value_only_shared_grid(self):
        coef = jax.random.normal(jax.random.key(15), (2, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1))

        def fn(value):
            fx = bs.evaluate(value, coef, batch_shape=(2,))
            return jnp.sum(fx)

        primal, tangent = jax.jvp(fn, (value,), (jnp.ones_like(value),))

        assert jnp.isfinite(primal)
        assert jnp.isfinite(tangent)

    def test_grad_through_shared_grid_coef(self):
        coef = jax.random.normal(jax.random.key(6), (2, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1))

        def fn(coef):
            fx, fxd = bs.evaluate_and_deriv(value, coef, batch_shape=(2,))
            return jnp.sum(fx + 0.01 * fxd)

        grad = jax.grad(fn)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_through_value_only_shared_grid_coef(self):
        coef = jax.random.normal(jax.random.key(16), (2, knots.nparam))
        value = jnp.linspace(-2.0, 2.0, 5).reshape((5, 1))

        def fn(coef):
            fx = bs.evaluate(value, coef, batch_shape=(2,))
            return jnp.sum(fx)

        grad = jax.grad(fn)(coef)

        assert grad.shape == coef.shape
        assert jnp.all(jnp.isfinite(grad))
