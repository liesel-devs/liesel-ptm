"""Splines pass through JAX as pytrees: freed when discarded, compiled once."""

import gc
import logging
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import liesel_ptm as ptm

SPLINES = {
    "onion": lambda **kw: ptm.OnionSpline.from_bounds(-4.0, 4.0, nparam=10, **kw),
    "ptm": lambda **kw: ptm.PTMSpline.from_bounds(-4.0, 4.0, nparam=10, **kw),
}

COEF = 0.3 * jnp.cumsum(jax.random.normal(jax.random.key(0), (10,)))
Y = jnp.linspace(-6.0, 6.0, 9)
P = jnp.linspace(0.05, 0.95, 9)


@pytest.fixture(params=list(SPLINES), ids=list(SPLINES))
def make_spline(request):
    return SPLINES[request.param]


def dist(spline, coef=COEF):
    return ptm.LocScaleTransformationDist(
        coef, loc=jnp.zeros(9), scale=1.0, bspline=spline, centered=True, scaled=True
    )


@jax.jit
def jitted_log_prob(d, y):
    return d.log_prob(y)


@jax.jit
def jitted_grad(spline, coef, y):
    return jax.grad(lambda c: jnp.sum(dist(spline, c).log_prob(y)))(coef)


def use(spline):
    """Run every spline kernel: values, derivatives, inverse, moments, jit, grad."""
    d = dist(spline)
    outputs = (
        d.log_prob(Y),
        d.quantile(P),
        d.transformation_spline_mean(),
        jitted_log_prob(d, Y),
        jitted_grad(spline, COEF, Y),
        spline.evaluate(Y, COEF),
        spline.evaluate_and_deriv(Y, COEF)[1],
        spline.inverse(Y, COEF),
    )
    return [np.asarray(x) for x in outputs]


class CompileCounter(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        if "Finished XLA compilation" in record.getMessage():
            self.count += 1


def count_compiles(fn):
    counter = CompileCounter()
    logger = logging.getLogger("jax")
    logger.addHandler(counter)
    try:
        with jax.log_compiles(True):
            fn()
    finally:
        logger.removeHandler(counter)
    return counter.count


def test_discarded_spline_is_freed(make_spline):
    spline = make_spline()
    use(spline)
    ref = weakref.ref(spline)
    del spline
    gc.collect()
    assert ref() is None


def test_equal_splines_share_compilation(make_spline):
    use(make_spline())
    assert count_compiles(lambda: use(make_spline())) == 0


def test_spline_round_trips_as_pytree(make_spline):
    spline = make_spline()
    leaves, treedef = jax.tree_util.tree_flatten(spline)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert type(rebuilt) is type(spline)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    for expected, actual in zip(use(spline), use(rebuilt), strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "changed",
    [
        lambda make: make(),
        lambda make: (
            ptm.OnionSpline.from_bounds(-3.0, 5.0, nparam=10)
            if make is SPLINES["onion"]
            else ptm.PTMSpline.from_bounds(-3.0, 5.0, nparam=10)
        ),
    ],
    ids=["same", "other_knots"],
)
def test_jitted_results_follow_the_spline_passed_in(make_spline, changed):
    @jax.jit
    def evaluate(spline, y):
        return spline.evaluate_and_deriv(y, COEF), spline.inverse(y, COEF)

    first = make_spline()
    second = changed(make_spline)
    evaluate(first, Y)
    jitted = evaluate(second, Y)
    eager = (second.evaluate_and_deriv(Y, COEF), second.inverse(Y, COEF))
    for actual, expected in zip(
        jax.tree_util.tree_leaves(jitted), jax.tree_util.tree_leaves(eager), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_ptm_settings_change_results_under_jit():
    @jax.jit
    def evaluate(spline, y):
        return spline.evaluate(y, COEF)

    narrow = ptm.PTMSpline.from_bounds(-4.0, 4.0, nparam=10, eps=0.1)
    wide = ptm.PTMSpline.from_bounds(-4.0, 4.0, nparam=10, eps=0.5)
    linear = ptm.PTMSpline.from_bounds(-4.0, 4.0, nparam=10, continue_linearly=True)
    y = jnp.array([-9.0, 9.0])
    for spline in (narrow, wide, linear):
        np.testing.assert_allclose(
            evaluate(spline, y), spline.evaluate(y, COEF), rtol=1e-6, atol=1e-6
        )
    assert not np.allclose(evaluate(narrow, y), evaluate(wide, y))
