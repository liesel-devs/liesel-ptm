from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import scipy

from .custom_types import Array


def initial_inverse_guess(y, z, znew):
    """
    For each element of znew, this looks for the closest element of z and returns the
    corresponding y.

    Assumes that f(y) = z, such that the returned y should be a good first guess at
    f^-1(znew) = ynew.
    """
    if not len(jnp.atleast_1d(znew).shape) == 1:
        raise ValueError(f"znew must be one-dimensional. Actual shape: {znew.shape}")
    znew = jnp.expand_dims(jnp.atleast_1d(znew), axis=-1)
    jnp.broadcast_shapes(z.shape, znew.shape)
    return y[jnp.argmin(jnp.abs(z - znew), axis=1)]


@partial(jnp.vectorize, signature="(m),(m),(n)->(n)")
def approximate_inverse(y: Array, z: Array, znew: Array) -> Array:
    """
    Given a grid of ``f(y) = z``, this returns an approximation of ``f^-1(znew) = ynew``
    by finding the closest grid point to each element of znew and interpolating
    linearly between the two closest grid points.
    """
    i = jnp.searchsorted(z, znew, side="right") - 1
    lo, hi = z[i], z[i + 1]
    step = hi - lo
    k = (znew - lo) / step
    k = jnp.where(jnp.isinf(k), 1.0, k)
    approx_y_new = (1.0 - k) * y[i] + (k * y[i + 1])
    return approx_y_new


def invert_fn(
    fn: Callable, tol: float = 1e-4, maxiter: int = 2000, jac: Callable | None = None
) -> Callable[[Array, Array | None], scipy.optimize.OptimizeResult]:
    """
    Returns a function that numerically inverts the input function using \
    :func:`scipy.optimize.root`.
    """

    def inverse(z: Array, y0: Array | None = None) -> Array:
        y0 = y0 if y0 is not None else jnp.zeros(jnp.shape(z))

        def diff_fn(y):
            return fn(y) - z

        return scipy.optimize.root(
            diff_fn, y0, jac=jac, method="hybr", tol=tol, options={"maxfev": maxiter}
        )

    return inverse


def numerical_inverse_jax(
    fn: Callable, tol: float = 1e-6, max_iter: int = 100
) -> Callable:
    fn_grad = jax.grad(fn)

    @jax.jit
    @partial(jnp.vectorize, signature="(),()->()")
    def inverse_fn(y, initial_guess):
        def newton_step(x, y):
            return x - (fn(x) - y) / fn_grad(x)

        def cond_fn(state):
            x, x_new, iter_count = state
            return jnp.logical_and(jnp.abs(x_new - x) >= tol, iter_count < max_iter)

        def body_fn(state):
            _, x, iter_count = state
            x_new = newton_step(x, y)
            return x, x_new, iter_count + 1

        # Initial state: (current x, new x, iteration count)
        state = (initial_guess, initial_guess + 2 * tol, 0)
        _, x_new, _ = jax.lax.while_loop(cond_fn, body_fn, state)

        return x_new

    return inverse_fn
