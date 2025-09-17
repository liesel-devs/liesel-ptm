from collections.abc import Callable

import jax
import jax.numpy as jnp
from interpax import interp1d
from jax import Array
from jax.typing import ArrayLike


def find_x_lo(
    min_y: ArrayLike, start: ArrayLike, fn: Callable[[ArrayLike], Array]
) -> Array:
    y_lo = start
    left_shift = jnp.array(0.1)
    min_zgrid = fn(y_lo - left_shift)
    min_zgrid = jnp.min(min_zgrid)

    def _cond_fun_l(val):
        _, min_zgrid = val
        return jnp.squeeze(min_zgrid >= min_y)

    def _body_fun_l(val):
        left_shift, min_zgrid = val
        left_shift = left_shift + 0.5
        min_zgrid = fn(y_lo - left_shift)
        min_zgrid = jnp.min(min_zgrid)
        return left_shift, min_zgrid

    left_shift, _ = jax.lax.while_loop(
        _cond_fun_l, _body_fun_l, (left_shift, min_zgrid)
    )
    return jnp.asarray(y_lo - left_shift)


def find_x_hi(
    max_y: ArrayLike, start: ArrayLike, fn: Callable[[ArrayLike], Array]
) -> Array:
    y_hi = start
    right_shift = jnp.array(0.1)
    max_zgrid = fn(y_hi + right_shift)
    max_zgrid = jnp.max(max_zgrid)

    def _cond_fun_r(val):
        _, max_zgrid = val
        return jnp.squeeze(max_zgrid <= max_y)

    def _body_fun_r(val):
        right_shift, max_zgrid = val
        right_shift = right_shift + 0.5
        max_zgrid = fn(y_hi + right_shift)
        max_zgrid = jnp.max(max_zgrid)
        return right_shift, max_zgrid

    right_shift, _ = jax.lax.while_loop(
        _cond_fun_r, _body_fun_r, (right_shift, max_zgrid)
    )
    return jnp.asarray(y_hi + right_shift)


def find_grid(
    y: ArrayLike,
    fn: Callable[[ArrayLike], Array],
    initial_xmin: ArrayLike,
    initial_xmax: ArrayLike,
    ngrid: int,
) -> tuple[Array, Array]:
    """
    Finds a grid of x values such that fn(xgrid) = ygrid covers the range of y values,
    that is, ygrid.min() < y < ygrid.max() for all elements of y.

    Returns a tuple of xgrid and ygrid.

    start_lo and start_high should be initial guesses for good xgrid.min() and
    xgrid.max() values.

    - inital_xmin should be smaller than initial_xmax
    - fn is assumed to be monotonically increasing
    """

    xlo = find_x_lo(jnp.min(y), start=initial_xmin, fn=fn)
    xhi = find_x_hi(jnp.max(y), start=initial_xmax, fn=fn)

    xgrid = jnp.linspace(xlo, xhi, ngrid)

    ygrid = fn(xgrid)

    return xgrid, ygrid


def inv1d(
    y: Array,
    f: Callable[[ArrayLike], Array],
    initial_xmin: float,
    initial_xmax: float,
    ngrid: int,
    method: str = "cubic",
) -> Array:
    was_scalar = jnp.ndim(y) == 0
    y_vec = jnp.atleast_1d(y)

    xgrid, ygrid = find_grid(
        y=y_vec,
        fn=f,
        initial_xmin=initial_xmin,
        initial_xmax=initial_xmax,
        ngrid=ngrid,
    )

    x = interp1d(y_vec, ygrid, xgrid, method=method)

    if was_scalar:
        return x[0]
    return x
