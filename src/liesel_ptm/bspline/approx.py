from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array, jit, lax, vmap
from jax.tree_util import Partial as partial


def equidistant_knots(
    x: Array, n_param: int, order: int = 3, eps: float = 0.01
) -> Array:
    """
    Create equidistant knots for a B-spline of the specified order.

    Parameters
    ----------
    x
        A 1d array of input data.
    order
        A positive integer giving the order of the spline function.
        A cubic spline has an order of 3.
    n_param
        Number of parameters of the B-spline.
    eps
        A factor by which the range of the interior knots is stretched. The range of
        interior knots will thus be ``eps * (jnp.max(x) - jnp.min(x))``.

    Returns
    -------
    A 1d array

    Notes
    -----

    Some additional info:

    - ``dim(knots) = n_param + order + 1``
    - ``n_param = dim(knots) - order - 1``
    - ``n_interior_knots = n_param - order + 1``

    """
    if order < 0:
        raise ValueError(f"Invalid {order=}.")

    if n_param < order:
        raise ValueError(f"{n_param=} must not be smaller than {order=}.")

    n_internal_knots = n_param - order + 1

    a = jnp.min(x)
    b = jnp.max(x)

    range_ = b - a

    min_k = a - range_ * (eps / 2)
    max_k = b + range_ * (eps / 2)

    internal_knots = jnp.linspace(min_k, max_k, n_internal_knots)

    step = internal_knots[1] - internal_knots[0]

    left_knots = jnp.linspace(min_k - (step * order), min_k - step, order)
    right_knots = jnp.linspace(max_k + step, max_k + (step * order), order)

    knots = jnp.concatenate((left_knots, internal_knots, right_knots))
    return knots


@partial(jit, static_argnames="order")
def _build_basis_vector(x: Array, knots: Array, order: int) -> Array:
    """
    Builds a vector of length ``dim(knots) - order - 1``. Each entry ``i`` is
    iterativaly updated. At time m, the entry i is the evaluation of the basis function
    at the observed value for the m-th order and for the i-th knot. The creation of the
    matrix needs a row-wise (order) loop (f1) and a column-wise (knot index) loop (f2).
    """
    k = jnp.shape(knots)[0] - order - 1
    bv = jnp.full(jnp.shape(knots)[0] - 1, jnp.nan)

    def basis_per_order(m, bv):
        def basis_per_knot(i, bv):
            def base_case(bv):
                return bv.at[i].set(
                    jnp.where(x >= knots[i], 1.0, 0.0)
                    * jnp.where(x < knots[i + 1], 1.0, 0.0)
                )

            def recursive_case(bv):
                b1 = (x - knots[i]) / (knots[i + m] - knots[i]) * bv[i]
                b2 = (
                    (knots[i + m + 1] - x)
                    / (knots[i + m + 1] - knots[i + 1])
                    * bv[i + 1]
                )

                return bv.at[i].set(b1 + b2)

            return lax.cond(m == 0, base_case, recursive_case, bv)

        return lax.fori_loop(0, k + order, basis_per_knot, bv)

    return lax.fori_loop(0, order + 1, basis_per_order, bv)[:k]


class Basis:
    """
    Provides methods to construct basis matrices.
    """

    @staticmethod
    def bspline(
        x: Array, knots: Array, order: int = 3, outer_ok: bool = False
    ) -> Array:
        """
        Builds a B-spline basis matrix.

        Parameters
        ----------
        x
            A 1d array of input data.
        knots
            A 1d array of knots. The knots will be sorted.
        order
            A positive integer giving the order of the spline function. \
            A cubic spline has an order of 3.
        outer_ok
            If ``False`` (default), values of x outside the range of interior knots \
            cause an error. If ``True``, they are allowed.

        Returns
        -------
        A 2d array, the B-spline basis matrix.


        Notes
        -----
        Under the hood, instead of applying the recursive
        definition of B-splines, a matrix of (order + 1) rows and
        (dim(knots) - order - 1)
        columns for each value in x is created. This matrix store the evaluation of
        the basis function at the observed value for the m-th order and for the i-th
        knot.

        .. rubric:: Jit-compilation

        The ``basis_matrix`` function internally uses a jit-compiled function to do the
        heavy lifting. However, you may want to make ``basis_matrix`` itself
        jit-compilable.
        In this case, you need to define the arguments ``order`` and ``outer_ok`` as
        static arguments. Further, ``outer_ok`` needs to be fixed to ``True``.

        If you just want to set up a basis matrix once, it is usually not necessary to
        go through this process.

        Example:

        .. code-block:: python

            from liesel.contrib.splines import equidistant_knots, basis_matrix

            x = jnp.linspace(-2.0, 2.0, 30)
            knots = equidistant_knots(x, n_param=10, order=3)

            basis_matrix_jit = jax.jit(
                basis_matrix, static_argnames=("order", "outer_ok")
            )

            B = basis_matrix_jit(x, knots, order, outer_ok=True)

        Another suitable way to go is to use ``functools.partial``::

            from functools import partial
            from liesel.contrib.splines import equidistant_knots, basis_matrix

            x = jnp.linspace(-2.0, 2.0, 30)
            knots = equidistant_knots(x, n_param=10, order=3)

            basis_matrix_fix = partial(basis_matrix, order=3, outer_ok=True)
            basis_matrix_jit = jax.jit(basis_matrix_fix)

            B = basis_matrix_jit(x, knots)

        """
        if order < 0:
            raise ValueError(f"Invalid {order=}.")

        # if x is a scalar, this ensures that the function still works
        x = jnp.atleast_1d(x)

        knots = jnp.sort(knots)

        if not outer_ok:
            min_ = knots[order]
            max_ = knots[knots.shape[0] - order - 1]
            geq_min = jnp.min(x) >= min_
            leq_max = jnp.max(x) <= max_
            if not (geq_min and leq_max):
                raise ValueError(
                    f"Values of x are not inside the range of interior knots, [{min_},"
                    f" {max_}]"
                )

        design_matrix = vmap(lambda x: _build_basis_vector(x, knots, order))(x)

        return design_matrix


def _bspline_basis(x, knots, order):
    """Return B-spline basis, allowing values outside knots."""
    return Basis.bspline(x, knots, order, outer_ok=True)


@partial(jax.jit, static_argnums=2)
@partial(jnp.vectorize, excluded=(1, 2), signature="(n)->(n,p)")
def bspline_basis(x, knots, order):
    """
    Vectorized B-spline basis function evaluation.

    Parameters
    ----------
    x
        Input array.
    knots
        Array of knots.
    order
        Order of the spline (``order=3`` for a cubic spline).

    Returns
    -------
    B-spline basis matrix.
    """
    min_knot = knots[order]
    max_knot = knots[-(order + 1)]
    basis = _bspline_basis(x, knots, order)
    mask = jnp.logical_or(x < min_knot, x > max_knot)
    mask = jnp.expand_dims(mask, -1)
    return jnp.where(mask, 0.0, basis)


@partial(jax.jit, static_argnums=2)
@partial(jnp.vectorize, excluded=(1, 2), signature="(n)->(n,p)")
def bspline_basis_deriv(x, knots, order):
    """Evaluate first derivative of B-spline basis."""
    min_knot = knots[order]
    max_knot = knots[-(order + 1)]

    basis = _bspline_basis(x, knots[1:-1], order - 1)
    dknots = jnp.diff(knots).mean()
    D = jnp.diff(jnp.identity(jnp.shape(knots)[-1] - order - 1)).T
    basis_grad = basis @ (D / dknots)

    mask = jnp.logical_or(x < min_knot, x > max_knot)
    mask = jnp.expand_dims(mask, -1)
    return jnp.where(mask, 0.0, basis_grad)


@partial(jax.jit, static_argnums=2)
@partial(jnp.vectorize, excluded=(1, 2), signature="(n)->(n,p)")
def bspline_basis_deriv2(x, knots, order):
    """Evaluate second derivative of B-spline basis."""
    min_knot = knots[order]
    max_knot = knots[-(order + 1)]
    basis = _bspline_basis(x, knots[2:-2], order - 2)

    dknots = jnp.diff(knots).mean()
    D = jnp.diff(jnp.identity(jnp.shape(knots)[-1] - order - 1)).T
    basis_grad = basis @ D[1::, 1:] @ (D / (dknots**2))

    mask = jnp.logical_or(x < min_knot, x > max_knot)
    mask = jnp.expand_dims(mask, -1)
    return jnp.where(mask, 0.0, basis_grad)


@jax.tree_util.register_pytree_node_class
class BSplineApprox:
    """
    Approximate B-spline evaluations on a fixed grid.

    Parameters
    ----------
    knots
        Knot positions.
    order
        Order of the spline.
    ngrid
        Number of grid points used to precompute the basis (default 1000).
    postmultiply_by
        Optional matrix to post-multiply the basis. In ``B(x) @ Z``, ``B(x)`` is \
        the basis matrix \
        and ``Z`` is the postmultiplication matrix. Will be applied to the grid, \
        which can save a lot of computation time.

    Attributes
    ----------
    knots
        Knot positions.
    dknots
        Average knot spacing.
    order
        Order of the spline.
    nparam
        Number of spline parameters.
    min_knot
        Minimum interior knot.
    max_knot
        Maximum interior knot.
    ngrid
        Number of grid points.
    grid
        Grid used for approximation.
    step
        Grid spacing.
    postmultiply_by
        Postmultiplication matrix.
    basis
        Precomputed basis on the grid.
    basis_deriv
        Precomputed first derivative on the grid.
    basis_deriv2
        Precomputed second derivative on the grid.
    """

    def __init__(
        self,
        knots: Array,
        order: int,
        ngrid: int = 1000,
        postmultiply_by: Array | None = None,
    ) -> None:
        self.knots = jnp.asarray(knots)
        self.dknots = jnp.mean(jnp.diff(knots))
        self.order = order
        self.nparam = int(jnp.shape(knots)[0]) - order - 1

        self.min_knot = self.knots[order]
        self.max_knot = self.knots[-(order + 1)]

        grid = jnp.linspace(self.min_knot, self.max_knot, ngrid)
        self.step = (self.max_knot - self.min_knot) / (ngrid - 1)
        prepend = jnp.array([self.min_knot - self.step])
        append = jnp.array([self.max_knot + self.step])
        self.ngrid = int(ngrid)
        self.grid = jnp.concatenate((prepend, grid, append))

        Z = jnp.eye(self.nparam) if postmultiply_by is None else postmultiply_by
        self.postmultiply_by = Z

        basis_grids = self._compute_basis_and_deriv2(self.grid)
        self.basis_grid = basis_grids[0]
        self.basis_deriv_grid = basis_grids[1]
        self.basis_deriv2_grid = basis_grids[2]

    # Splines are pytrees: arrays are leaves and the integer settings are static
    # aux data. JAX's caches then hold no spline objects, and splines with equal
    # settings share compiled kernels.
    _array_fields = (
        "knots",
        "dknots",
        "min_knot",
        "max_knot",
        "step",
        "grid",
        "postmultiply_by",
        "basis_grid",
        "basis_deriv_grid",
        "basis_deriv2_grid",
    )
    _static_fields = ("order", "nparam", "ngrid")

    def tree_flatten(self) -> tuple[tuple[Array, ...], tuple[object, ...]]:
        children = tuple(getattr(self, name) for name in self._array_fields)
        aux = tuple(getattr(self, name) for name in self._static_fields)
        return children, aux

    @classmethod
    def tree_unflatten(
        cls, aux: tuple[object, ...], children: tuple[Array, ...]
    ) -> BSplineApprox:
        obj = object.__new__(cls)
        for name, value in zip(cls._static_fields, aux, strict=True):
            setattr(obj, name, value)
        for name, value in zip(cls._array_fields, children, strict=True):
            setattr(obj, name, value)
        return obj

    @staticmethod
    def _contract_basis_coef(basis: Array, coef: Array) -> Array:
        return _contract_basis_coef(basis, coef)

    def _compute_basis_and_deriv2(
        self, x: Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        basis = bspline_basis(x, self.knots, self.order) @ self.postmultiply_by
        deriv = bspline_basis_deriv(x, self.knots, self.order) @ self.postmultiply_by
        deriv2 = bspline_basis_deriv2(x, self.knots, self.order) @ self.postmultiply_by
        return basis, deriv, deriv2

    def get_basis(self, x: Array) -> Array:
        """
        Get B-spline basis for input data.

        Parameters
        ----------
        x
            Input data.

        Returns
        -------
        Basis matrix.
        """
        return _approx_basis(self, x)

    def get_basis_and_deriv(self, x: Array) -> tuple[Array, Array]:
        """
        Get basis and first derivative.

        Parameters
        ----------
        x
            Input data.

        Returns
        -------
        basis
            Basis matrix.
        deriv
            First derivative.
        """
        return _approx_basis_and_deriv(self, x)

    def get_basis_and_deriv2(self, x: Array) -> tuple[Array, Array, Array]:
        """
        Get basis, first and second derivative.

        Parameters
        ----------
        x
            Input data.

        Returns
        -------
        basis : Array
            Basis matrix.
        deriv : Array
            First derivative.
        deriv2 : Array
            Second derivative.
        """
        return _approx_basis_deriv_and_deriv2(self, x)

    def dot_n(self, x: Array, coef: Array) -> Array:
        """
        Evaluate spline at given points.

        Parameters
        ----------
        x
            Input data, an array of shape (n,).
        coef
            Spline coefficients.

        Returns
        -------
        Evaluated spline values.
        """
        return _dot(self, x, coef)

    def dot_and_deriv_n(self, x: Array, coef: Array) -> tuple[Array, Array]:
        """
        Evaluate spline and its derivative.

        Parameters
        ----------
        x
            Input data, an array of shape (n,).
        coef
            Spline coefficients.

        Returns
        -------
        value : Array
            Spline values.
        deriv : Array
            Spline derivatives.
        """
        return _dot_and_deriv(self, x, coef)


def _grid_index(approx: BSplineApprox, x: Array) -> Array:
    i = jnp.floor((x - approx.grid[0]) / approx.step).astype(jnp.int32)
    i = jnp.clip(i, 0, approx.ngrid)
    i += (x >= approx.grid[i + 1]).astype(i.dtype)
    i -= (x < approx.grid[i]).astype(i.dtype)
    return jnp.clip(i, 0, approx.ngrid)


def _interpolation_weights(approx: BSplineApprox, x: Array) -> tuple[Array, Array]:
    i = _grid_index(approx, x)
    lo = approx.grid[i]
    k = jnp.expand_dims((x - lo) / (approx.grid[i + 1] - lo), -1)
    return i, k


@jax.jit
def _approx_basis(approx: BSplineApprox, x: Array) -> Array:
    i, k = _interpolation_weights(approx, x)
    return (1.0 - k) * approx.basis_grid[i, :] + (k * approx.basis_grid[i + 1, :])


@jax.jit
def _approx_basis_and_deriv(approx: BSplineApprox, x: Array) -> tuple[Array, Array]:
    """Basis matrix approximation and its derivative with respect to the data."""
    i, k = _interpolation_weights(approx, x)
    basis = (1.0 - k) * approx.basis_grid[i, :] + (k * approx.basis_grid[i + 1, :])
    basis_deriv = (1.0 - k) * approx.basis_deriv_grid[i, :] + (
        k * approx.basis_deriv_grid[i + 1, :]
    )
    return basis, basis_deriv


@jax.jit
def _approx_basis_deriv_and_deriv2(
    approx: BSplineApprox, x: Array
) -> tuple[Array, Array, Array]:
    """Basis matrix approximation and its first two derivatives in the data."""
    i, k = _interpolation_weights(approx, x)
    basis = (1.0 - k) * approx.basis_grid[i, :] + (k * approx.basis_grid[i + 1, :])
    basis_deriv = (1.0 - k) * approx.basis_deriv_grid[i, :] + (
        k * approx.basis_deriv_grid[i + 1, :]
    )
    basis_deriv2 = (1.0 - k) * approx.basis_deriv2_grid[i, :] + (
        k * approx.basis_deriv2_grid[i + 1, :]
    )
    return basis, basis_deriv, basis_deriv2


def _contract_basis_coef(basis: Array, coef: Array) -> Array:
    if jnp.ndim(coef) < 2 or jnp.shape(coef)[-2] != 1:
        raise ValueError("Spline coefficients must have shape (..., 1, n_param).")

    return jnp.einsum("...nj,...j->...n", basis, coef[..., 0, :])


# The custom JVPs give derivatives in the data from the precomputed derivative
# grids. The spline's own arrays are fixed geometry: their tangents are ignored.


@jax.custom_jvp
def _dot_impl(approx: BSplineApprox, x: Array, coef: Array) -> Array:
    return _contract_basis_coef(_approx_basis(approx, x), coef)


@_dot_impl.defjvp
def _dot_jvp(primals, tangents):
    approx, x, coef = primals
    _, x_dot, coef_dot = tangents

    basis, basis_deriv = _approx_basis_and_deriv(approx, x)
    smooth = _contract_basis_coef(basis, coef)
    smooth_deriv = _contract_basis_coef(basis_deriv, coef)

    tangent = smooth_deriv * x_dot + _contract_basis_coef(basis, coef_dot)
    return smooth, tangent


@jax.custom_jvp
def _dot_and_deriv_impl(
    approx: BSplineApprox, x: Array, coef: Array
) -> tuple[Array, Array]:
    """``x`` has shape ``(..., n)`` and ``coef`` has shape ``(..., 1, p)``."""
    basis, basis_deriv = _approx_basis_and_deriv(approx, x)
    return _contract_basis_coef(basis, coef), _contract_basis_coef(basis_deriv, coef)


@_dot_and_deriv_impl.defjvp
def _dot_and_deriv_jvp(primals, tangents):
    approx, x, coef = primals
    _, x_dot, coef_dot = tangents

    basis, basis_deriv, basis_deriv2 = _approx_basis_deriv_and_deriv2(approx, x)
    smooth = _contract_basis_coef(basis, coef)
    smooth_deriv = _contract_basis_coef(basis_deriv, coef)
    smooth_deriv2 = _contract_basis_coef(basis_deriv2, coef)

    tangent_bdot = smooth_deriv * x_dot + _contract_basis_coef(basis, coef_dot)
    tangent_deriv = smooth_deriv2 * x_dot + _contract_basis_coef(basis_deriv, coef_dot)

    return (smooth, smooth_deriv), (tangent_bdot, tangent_deriv)


_dot = jax.jit(_dot_impl)
_dot_and_deriv = jax.jit(_dot_and_deriv_impl)
